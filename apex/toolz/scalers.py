import numpy as np
import pandas as pd
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

from dataclasses import dataclass, field
from apex.toolz.dask import ApexDaskClient
import typing


SCALERS = {
    'robust': RobustScaler,
    'standard': StandardScaler,
    'normalizer': Normalizer,
    'maxabs': MaxAbsScaler,
    'minmax': lambda: MinMaxScaler(feature_range=(-1, 1)),
    'positive_minmax': lambda: MinMaxScaler(feature_range=(0, 1)),
    'quantile': lambda: QuantileTransformer(n_quantiles=10),
}

def scale_fn_creator(scaler, min_periods=35):
    try:
        scaler = scaler(with_centering=False)
    except:
        try:
            scaler = scaler(with_mean=False)
        except:
            scaler = scaler()

    def scale_fn(data):
        result = scaler.fit_transform(data.reshape(-1, 1))
        return result[-1]

    def mp_fn(data):
        data = data.expanding(min_periods=min_periods).apply(scale_fn)
        data = data - data.mean()
        return data

    def scaler_callable(data):
        dask = ApexDaskClient()
        result = {}
        for c in data.columns:
            result[c] = dask.submit(mp_fn, data[c])
        result_df = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
        for c in data.columns:
            result_df[c] = result[c].result()
        return result_df
    return scaler_callable


def cs_scale_fn_creator(scaler):
    scaler = scaler()
    def scaler_callable(data: pd.DataFrame):
        data = pd.DataFrame(scaler.fit_transform(data.T), columns=data.index, index=data.columns).T
        return data
    return scaler_callable


CROSS_SECTIONAL_SCALERS = {k: cs_scale_fn_creator(SCALERS[k]) for k in SCALERS}
COLUMN_SCALERS = {k: scale_fn_creator(SCALERS[k]) for k in SCALERS}

@dataclass
class DataScaler:
    columns: str = field(default=None)
    rows: str = field(default=None)
    min_periods: int = field(default=30)
    column_first: bool = field(default=False)
    normalize: bool = field(default=True)
    _column_scaler: typing.Any = field(default=None)
    _row_scaler: typing.Any = field(default=None)
    def __post_init__(self):
        assert self.columns in SCALERS or self.columns is None
        assert self.rows in SCALERS or self.rows is None
        if self.columns is not None:
            self._column_scaler = scale_fn_creator(SCALERS[self.columns], min_periods=self.min_periods)
        if self.rows is not None:
            self._row_scaler = cs_scale_fn_creator(SCALERS[self.rows])

    def fit_transform(self, data):
        pipe = []
        if self.columns is not None:
            pipe.append(self._column_scaler)
        if self.rows is not None:
            pipe.append(self._row_scaler)
        if not self.column_first:
            pipe = list(reversed(pipe))
        if self.normalize:
            pipe.append(cs_scale_fn_creator(SCALERS['minmax']))
        for fn in pipe:
            data = fn(data)
        return data