# Variables for setting import
import concurrent.futures as cf
import datetime as dt
import logging
import math
import pickle
import re
import time
import typing
import uuid
import warnings
from collections import ChainMap, OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial, reduce
from pathlib import Path
from pprint import pprint
from typing import Mapping, Sequence, TypeVar

import boltons as bs
import dask_ml
import dogpile.cache as dc
import funcy as fy
import inflect
import joblib
import matplotlib.pyplot as plt
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import redis
import scipy as sc
import sklearn as sk
# Other
import toml
import toolz as tz
# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from IPython.core.debugger import set_trace as bp
from joblib import Parallel, delayed, parallel_backend
from scipy.optimize import Bounds, minimize
from toolz import (curry, keyfilter, keymap, partition_all, reduce, valfilter,
                   valmap)

import pyomo as po
import statsmodels.api as sm
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as to
import torch.tensor as tt


##########
## APEX ##
##########
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.toolz.bloomberg import ApexBloomberg
from apex.toolz.bloomberg import apex__adjusted_market_data as apex__amd
from apex.toolz.bloomberg import (fix_tickers_with_bbg, get_correct_ticker,
                                  get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.caches import INDEX_WEIGHTS_CACHING
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.deco import lazyproperty
from apex.toolz.itertools import flatten
from apex.toolz.storage import ApexMinio
from apex.toolz.strings import camelcase_to_snakecase

from .universes import (broad_universe, downstream_universe, enp_universe,
                        equipment_oilfield_svcs_universe, integrateds_universe,
                        midstream_universe)

# coding: utf-8








def get_index_member_weights_multiday(index, start_date=None, end_date=None, freq='Q'):
    """
    Gets index member weights weights
    """
    today = pd.Timestamp.now(tz='America/Chicago')
    if start_date is None:
        start_date = pd.to_datetime('1/1/1996').tz_localize('America/Chicago')
    if end_date is None:
        end_date = today
    if index is None:
        raise NotImplementedError("Benchmark cannot be none.")
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).tz_localize('America/Chicago')
    date_range = pd.date_range(start_date, end_date, freq=freq)
    # Now let's split this into 5, and use dask client to do even better...

    result = []
    pool = ThreadPoolExecutor(max_workers=32)
    target = curry(get_index_member_weights_on_day)(index)
    result = dict(zip(date_range, pool.map(target, date_range)))
    return result


@INDEX_WEIGHTS_CACHING.cache_on_arguments(namespace='index_member_weights_models')
def get_index_member_weights_on_day(index, day):
    if index is None:
        raise NotImplementedError("Benchmark cannot be none.")
    bbg = ApexBloomberg()
    day = pd.to_datetime(day)
    try:
        weights = bbg.reference(index, 'INDX_MWEIGHT_HIST', kwargs={'END_DATE_OVERRIDE': day.strftime('%Y%m%d')})['INDX_MWEIGHT_HIST'][index]
    except TypeError:
        return None

    weights['Index Member'] += ' Equity'
    weights = weights.set_index('Index Member')['Percent Weight'] / 100
    return weights

def get_index_members_multiday(index, start_date='1990-01-01', end_date=None, freq='Q'):
    """
    Gets index members
    """
    result = get_index_member_weights_multiday(index, start_date=start_date, end_date=end_date, freq=freq)
    result = valfilter(lambda x: x is not None, result)
    result = reduce(lambda x, y: x.union(y.index.tolist()), result.values(), set())
    result = sorted(result)
    result = [x for x in result if x is not None]
    return result



def decay_linear(series, window):
    weights = np.arange(1, window + 1)[::-1]
    weights = weights / np.sum(weights)
    return series.rolling(window).apply(lambda x: np.dot(x, weights))

def signed_power(series, a):
    return series.pow(a) * np.sign(series)

def rolling_zscore(x, days):
    return (x - x.rolling(days).mean()) / x.rolling(days).std()

def rank_signal(result):
    result = result.copy()
    result = result.rank(axis=1, pct=True)
    result = result - 0.5
    return result

def scale(df):
    return df.divide(df.abs().sum(axis=1), axis=0)


@nb.jit
def timeseries_rank_fn(x):
    array = np.array(x)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks[-1] + 1


def call_market_alpha(market_alpha, market_data, **kwargs):
    return market_alpha(highs=market_data['px_high'].fillna(method='ffill', limit=2),
                  opens=market_data['px_open'].fillna(method='ffill', limit=2),
                  lows=market_data['px_low'].fillna(method='ffill', limit=2),
                  closes=market_data['px_last'].fillna(method='ffill', limit=2),
                  returns=market_data['returns'].fillna(0),
                  volumes=market_data['px_volume'].fillna(method='ffill', limit=2),
                  **kwargs)


MODEL_CACHE = dc.make_region(key_mangler=lambda key: "apex:v2:cache" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.160',
        'port': 16383,
        'db': 6,
        'redis_expiration_time': 60*60*4,   # 4 hours
    },
)

BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS = [
    'current_ev_to_t12m_ebitda',
    'current_px_to_free_cash_flow',
    'earn_yld_hist',
    'free_cash_flow_yield',
    'pe_ratio',
    'px_to_book_ratio',
    'px_to_cash_flow',
    'px_to_ebitda',
    'px_to_free_cash_flow',
    'px_to_sales_ratio',
    'shareholder_yield',
    'short_int_ratio',
]


@MODEL_CACHE.cache_multi_on_arguments(namespace='fundamental_data:v1', asdict=True)
def apex__fundamental_data(*identifiers):
    result = {}
    def get_data_fn(identifier):
        bbg = ApexBloomberg()
        return bbg.history(identifier, BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS)

    with parallel_backend('threading', n_jobs=40):
        result = Parallel()(delayed(get_data_fn)(i) for i in identifiers)

    result = funcy.zipdict(identifiers, result)
    return result

def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x




@dataclass
class ApexUniverse:
    name: str
    bbg_subgroup_fld: str
    tickers: typing.FrozenSet
    _db_client: typing.Any = field(init=False)
    def __post_init__(self):
        self._db_client = redis.Redis(host='10.15.201.154', port='6379', db=0)
        self['tickers'] = self.tickers

    def mangle_key(self, key):
        return self.name + ':' + key

    def __getitem__(self, key):
        try:
            return pickle.loads(self._db_client.get(self.mangle_key(key)))
        except:
            return None

    def __setitem__(self, key, value):
        return self._db_client.set(self.mangle_key(key), pickle.dumps(value))

    def market_data(self, update=False):
        if not update:
            try:
                result = self['market_data']
                if result is None:
                    return self.market_data(update=True)
            except:
                pass
        self['market_data'] = apex__amd(*self.tickers, parse=True)
        return self['market_data']

    def fundamental_data(self, update=False):
        bbg = ApexBloomberg()
        if not update:
            try:
                result = self['fundamental_data']
                if result is None:
                    return self.fundamental_data(update=True)
            except:
                pass
        fundamental_data = apex__fundamental_data(*self.tickers)
        self['fundamental_data'] = pd.concat([x.apply(fill_nas_series) for x in fundamental_data.values()], axis=1).apply(fill_nas_series)
        return self['fundamental_data']

    @lazyproperty
    def availability_high_liquidity(self):
        market_data = self.market_data()
        availability = market_data['px_last'].apply(fill_nas_series).rolling(250).median() > 10
        availability = availability & ((market_data['px_volume'] * market_data['px_last']).apply(fill_nas_series).rolling(250).mean() > 1000000)
        return availability

    @property
    def subuniverses(self):
        subgroup_field = self.bbg_subgroup_fld
        tickers = self.tickers
        bbg = ApexBloomberg()
        subgroups = bbg.reference(tickers, subgroup_field)[subgroup_field]
        subgroups = subgroups.fillna('Other')
        subgroup_count = subgroups.groupby(subgroups).count().sort_values(ascending=False)
        subgroup_count = subgroup_count[subgroup_count >= 4]

        subuniverses = {}
        for group in subgroup_count.index:
            subuniverse_name = re.sub('[^0-9a-zA-Z\w]+', '_', group).lower()
            subuniverse_name = self.name + ':' + subuniverse_name
            tickers = subgroups[subgroups == group].index.tolist()
            subuniverses[subuniverse_name] = ApexUniverse(name=subuniverse_name, tickers=frozenset(tickers), bbg_subgroup_fld='INDUSTRY_SUBGROUP')
        return subuniverses

@dataclass
class ApexIndexUniverse:
    def __post_init__(self):
        self._db_client = redis.Redis(host='10.15.201.160', port='16383', db=0)

    def mangle_key(self, key):
        return self.name + ':' + key

    def __getitem__(self, key):
        try:
            return pickle.loads(self._db_client.get(self.mangle_key(key)))
        except:
            return None

    def __setitem__(self, key, value):
        return self._db_client.set(self.mangle_key(key), pickle.dumps(value))

    @property
    def subgroups(self):
        subgroup_field = self.bbg_subgroup_fld
        tickers = self.tickers
        bbg = ApexBloomberg()
        groups = bbg.reference(tickers, subgroup_field)[subgroup_field]
        return groups

    @property
    def subuniverses(self):
        subgroup_field = self.bbg_subgroup_fld
        tickers = self.tickers
        bbg = ApexBloomberg()
        subgroups = bbg.reference(tickers, subgroup_field)[subgroup_field]
        subgroups = universe.subgroups.fillna('Other')
        subgroup_count = subgroups.groupby(subgroups).count().sort_values(ascending=False)
        subgroup_count = subgroup_count[subgroup_count >= 4]

        subuniverses = {}
        for group in subgroup_count.index:
            subuniverse_name = re.sub('[^0-9a-zA-Z\w]+', '_', group).lower()
            subuniverse_name = universe.name + ':' + subuniverse_name
            tickers = subgroups[subgroups == group].index.tolist()
            subuniverses[subuniverse_name] = ApexUniverse(name=subuniverse_name, tickers=frozenset(tickers), bbg_subgroup_fld='INDUSTRY_SUBGROUP')
        return subuniverses

    @property
    def tickers(self):
        if self['tickers'] is None:
            self.update()
        return self['tickers']

    def update(self):
        bbg = ApexBloomberg()
        tickers = get_universe_with_indices(*self.indices)
        self['tickers'] = tickers


    def market_data(self, update=False):
        if not update:
            try:
                result = self['market_data']
                if result is None:
                    return self.market_data(update=True)
            except:
                pass
        self['market_data'] = apex__amd(*self.tickers, parse=True)
        return self['market_data']

    def fundamental_data(self, update=False):
        bbg = ApexBloomberg()
        if not update:
            try:
                result = self['fundamental_data']
                if result is None:
                    return self.fundamental_data(update=True)
            except:
                pass
        fundamental_data = apex__fundamental_data(*self.tickers)
        self['fundamental_data'] = pd.concat([x.apply(fill_nas_series) for x in fundamental_data.values()], axis=1).apply(fill_nas_series)
        return self['fundamental_data']


    @lazyproperty
    def availability_high_liquidity(self):
        market_data = self.market_data()
        availability = market_data['px_last'].apply(fill_nas_series).rolling(250).median() > 10
        availability = availability & ((market_data['px_volume'] * market_data['px_last']).apply(fill_nas_series).rolling(250).mean() > 1000000)
        return availability


@dataclass
class ApexStrategyUniverse:
    name: str
    value: typing.Mapping

    @staticmethod
    def parse_dict(data: typing.Mapping):
        result = OrderedDict()

        for k, v in data.items():
            if isinstance(v, typing.Mapping):
                v = ApexStrategyUniverse.from_dict(k, v, path=None)
                result[k] = v
            else:
                result[k] = ApexConcreteUniverse(name=k, tickers=frozenset(v))
        return result

    @classmethod
    def from_dict(cls, name: str, data: typing.Mapping, path: typing.Any = None):
        result = ApexStrategyUniverse.parse_dict(data)
        return cls(name=name, value=result, path=path)

    @property
    def universes(self):
        return self.value

def setup_directory_structure(universe_name, ds):
    base_dir = Path('/apex.data/apex.portfolios/') / ds / universe_name
    base_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in ['master', 'raw_alpha', 'raw_data', 'staging']:
        (base_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    return True





def get_universe_with_indices(*indices):
    pool = ThreadPoolExecutor()
    result = flatten(pool.map(get_index_members_multiday, indices))
    result = sorted(set(result))
    result = pool.map(get_correct_ticker, result)
    return sorted(set(result))


def get_custom_mlp_subuniverses():
    minio = ApexMinio()
    data = pd.read_excel(minio.get('universe', 'custom_subuniverses.xlsx'), sheet_name='Indices').dropna(axis=1, how='all')
    result = {
        k: data[k].dropna().tolist() for k in data.columns
    }
    tickers = sorted(set(reduce(lambda x, y: x + y, result.values(), [])))

    tickers = funcy.zipdict(tickers, fix_tickers_with_bbg(*tickers).values())
    result = {
        k: [tickers[x] for x in result[k] if x in tickers] for k in result
    }
    result['e&p'] = result['EnP']
    del result['EnP']
    result = keymap(camelcase_to_snakecase, result)
    del result['other']
    return result


@dataclass
class ApexBroadEnergyUniverse(ApexIndexUniverse):
    name: str = field(default='apex:universe:broad_energy')
    bbg_subgroup_flds = ['INDUSTRY_SUBGROUP', 'GICS_SUB_INDUSTRY_NAME', 'BICS_LEVEL_4_SUB_INDUSTRY_NAME']
    indices: frozenset = field(default=frozenset([
        'S15IOIL Index',
        'S5IOIL Index',
        'SYIOIL Index',
        'R2GOIDM Index',
        'S15OILE Index',
        'S5OILE Index',
        'SPSIOS Index',
        'DJSOES Index',
        'S15OILP Index',
        'S5OILP Index',
        'SPSIOP Index',
        'S12OILP Index',
        'S15OILR Index',
        'S4OILR Index',
        'S12OILR Index',
        'SOILR Index',
        'AMNA Index',
        'AMZ Index',
        'AMEI Index',
        'AMNA Index',
        'AMZ Index',
    ]))

    @property
    def subuniverses(self):
        subuniverse_tickers = {
            'integrateds': integrateds_universe(),
            'oilfield_svcs_and_equipment': equipment_oilfield_svcs_universe(),
            'midstream': midstream_universe(),
            'enp': enp_universe(),
            'downstream': downstream_universe(),
            'broad_energy': broad_universe(),
        }

        subuniverse_tickers['universe'] = self.tickers
        subuniverses = {}
        for subuniverse_name in subuniverse_tickers:
            tickers = subuniverse_tickers[subuniverse_name]
            tickers = [ApexSecurity.from_id(x).parsekyable_des for x in tickers]
            subuniverse_name = self.name + ':subuniverse:' + subuniverse_name
            subuniverses[subuniverse_name] = ApexUniverse(name=subuniverse_name, tickers=frozenset(tickers), bbg_subgroup_fld='INDUSTRY_SUBGROUP')
        return subuniverses


TRANSFORMS = {
    'identity': lambda x: x,
    'identity_lagged': lambda x: x.shift(1),
    'ewm2': lambda x: x.ewm(span=2).mean(),
    'ewm5': lambda x: x.ewm(span=5).mean(),
    'ewm10': lambda x: x.ewm(span=10).mean(),
    'ewm15': lambda x: x.ewm(span=15).mean(),
    'ewm20': lambda x: x.ewm(span=20).mean(),
    'ewm40': lambda x: x.ewm(span=40).mean(),
    'ewm100': lambda x: x.ewm(span=100).mean(),
}

def vol_target_wrapper(fn, availability, returns):
    def new_fn(*args, **kwargs):
        alpha_res = fn(*args, **kwargs)[availability].rank(axis=1)
        alpha_res = alpha_res.subtract(alpha_res.mean(axis=1), axis=0)
        alpha_res = alpha_res.divide(alpha_res.abs().sum(axis=1), axis=0)
        alpha_returns = (alpha_res.shift(1) * returns).sum(axis=1)
        return alpha_res * 0.1/(alpha_returns.std()*np.sqrt(252))
    return new_fn

def compute_alpha_and_save(alpha_name, alpha_fn, market_data, availability, base_path):
    try:
        alpha = call_market_alpha(alpha_fn, market_data)
    except:
        print('Failure computing alpha', alpha_name)
        import traceback
        traceback.print_exc()
        return True
    returns = market_data['returns']
    std = returns.std()
    base_path = Path(base_path)
    pool = ThreadPoolExecutor()
    futs = []
    for transform_name, transform in TRANSFORMS.items():
        transform = vol_target_wrapper(transform, availability, returns)

        alpha_signal = transform(alpha.fillna(0))
        alpha_signal = alpha_signal[availability]
        alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
        strategy_returns = (alpha_signal.shift(2) * returns)

        sign = 1
        if strategy_returns.sum(axis=1).mean() < 0:
            sign = -1
        alpha_signal *= sign

        strategy_returns = (alpha_signal.shift(2) * returns)
        strategy_returns_tc = strategy_returns - alpha_signal.shift(2).diff().abs() * 0.0025
        strategy_returns = strategy_returns.loc['2000' :].sum(axis=1)
        strategy_returns_tc = strategy_returns_tc.loc['2000' :].sum(axis=1)
        start_year = pd.Timestamp.now() - pd.DateOffset(years=5)
        start_year = start_year.strftime('%Y')

        if strategy_returns_tc.loc[start_year:].mean() > 0.:
            fname = alpha_name + '_' + transform_name + '_5y.pq'
            fut = pool.submit(alpha_signal.to_parquet, base_path / 'raw_alpha' / fname)
            futs.append(fut)

        alpha_signal = transform(alpha.fillna(0))
        alpha_signal = alpha_signal[availability]
        alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
        strategy_returns = (alpha_signal.shift(2) * returns)

        sign = 1
        if strategy_returns.sum(axis=1).mean() < 0:
            sign = -1
        alpha_signal *= sign

        strategy_returns = (alpha_signal.shift(2) * returns)
        strategy_returns_tc = strategy_returns - alpha_signal.shift(2).diff().abs() * 0.0025
        strategy_returns = strategy_returns.loc['2000' :].sum(axis=1)
        strategy_returns_tc = strategy_returns_tc.loc['2000' :].sum(axis=1)
        start_year = pd.Timestamp.now() - pd.DateOffset(years=15)
        start_year = start_year.strftime('%Y')
        sr_tc = strategy_returns_tc.loc[start_year:].mean()

        if sr_tc > 0.:
            fname = alpha_name + '_' + transform_name + '_10y.pq'
            fut = pool.submit(alpha_signal.to_parquet, base_path / 'raw_alpha' / fname)
            futs.append(fut)

        alpha_signal = transform(alpha.fillna(0))
        alpha_signal = alpha_signal[availability]
        alpha_signal = alpha_signal.divide(std)
        alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
        strategy_returns = (alpha_signal.shift(2) * returns)
        if strategy_returns.sum(axis=1).mean() < 0:
            sign = -1
        alpha_signal *= sign
        strategy_returns = (alpha_signal.shift(2) * returns)

        strategy_returns_tc = strategy_returns - alpha_signal.shift(2).diff().abs() * 0.0025
        strategy_returns = strategy_returns.loc['2000' :].sum(axis=1)
        strategy_returns_tc = strategy_returns_tc.loc['2000' :].sum(axis=1)
        start_year = pd.Timestamp.now() - pd.DateOffset(years=15)
        start_year = start_year.strftime('%Y')
        sr_tc = strategy_returns_tc.loc[start_year:].mean()
        if sr_tc > 0.:
            fname = alpha_name + '_' + transform_name + '_vol_adj.pq'
            fut = pool.submit(alpha_signal.to_parquet, base_path / 'raw_alpha' / fname)
            futs.append(fut)


        alpha_signal = transform(alpha.fillna(0))
        alpha_signal = alpha_signal[availability]
        alpha_signal = alpha_signal.divide(std)
        alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
        strategy_returns = (alpha_signal.shift(2) * returns)
        if strategy_returns.sum(axis=1).mean() < 0:
            sign = -1
        alpha_signal *= sign
        strategy_returns = (alpha_signal.shift(2) * returns)

        strategy_returns_tc = strategy_returns - alpha_signal.shift(2).diff().abs() * 0.0025
        strategy_returns = strategy_returns.loc['2000' :].sum(axis=1)
        strategy_returns_tc = strategy_returns_tc.loc['2000' :].sum(axis=1)
        start_year = pd.Timestamp.now() - pd.DateOffset(years=5)
        start_year = start_year.strftime('%Y')
        sr_tc = strategy_returns_tc.loc[start_year:].mean()
        if sr_tc > 0.:
            fname = alpha_name + '_' + transform_name + '_5y_vol_adj.pq'
            fut = pool.submit(alpha_signal.to_parquet, base_path / 'raw_alpha' / fname)
            futs.append(fut)

    return True
