from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import (OAS, GraphicalLassoCV, MinCovDet,
                                graphical_lasso, ledoit_wolf,
                                ledoit_wolf_shrinkage, oas)
from toolz import juxt, partition_all

import joblib
from apex.store import ApexDataStore
from apex.toolz.dask import ApexDaskClient
from apex.toolz.deco import timed
from apex.toolz.dicttools import keys, values
from funcy import partial, zipdict
from joblib import Parallel, delayed



#@timed
def min_cov_det(X):
    result = MinCovDet(assume_centered=True).fit(X.fillna(0).values)
    return pd.DataFrame(result.covariance_, X.columns, X.columns)

#@timed
def lw_cov(x):
    return pd.DataFrame(ledoit_wolf(x.fillna(0).values, assume_centered=True)[0], columns=x.columns, index=x.columns)

#@timed
def oas_cov(x):
    return pd.DataFrame(oas(x.fillna(0).values, assume_centered=True)[0], columns=x.columns, index=x.columns)

#@timed
def graphlasso_cv(x, cv=5):
    model = GraphicalLassoCV(cv=cv)
    model.fit(x.fillna(0))
    return pd.DataFrame(oas(model.covariance_, assume_centered=True)[0], columns=x.columns, index=x.columns)

covariances = {
    'lw': lw_cov,
    'oas': oas_cov,
    'gl': graphlasso_cv
}

def default_covariance(returns):
    result = None
    cov_pipe = [
        lw_cov,
        lambda x: lw_cov(x.iloc[-252:]),
        oas_cov,
        min_cov_det,
        lambda x: x.cov()
    ]
    for f in cov_pipe:
        fn = tz.excepts(Exception, f)
        result = fn(returns)
        if result is not None:
            return result
    raise ValueError

cov_fns = values(covariances)

def compute_covariances(data):
    result = {}
    for covariance in keys(covariances):
        result[covariance] = covariances[covariance]
    return zipdict(keys(covariances), juxt(*cov_fns)(data))


def compute_covariance_matrix_for_date_with_fn(fn, returns, date):
    date = pd.to_datetime(date)
    returns_cov = returns.loc[:date.date()]
    returns_cov[returns_cov.abs() < 1e-10] = np.nan
    returns_cov = returns.iloc[-252*3:].dropna(how='all').iloc[:-1] # At most 3yrs of data, remove last
    returns_cov = returns_cov.rolling(2).mean()
    returns_cov = returns_cov.dropna(thresh=252, axis=1).fillna(0)  # At least 1 year of data to compute it
    if len(returns_cov) == 0:
        return None
    result = fn(returns_cov)
    return result

FNS_BY_KIND = {
    'lw': partial(compute_covariance_matrix_for_date_with_fn, lw_cov),
    'oas': partial(compute_covariance_matrix_for_date_with_fn, oas_cov),
    'gl': partial(compute_covariance_matrix_for_date_with_fn, graphlasso_cv),
}


def compute_covariance_matrix_for_date(returns, date):
    date = pd.to_datetime(date)
    returns_cov = returns.loc[:date.date()]
    returns_cov[returns_cov.abs() < 1e-10] = np.nan
    returns_cov = returns.iloc[-252*3:].dropna(how='all').iloc[:-1] # At most 3yrs of data, remove last
    returns_cov = returns_cov.rolling(2).mean()
    returns_cov = returns_cov.dropna(thresh=252, axis=1).fillna(0)  # At least 1 year of data to compute it
    if len(returns_cov) == 0:
        return None
    result = compute_covariances(returns_cov)
    return result

def save_covariance_matrices(compute_date, result):
    compute_date = pd.to_datetime(compute_date)
    ds_nodash = compute_date.strftime('%Y%m%d')
    for kind, v in result.items():
        loc = f'/apex.data/covariance_matrix/{ds_nodash}.{kind}.parquet'
        v.to_parquet(loc, engine='fastparquet', compression='gzip')
    return True

def compute_and_save_covariance(returns, date, kind='all'):
    if kind == 'all':
        return save_covariance_matrices(date, compute_covariance_matrix_for_date(returns, date))
    else:
        return save_covariance_matrices(date, FNS_BY_KIND[kind](returns, date))


def apex_build_and_save_covariance_for_dates(returns, dates, kind='all', n_jobs=25):
    client = ApexDaskClient()
    with joblib.parallel_backend("dask", scatter=[returns]):
        result = Parallel(n_jobs=n_jobs)(delayed(compute_and_save_covariance)(returns, date, kind=kind) for date in dates)
    return result

def apex_covariance_compute(kind='all'):
    store = ApexDataStore()
    returns = store.returns
    base_days = returns.index.tolist()
    days = []
    for day in base_days:
        if kind == 'all':
            loc = f'/apex.data/covariance_matrix/{day.strftime("%Y%m%d")}.lw.parquet'
        else:
            loc = f'/apex.data/covariance_matrix/{day.strftime("%Y%m%d")}.{kind}.parquet'
        if Path(loc).exists():
            continue
        days.append(day)

    days = sorted(days)
    results = apex_build_and_save_covariance_for_dates(returns, days, kind=kind)
    return results


