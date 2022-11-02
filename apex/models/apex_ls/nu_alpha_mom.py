# Variables for setting import
import concurrent.futures as cf
import datetime
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

# Other
import toml

# Default imports
import numpy as np
import numba as nb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import pyomo as po
import numba as nb
import scipy as sc
import sklearn as sk
import statsmodels.api as sm
import toolz as tz
import funcy as fy
import boltons as bs
import dogpile.cache as dc
import matplotlib.pyplot as plt
import torch.optim as to
import torch
import torch.tensor as tt
import torch.nn as tnn
import torch.nn.functional as F


# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from IPython.core.debugger import set_trace as bp
from scipy.optimize import Bounds, minimize
from toolz import partition_all

import dask_ml
import inflect
import joblib

##########
## APEX ##
##########
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.toolz.bloomberg import ApexBloomberg, get_security_metadata, apex__adjusted_market_data as apex__amd, apex__adjusted_market_data
from apex.toolz.dask import ApexDaskClient, compute_delayed
from joblib import Parallel, delayed, parallel_backend

def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def get_returns(x):
    x = list(x)
    return apex__amd(*x, parse=True)['returns']
from apex.factors.by_universe import apex__universe_bloomberg_field_factor, apex__universe_bloomberg_fundamental_field_score
from apex.pipelines.factor_model import get_market_data

def rank_signal(result, cutoff=0.75):
    result = result.rank(axis=1)
    max_rank = result.max(axis=1)
    result = result.divide(max_rank, axis=0)
    result = result.subtract(result.mean(axis=1), axis=0)
    result = result * 2
    result[result.abs() < cutoff] = 0
    return result.dropna(how='all')


def compute_signals(universe):
    volume = apex__universe_bloomberg_field_factor(universe, 'PX_VOLUME') * apex__universe_bloomberg_field_factor(universe, 'PX_LAST')
    volume_signal = volume.fillna(0).rolling(20).sum()/volume.fillna(0).rolling(252).sum()
    returns = get_market_data('AMNA')['returns'].fillna(0)
    short_term_mr = returns.ewm(span=20).mean()
    momentum = returns.ewm(span=250).mean()
    momentum_chg = returns.ewm(span=250).mean().diff(20)
    volatility = returns.ewm(span=250).std().diff(20)
    SIGNAL_DF = {
        'pe_ratio': rank_signal(apex__universe_bloomberg_field_factor(universe, 'PE_RATIO'), cutoff=0.75).apply(fill_nas_series),
        'px_to_sales': rank_signal(apex__universe_bloomberg_field_factor(universe, 'PX_TO_SALES_RATIO'), cutoff=0.75).apply(fill_nas_series),
        'eps_revisions': rank_signal(apex__universe_bloomberg_field_factor(universe, 'PX_TO_SALES_RATIO'), cutoff=0.75).apply(fill_nas_series),
        'volume_20d': rank_signal(volume_signal, cutoff=0.75).apply(fill_nas_series),
        'trail_12m_net_sales': apex__universe_bloomberg_fundamental_field_score(universe, 'TRAIL_12M_NET_SALES', cutoff=0.75).apply(fill_nas_series),
        'asset_turnover': apex__universe_bloomberg_fundamental_field_score(universe, 'ASSET_TURNOVER', cutoff=0.75).apply(fill_nas_series),
        'short_term_mr': rank_signal(short_term_mr, cutoff=0.75).apply(fill_nas_series),
        'momentum': rank_signal(momentum, cutoff=0.75).apply(fill_nas_series),
        'vol_change': rank_signal(volatility, cutoff=0.75).apply(fill_nas_series),
        'mom_change': rank_signal(momentum_chg, cutoff=0.75).apply(fill_nas_series),
    }
    SIGNAL_DF = pd.concat(SIGNAL_DF, axis=1)
    return SIGNAL_DF


def compute_weekly_signal(factor_df, returns, availability):
    factor_returns = factor_df.shift(1).multiply(returns, axis=1, level=1).sum(axis=1, level=0)
    factor_returns += factor_df.shift(2).multiply(returns, axis=1, level=1).sum(axis=1, level=0)
    factor_returns += factor_df.ewm(span=3).mean().shift(2).multiply(returns, axis=1, level=1).sum(axis=1, level=0)
    factor_returns = factor_returns / 3
    available = availability.sum(axis=1)
    dates = available[available > 20].iloc[104:].index.tolist()
    securities = sorted(set(returns.columns))
    result = {}
    dask = ApexDaskClient()
    factor_returns_sc = dask.scatter(factor_returns)
    returns_sc = dask.scatter(returns)
    result = {}
    job_futures = []
    job_count = 0
    cur_jobs = []
    for dt in dates:
        for security in securities:
            job_futures.append(dask.submit(compute_betas_for_dt, factor_returns_sc, returns_sc, security, dt))
            cur_jobs.append((security, dt))
            job_count += 1

        if len(job_futures) > 10000 or dt == dates[-1]:
            job_futures = dask.gather(job_futures)
            result.update(dict(zip(cur_jobs, job_futures)))
            cur_jobs = []
            job_futures = []

    for job in cur_jobs:
        result[job] = result[job].result()
    return result


def compute_betas_for_dt(factor_returns, returns, security, dt):
    from sklearn.linear_model import HuberRegressor
    data = factor_returns.loc[:dt]
    security_returns = returns[security].loc[:dt]
    security_returns = security_returns[security_returns.abs() > 1e-9].iloc[-252:]
    if len(security_returns) < 20:
        return None
    return pd.Series(HuberRegressor().fit(data.reindex(security_returns.index), security_returns).coef_, data.columns)

def get_new_alpha_momentum_signal_daily(universe):
    factor_scores = compute_signals(universe).loc['2000':]
    market_data = apex__adjusted_market_data(*sorted(set(factor_scores.columns.get_level_values(1))), parse=True)
    returns = market_data['returns'].fillna(0)

    result = compute_weekly_signal(factor_scores, returns, (market_data['px_last'].resample('W').first() > 5).fillna(False))
    security_betas = pd.DataFrame(result).T.unstack(level=0)
    security_betas = security_betas.reindex(returns.index).fillna(method='ffill', limit=5)
    security_factor_returns = security_betas.multiply(returns, level=1).sum(axis=1, level=1)
    security_alphas = returns - security_factor_returns

    #alpha_signal = security_alphas.fillna(0).rank(axis=1, pct=True, ascending=False) - 0.5
    alpha_signal = security_alphas.fillna(0).ewm(span=5).mean().rank(axis=1, pct=True, ascending=False) - 0.5
    alpha_signal += security_alphas.fillna(0).ewm(span=3*7).mean().rank(axis=1, pct=True, ascending=False)
    alpha_signal -= security_alphas.fillna(0).ewm(span=252).mean().rank(axis=1, pct=True, ascending=False)
    alpha_signal = alpha_signal[~market_data['px_last'].isnull()]
    alpha_signal = (alpha_signal.rank(axis=1, pct=True) - 0.5) * 2
    alpha_signal[alpha_signal.abs() < 0.] = 0
    alpha_signal = alpha_signal[~market_data['px_last'].isnull()].fillna(0)
    alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
    return alpha_signal.fillna(0).rolling(7).mean()