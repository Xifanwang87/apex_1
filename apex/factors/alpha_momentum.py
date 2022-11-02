#!/usr/bin/env python
import pandas as pd

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
import dask.dataframe as dd
import numpy as np
import numba as nb
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
from joblib import Parallel, delayed, parallel_backend

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.store import ApexDataStore
from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import (ApexBloomberg, get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.toolz.sampling import sample_indices, sample_values, ssample
from apex.universe import APEX_UNIVERSES
from apex.alpha.market_alphas import MARKET_ALPHAS
from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.pipelines.covariance import lw_cov, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__adjusted_market_data,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_bloomberg_field_score,
                                         get_market_data, rank_signal_from_bbg)
from apex.pipelines.risk import (compute_account_active_risk_contrib,
                                 compute_account_total_risk_contrib)
from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg,
                                  fix_security_name_index, get_security_metadata,
                                  get_index_member_weights_on_day)
from sklearn.linear_model import HuberRegressor

from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.universe import ApexCustomRoweIndicesUniverse, ApexUniverseAMNA
from apex.toolz.bloomberg import get_index_member_weights_multiday, get_security_historical_data
from apex.factors.by_universe import universe_size_factor
from apex.toolz.volatility import normalize_vol


# In[31]:


from apex.toolz.deco import lazyproperty
from apex.toolz.itertools import flatten
from apex.factors.by_universe import UNIVERSE_NEUTRAL_FACTORS
from joblib import parallel_backend, Parallel, delayed, parallel
from funcy import partial
from functools import lru_cache

def universe_factor_scores(universe):
    def compute_universe_factor_scores(universe, factor_dictionary=UNIVERSE_NEUTRAL_FACTORS.copy()):
        def factor_wrapper(fn):
            def wrapped(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except:
                    return None
            return wrapped
        dask = ApexDaskClient()
        factor_dictionary = {x: factor_wrapper(factor_dictionary[x]) for x in factor_dictionary}
        factor_futures = {x: dask.submit(factor_dictionary[x], universe, cutoff=0.) for x in factor_dictionary}
        factor_result = {x: factor_futures[x].result() for x in factor_dictionary}
        return pd.concat(factor_result, axis=1)
    return compute_universe_factor_scores(universe)


# In[32]:


def compute_betas_for_dt(factor_returns, returns, security, dt):
    data = factor_returns.loc[:dt]
    security_returns = returns[security].loc[:dt]
    security_returns = security_returns[security_returns.abs() > 1e-9].iloc[-252*2:]
    if len(security_returns) <= 30:
        return None
    try:
        return pd.Series(HuberRegressor().fit(data.reindex(security_returns.index), security_returns).coef_, data.columns)
    except:
        return None


# In[118]:


def compute_weekly_signal(factor_df, returns, availability):
    factor_returns = factor_df.shift(1).multiply(returns, axis=1, level=1).sum(axis=1, level=0)
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

from functools import lru_cache

@lru_cache(maxsize=None)
def get_alpha_momentum_signal_daily(universe):
    factor_scores = universe_factor_scores(universe)
    factor_scores['Momentum'] = factor_scores['Momentum'][factor_scores['Momentum'].abs() > 1e-5]
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
    alpha_signal = alpha_signal[~market_data['px_last'].isnull()].fillna(0).ewm(span=7).mean()
    alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
    return alpha_signal


# In[119]:


def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

