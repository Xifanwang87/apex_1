
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

plt.style.use('bmh')
plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['figure.figsize'] = 10,7

pd.set_option('display.max_rows', 100)


# In[2]:


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
from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg, apex__unadjusted_market_data,
                                  fix_security_name_index, get_security_metadata,
                                  get_index_member_weights_on_day)
from sklearn.linear_model import HuberRegressor

from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.universe import ApexCustomRoweIndicesUniverse, ApexUniverseAMNA
from apex.toolz.bloomberg import get_index_member_weights_multiday, get_security_historical_data
from apex.factors.by_universe import universe_size_factor
from apex.toolz.volatility import normalize_vol
from apex.toolz.universe import ApexUniverseAMNA, AMNA_AVAILABILITY_REMOVALS


# In[3]:


import hermes
import hermes.backend.redis

STRATEGY_CACHE = hermes.Hermes(hermes.backend.redis.Backend, ttl = 120, host = '10.15.201.154', db = 9, port=6379)
FACTOR_CACHE = hermes.Hermes(hermes.backend.redis.Backend, ttl = 60*60*24, host = '10.15.201.154', db = 10, port=6379) # a day


# In[4]:


from apex.toolz.deco import lazyproperty
from apex.toolz.itertools import flatten
from apex.factors.by_universe import UNIVERSE_NEUTRAL_FACTORS
from joblib import parallel_backend, Parallel, delayed, parallel
from funcy import partial
from functools import lru_cache
from apex.factors.alpha_momentum import get_alpha_momentum_signal_daily

def r2_score(series):
    return np.corrcoef(series, np.arange(len(series)))[0, 1]**2

def column_rolling_apply_on_cluster(data, fn, window):
    dask = ApexDaskClient()
    columns = data.columns
    data = dask.scatter(data)
    def wrapped(column):
        return data.result()[column].rolling(window).apply(fn)
    result = {}
    for column in columns:
        result[column] = dask.submit(wrapped, column)
    return pd.DataFrame({x: result[x].result() for x in result})


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
        factor_futures = {x: dask.submit(factor_dictionary[x], universe) for x in factor_dictionary}
        factor_result = {x: factor_futures[x].result() for x in factor_dictionary}
        return pd.concat(factor_result, axis=1)
    return compute_universe_factor_scores(universe)


def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def compute_and_plot_signal_returns(model, signal_name):
    signal = model['signals'][signal_name]
    returns = model['market_data']['adjusted']['px_last'].apply(fill_nas_series).dropna(how='all').pct_change().fillna(0)
    tcs = 0.01/model['market_data']['adjusted']['px_last'].apply(fill_nas_series)

    signal_returns = signal.shift(1) * returns.reindex(signal.index) - signal.diff().abs()*tcs
    signal_returns = signal_returns.sum(axis=1)
    print('SR:', signal_returns.mean()/signal_returns.std()*np.sqrt(252))
    amz = apex__adjusted_market_data('AMZ Index', parse=True)['returns']['AMZ Index']
    pd.DataFrame({
        'AMZ': amz,
        'Signal': signal_returns
    }).dropna().loc['2005':].cumsum().plot(figsize=(25, 15))


# In[5]:


Blackboard = lambda: defaultdict(Blackboard)


# In[6]:


from apex.toolz.universe import ApexUniverseAMNA

def instantiate_model(model_name, universe):
    bb = Blackboard()
    base_tickers = universe().tickers
    final_tickers = []
    securities = []
    for ticker in base_tickers:
        try:
            sec = ApexSecurity.from_id(ticker)
            final_tickers.append(ticker)
            securities.append(sec)
        except:
            pass

    bb['name'] = model_name
    bb['universe'] = final_tickers
    bb['securities'] = securities
    bb['security_metadata'] = get_security_metadata(*final_tickers)
    APEX_UNIVERSES[model_name] = final_tickers
    return bb


def load_market_data(model):
    tickers = model['universe']
    model['market_data']['adjusted'] = apex__adjusted_market_data(*tickers, parse=True).loc['1990':]
    model['market_data']['unadjusted'] = apex__unadjusted_market_data(*tickers, parse=True).loc['1990':]

    market_caps = get_security_historical_data(*tickers, fields=['CUR_MKT_CAP'])
    market_caps.columns = market_caps.columns.droplevel(1)
    model['market_caps'] = market_caps
    return model

def compute_availability(model):
    data = model['market_data']['adjusted']['px_last'].apply(fill_nas_series).dropna(how='all')
    availability = (~data.isnull()) & (data.rolling(50).median() > 10)
    for bbid, date in AMNA_AVAILABILITY_REMOVALS.items():
        if bbid in availability.columns:
            availability.loc[date:, bbid] = False
    model['availability'] = availability
    return model

def create_settings(model):
    model['settings']['base_signal'] = {
        'window': 200,
    }
    return model


def compute_base_signal(model, window=200):
    """
    This is the base signal for the apex model.
    """
    data = model['market_data']['adjusted']['returns'].fillna(0)
    volatility = data.ewm(span=252).std().rank(axis=1, pct=True, ascending=True)
    momentum = (data.ewm(span=250).mean() - data.ewm(span=20).mean())
    signal = momentum.rank(axis=1, pct=True, ascending=True)
    signal += volatility
    signal = signal.ewm(span=10).mean()
    signal = signal.rank(axis=1, pct=True)
    signal = signal.divide(signal.abs().sum(axis=1), axis=0)
    model['signals']['base_signal'] = signal
    return model


def compute_short_term_mean_reversion_signal(model):
    returns = model['market_data']['adjusted']['px_last'].apply(fill_nas_series).pct_change().fillna(0)
    monthly_returns = returns.ewm(span=30).mean()/returns.expanding().std()
    signal = monthly_returns[model['availability']]
    signal = signal.rank(axis=1, pct=True)
    signal = rank_signal_from_bbg(-signal, cutoff=0.)
    signal = signal.divide(signal.abs().sum(axis=1), axis=0)
    model['signals']['short_term_mean_reversion'] = signal
    return model


def compute_factor_signal(model, factors=['Growth', 'CFO Yield', 'Liquidity', 'Credit', 'Momentum', 'Size']):
    @UNIVERSE_DATA_CACHING.cache_on_arguments(namespace=f'model["name"]')
    def universe_factor_scores(factor):
        assert factor in UNIVERSE_NEUTRAL_FACTORS
        result = UNIVERSE_NEUTRAL_FACTORS[factor]('AMNA', cutoff=0)
        return result.apply(fill_nas_series)

    def factor_scores():
        RESULT = defaultdict(list)
        for factor_name in factors:
            try:
                factor = universe_factor_scores(factor_name)
                factor = factor.loc[:, ~factor.columns.duplicated()]
                RESULT[factor_name] = factor.apply(fill_nas_series)
            except ValueError:
                print('ValueError in', factor_name)
                continue
        return RESULT
    factor_signal = factor_scores()
    factor_signal['Size'] = -factor_signal['Size']
    factor_signal = pd.concat(factor_signal, axis=1)
    factor_signal = factor_signal.mean(axis=1, level=1)
    factor_weights = factor_signal.apply(fill_nas_series).fillna(0)
    factor_weights = factor_weights[model['availability']].rank(axis=1, pct=True, ascending=True)
    factor_weights = factor_weights.divide(factor_weights.abs().sum(axis=1), axis=0)
    model['signals']['factors'] = factor_weights
    return model



def compute_alpha_strategies(model):
    from apex.factors.alpha_strategies import get_model as alpha__get_model
    alpha_model = alpha__get_model()
    model['alpha_model'] = alpha_model
    #alpha_weights = alpha_model['combined_weights']
    return model

def compute_core_portfolio(model, num_core=5):
    tickers = model['universe']
    market_caps = model['market_caps']
    market_caps = market_caps.apply(fill_nas_series)
    availability = model['availability']
    selection = market_caps.rank(axis=1, ascending=True)
    selection = selection >= num_core
    # If it has been in the group it remains in it
    selection[selection == False] = np.nan
    selection = selection.fillna(method='ffill', limit=252*4).apply(fill_nas_series)
    selection = selection.fillna(False)

    selection_cols = selection.sum() > 252
    selection_cols = sorted(selection_cols[selection_cols].index.tolist())
    selection = selection[selection_cols]
    selection = selection.fillna(False)
    core = selection.copy().fillna(False)
    core = core > 0

    availability = availability & core

    # Base signal
    core_base_signal = model['signals']['base_signal']
    core_mr_signal = model['signals']['short_term_mean_reversion']
    core_factor_signal = model['signals']['factors']

    # Alpha signals
    alpha_model = model['alpha_model']
    alpha_weights = alpha_model['weights'].copy()
    alpha_weights_d = {}
    for c in alpha_weights:
        df = alpha_weights[c].copy()
        df = df.divide(df.abs().sum(axis=1), axis=0)
        df.columns = [ApexSecurity.from_id(x).id for x in df.columns]
        alpha_weights_d[c] = df


    alpha_weights_d['base_signal'] = model['signals']['base_signal']
    alpha_weights_d['factors'] = model['signals']['factors']

    for alpha_name, alpha_df in alpha_weights_d.items():
        alpha_df = alpha_df
    core_signal = pd.concat(alpha_weights_d, axis=1).fillna(0).mean(axis=1, level=1)[core.columns]
    core_wts = core_signal.rank(axis=1, ascending=True, pct=True).fillna(0).rolling(20).mean()[availability].fillna(method='ffill', limit=2)
    core_wts = core_wts.divide(core_wts.abs().sum(axis=1), axis=0)
    core_wts[core_wts > 0.09] = 0.09
    core_wts = core_wts.divide(core_wts.abs().sum(axis=1), axis=0)
    return core_wts

def get_model():
    model = instantiate_model('apex__amna_salient_v1.0a', ApexUniverseAMNA)
    model = load_market_data(model)
    model = compute_availability(model)
    model = create_settings(model)
    model = compute_alpha_strategies(model)
    model = compute_base_signal(model, **model['settings']['base_signal'])
    model = compute_short_term_mean_reversion_signal(model)
    model = compute_factor_signal(model)

    core_wts = compute_core_portfolio(model)
    model['weights'] = core_wts
    return dict(model)