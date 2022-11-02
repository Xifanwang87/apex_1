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
from functools import lru_cache, partial, reduce
from pathlib import Path
from pprint import pprint
from typing import Mapping, Sequence, TypeVar

import matplotlib.pyplot as plt
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
import toolz as tz
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as to
import torch.tensor as tt
from IPython.core.debugger import set_trace as bp
from scipy.optimize import Bounds, minimize
from sklearn.linear_model import HuberRegressor
from toolz import partition_all

import boltons as bs
import dask_ml
import dogpile.cache as dc
import funcy as fy
import inflect
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import statsmodels.api as sm
# Other
import toml
from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.alpha.market_alphas import MARKET_ALPHAS
##########
## APEX ##
##########
##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.factors.alpha_momentum import get_alpha_momentum_signal_daily
from apex.factors.by_universe import (UNIVERSE_NEUTRAL_FACTORS,
                                      universe_size_factor)
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__adjusted_market_data,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_bloomberg_field_score,                                         construct_alpha_portfolio_from_signal,
                                         get_market_data, max_abs_scaler,
                                         min_max_scaler, rank_signal_from_bbg)
from apex.pipelines.risk import (compute_account_active_risk_contrib,
                                 compute_account_total_risk_contrib)
from apex.security import ApexSecurity
from apex.store import ApexDataStore
from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import (ApexBloomberg, apex__adjusted_market_data,
                                  apex__unadjusted_market_data,
                                  fix_security_name_index,
                                  get_index_member_weights_multiday,
                                  get_index_member_weights_on_day,
                                  get_security_fundamental_data,
                                  get_security_historical_data,
                                  get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.deco import lazyproperty
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.toolz.itertools import flatten
from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.sampling import sample_indices, sample_values, ssample
from apex.toolz.universe import (AMNA_AVAILABILITY_REMOVALS,
                                 ApexCustomRoweIndicesUniverse,
                                 ApexUniverseAMNA)
from apex.toolz.volatility import normalize_vol
from apex.universe import APEX_UNIVERSES
# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from funcy import partial
from joblib import Parallel, delayed, parallel, parallel_backend
from apex.system.v11.backtest import apex__compute_strategy_returns

# In[3]:



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



Blackboard = lambda: defaultdict(Blackboard)


def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def universe():
    curr_amna_members = pd.DataFrame(get_index_member_weights_multiday('AMNA Index', start_date='2018-01-01', freq='Q')).index.tolist()
    historical_amna_members = ApexUniverseAMNA().tickers
    universe = historical_amna_members + curr_amna_members
    universe = sorted(set([ApexSecurity.from_id(x).id for x in universe]))
    return universe

def instantiate_model(model_name, universe):
    bb = Blackboard()
    base_tickers = universe()
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
    return model

def compute_availability(model):
    data = model['market_data']['adjusted']['px_last'].apply(fill_nas_series).dropna(how='all')
    for bbid, date in AMNA_AVAILABILITY_REMOVALS.items():
        data.loc[date:, bbid] = np.nan
    availability = ~data.isnull()
    availability = availability & (data > 10)
    model['availability'] = availability
    return model

def create_settings(model):
    model['settings']['base_signal'] = {
        'window': 200,
    }
    return model


def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def alpha_wrapper(fn):
    def wrapped(market_data):
        returns = market_data['returns'].fillna(0)
        market_data =  market_data[['px_open', 'px_high', 'px_low', 'px_last', 'px_volume']].apply(fill_nas_series)
        return fn(opens=market_data['px_open'],
                  highs=market_data['px_high'],
                  lows=market_data['px_low'],
                  closes=market_data['px_last'],
                  returns=returns,
                  volumes=market_data['px_volume'])
    return wrapped


def get_raw_alpha_signals():
    model = instantiate_model('apex__amna_salient_v1.0a', universe)
    model = load_market_data(model)
    model = compute_availability(model)
    model = create_settings(model)

    market_data = model['market_data']['adjusted']

    dask = ApexDaskClient()
    market_data_sc = dask.scatter(market_data)
    returns = market_data['returns']

    # Send the signal computing to cluster
    sharpe_ratio = lambda x: x.mean()/x.std()*np.sqrt(252)

    SIGNAL_RESULT = {}
    SIGNAL_FUTURES = {}
    market_alphas = []

    for market_alpha in MARKET_ALPHAS:
        market_alpha_name = market_alpha.__name__
        market_alpha = alpha_wrapper(market_alpha)
        market_alphas.append(market_alpha)
        alpha_signal = dask.submit(market_alpha, market_data_sc)
        SIGNAL_FUTURES[market_alpha_name] = alpha_signal


    def get_short_term_reversal(market_data):
        returns = market_data['returns']
        monthly_returns = returns.ewm(span=20).mean()
        signal = monthly_returns.rank(axis=1)
        signal = rank_signal_from_bbg(-signal, cutoff=0)
        return signal

    weights = model['weights']
    weights['sr_reversal'] = get_short_term_reversal(market_data)
    weights['alpha_momentum'] = get_alpha_momentum_signal_daily('AMNA')


    std = market_data['returns'].std()
    normalizing_factor = std.sum()/std
    normalizing_factor = normalizing_factor / normalizing_factor.sum()

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    # Aggregate results
    for i in range(20):
        for_deletion = set()
        for alpha_name, alpha_fut in SIGNAL_FUTURES.items():
            if not alpha_fut.done():
                continue
            try:
                alpha_signal = alpha_fut.result()
            except:
                for_deletion.add(alpha_name)
                continue

            alpha_signal = alpha_signal.fillna(0)[model['availability']]
            alpha_signal = alpha_signal.rank(axis=1, pct=True) * 2 - 1.0
            alpha_signal = alpha_signal * normalizing_factor
            alpha_signal = alpha_signal.ewm(span=alpha_slowdown).mean()
            alpha_signal = alpha_signal[model['availability']]
            alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
            strategy_returns = apex__compute_strategy_returns(market_data, alpha_signal, 0)

            sign = 1
            if strategy_returns.mean() < 0:
                strategy_returns = -strategy_returns
                sign = -1
            strategy_returns_tc = apex__compute_strategy_returns(market_data, alpha_signal, 5)
            strategy_returns = strategy_returns.loc['2000' :]
            try:
                strategy_returns_tc = strategy_returns_tc.loc['2000':]
                SIGNAL_RESULT[alpha_name] = {
                    'returns': strategy_returns,
                    'returns_tc': strategy_returns_tc,
                    'sharpe_tc': sharpe_ratio(strategy_returns_tc.loc['2017':]),
                    'sharpe_no_tc': sharpe_ratio(strategy_returns),
                    'signal': alpha_signal,
                    'sign': sign
                }
                for_deletion.add(alpha_name)
            except:
                for_deletion.add(alpha_name)
                continue
            print(alpha_name, sharpe_ratio(strategy_returns), SIGNAL_RESULT[alpha_name]['sharpe_tc'])
        for x in for_deletion:
            del SIGNAL_FUTURES[x]
        if len(SIGNAL_FUTURES) == 0:
            break
        time.sleep(60)

    weights = model['weights']
    for alpha in SIGNAL_RESULT:
        if SIGNAL_RESULT[alpha]['sharpe_tc'] > 0.5:
            weights[alpha] = SIGNAL_RESULT[alpha]['signal'] * SIGNAL_RESULT[alpha]['sign']

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    model['weights'] = weights
    model['signals'] = SIGNAL_RESULT
    return model



def get_model(alpha_slowdown=1):
    model = instantiate_model('apex__amna_salient_v1.0a', universe)
    model = load_market_data(model)
    model = compute_availability(model)
    model = create_settings(model)

    market_data = model['market_data']['adjusted']

    dask = ApexDaskClient()
    market_data_sc = dask.scatter(market_data)
    returns = market_data['returns']

    # Send the signal computing to cluster
    sharpe_ratio = lambda x: x.mean()/x.std()*np.sqrt(252)

    SIGNAL_RESULT = {}
    SIGNAL_FUTURES = {}
    market_alphas = []

    for market_alpha in MARKET_ALPHAS:
        market_alpha_name = market_alpha.__name__
        market_alpha = alpha_wrapper(market_alpha)
        market_alphas.append(market_alpha)
        alpha_signal = dask.submit(market_alpha, market_data_sc)
        SIGNAL_FUTURES[market_alpha_name] = alpha_signal


    def get_short_term_reversal(market_data):
        returns = market_data['returns']
        monthly_returns = returns.ewm(span=20).mean()
        signal = monthly_returns.rank(axis=1)
        signal = rank_signal_from_bbg(-signal, cutoff=0)
        return signal

    weights = model['weights']
    weights['sr_reversal'] = get_short_term_reversal(market_data)
    weights['alpha_momentum'] = get_alpha_momentum_signal_daily('AMNA')


    std = market_data['returns'].std()
    normalizing_factor = std.sum()/std
    normalizing_factor = normalizing_factor / normalizing_factor.sum()

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    # Aggregate results
    for i in range(20):
        for_deletion = set()
        for alpha_name, alpha_fut in SIGNAL_FUTURES.items():
            if not alpha_fut.done():
                continue
            try:
                alpha_signal = alpha_fut.result()
            except:
                for_deletion.add(alpha_name)
                continue

            alpha_signal = alpha_signal.fillna(0)[model['availability']]
            alpha_signal = alpha_signal.rank(axis=1, pct=True) * 2 - 1.0
            alpha_signal = alpha_signal * normalizing_factor
            alpha_signal = alpha_signal.ewm(span=alpha_slowdown).mean()
            alpha_signal = alpha_signal[model['availability']]
            alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
            strategy_returns = apex__compute_strategy_returns(market_data, alpha_signal, 0)

            sign = 1
            if strategy_returns.mean() < 0:
                strategy_returns = -strategy_returns
                sign = -1
            strategy_returns_tc = apex__compute_strategy_returns(market_data, alpha_signal, 5)
            strategy_returns = strategy_returns.loc['2000' :]
            try:
                strategy_returns_tc = strategy_returns_tc.loc['2000':]
                SIGNAL_RESULT[alpha_name] = {
                    'returns': strategy_returns,
                    'returns_tc': strategy_returns_tc,
                    'sharpe_tc': sharpe_ratio(strategy_returns_tc.loc['2017':]),
                    'sharpe_no_tc': sharpe_ratio(strategy_returns),
                    'signal': alpha_signal,
                    'sign': sign
                }
                for_deletion.add(alpha_name)
            except:
                for_deletion.add(alpha_name)
                continue
            print(alpha_name, sharpe_ratio(strategy_returns), SIGNAL_RESULT[alpha_name]['sharpe_tc'])
        for x in for_deletion:
            del SIGNAL_FUTURES[x]
        if len(SIGNAL_FUTURES) == 0:
            break
        time.sleep(60)

    weights = model['weights']
    for alpha in SIGNAL_RESULT:
        if SIGNAL_RESULT[alpha]['sharpe_tc'] > 0.5:
            weights[alpha] = SIGNAL_RESULT[alpha]['signal'] * SIGNAL_RESULT[alpha]['sign']

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    model['weights'] = weights
    model['signals'] = SIGNAL_RESULT
    return model


def get_energy_infrastructure_alpha_model(alpha_slowdown=1):
    universe = lambda: APEX_UNIVERSES['Energy Infrastructure']
    model = instantiate_model('apex__amna_salient_v1.0a', universe)
    model = load_market_data(model)
    model = compute_availability(model)
    model = create_settings(model)

    market_data = model['market_data']['adjusted']

    dask = ApexDaskClient()
    market_data_sc = dask.scatter(market_data)
    returns = market_data['returns']

    # Send the signal computing to cluster
    sharpe_ratio = lambda x: x.mean()/x.std()*np.sqrt(252)

    SIGNAL_RESULT = {}
    SIGNAL_FUTURES = {}
    market_alphas = []

    for market_alpha in MARKET_ALPHAS:
        market_alpha_name = market_alpha.__name__
        market_alpha = alpha_wrapper(market_alpha)
        market_alphas.append(market_alpha)
        alpha_signal = dask.submit(market_alpha, market_data_sc)
        SIGNAL_FUTURES[market_alpha_name] = alpha_signal


    def get_short_term_reversal(market_data):
        returns = market_data['returns']
        monthly_returns = returns.ewm(span=20).mean()
        signal = monthly_returns.rank(axis=1)
        signal = rank_signal_from_bbg(-signal, cutoff=0)
        return signal

    weights = model['weights']
    weights['sr_reversal'] = get_short_term_reversal(market_data)
    weights['alpha_momentum'] = get_alpha_momentum_signal_daily('Energy Infrastructure')


    std = market_data['returns'].std()
    normalizing_factor = std.sum()/std
    normalizing_factor = normalizing_factor / normalizing_factor.sum()

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    # Aggregate results
    for i in range(20):
        for_deletion = set()
        for alpha_name, alpha_fut in SIGNAL_FUTURES.items():
            if not alpha_fut.done():
                continue
            try:
                alpha_signal = alpha_fut.result()
            except:
                for_deletion.add(alpha_name)
                continue

            alpha_signal = alpha_signal.fillna(0)[model['availability']]
            alpha_signal = alpha_signal.rank(axis=1, pct=True) * 2 - 1.0
            alpha_signal = alpha_signal * normalizing_factor
            alpha_signal = alpha_signal.ewm(span=alpha_slowdown).mean()
            alpha_signal = alpha_signal[model['availability']]
            alpha_signal = alpha_signal.divide(alpha_signal.abs().sum(axis=1), axis=0)
            strategy_returns = apex__compute_strategy_returns(market_data, alpha_signal, 5)

            sign = 1
            if strategy_returns.mean() < 0:
                strategy_returns = -strategy_returns
                sign = -1
            strategy_returns_tc = apex__compute_strategy_returns(market_data, alpha_signal, 5)
            strategy_returns = strategy_returns.loc['2000' :]
            try:
                strategy_returns_tc = strategy_returns_tc.loc['2000':]
                SIGNAL_RESULT[alpha_name] = {
                    'returns': strategy_returns,
                    'returns_tc': strategy_returns_tc,
                    'sharpe_tc': sharpe_ratio(strategy_returns_tc.loc['2017':]),
                    'sharpe_no_tc': sharpe_ratio(strategy_returns),
                    'signal': alpha_signal,
                    'sign': sign
                }
                for_deletion.add(alpha_name)
            except:
                for_deletion.add(alpha_name)
                continue
            print(alpha_name, sharpe_ratio(strategy_returns), SIGNAL_RESULT[alpha_name]['sharpe_tc'])
        for x in for_deletion:
            del SIGNAL_FUTURES[x]
        if len(SIGNAL_FUTURES) == 0:
            break
        time.sleep(60)

    weights = model['weights']
    for alpha in SIGNAL_RESULT:
        if SIGNAL_RESULT[alpha]['sharpe_tc'] > 0.5:
            weights[alpha] = SIGNAL_RESULT[alpha]['signal'] * SIGNAL_RESULT[alpha]['sign']

    for x in weights:
        wts = weights[x].rank(axis=1, pct=True) * 2 - 1.0
        wts = wts * normalizing_factor
        weights[x] = wts.divide(wts.abs().sum(axis=1), axis=0)

    model['weights'] = weights
    model['signals'] = SIGNAL_RESULT
    return model
