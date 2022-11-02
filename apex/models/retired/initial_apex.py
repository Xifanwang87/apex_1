#!/usr/bin/env python
# coding: utf-8

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
from joblib import Parallel, delayed, parallel_backend

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


from apex.factors.by_universe import UNIVERSE_NEUTRAL_FACTORS
from joblib import parallel_backend, Parallel, delayed, parallel
from funcy import partial


@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='universe_factor_scores')
def universe_factor_scores(universe):
    def compute_universe_factor_scores(universe, factor_dictionary=UNIVERSE_NEUTRAL_FACTORS.copy()):
        factors = list(factor_dictionary.keys())
        def factor_wrapper(fn):
            def wrapped(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except:
                    return None
            return wrapped

        with parallel_backend(backend='loky', n_jobs=20):
            factor_result = Parallel()(delayed(factor_wrapper(factor_dictionary[x]))(universe) for x in factors)
        return dict(zip(factors, factor_result))
    factor_scores = compute_universe_factor_scores(universe)
    return pd.concat(factor_scores, axis=1)


@dataclass
class ApexModelPortfolio:
    blackboard: dict = field(default_factory=dict)
    benchmark: str = field(default='AMNA Index')
    universe_name: str = field(default='AMNA')
    universe: typing.Any = field(default=None)
    universe_metadata: typing.Any = field(default=None)
    universe_factors: typing.Any = field(default=None)
    utilities: typing.Any = field(default=None)
    canadians: typing.Any = field(default=None)
    market_data: typing.Any = field(default=None)
    availability: typing.Any = field(default=None)
    core_portfolio: typing.Any = field(default=None)
    canadian_portfolio: typing.Any = field(default=None)
    utility_portfolio: typing.Any = field(default=None)
    tail_portfolio: typing.Any = field(default=None)
    sub_portfolio_weights: typing.Any = field(default=None)
    portfolio: typing.Any = field(default=None)

    def __post_init__(self):
        utility_ids = [ApexSecurity.from_id(x).id for x in ApexCustomRoweIndicesUniverse().custom_indices.utilities]
        energy_infrastructure_ids = APEX_UNIVERSES['AMNA']
        canadian_ids = [ApexSecurity.from_id(x).id for x in ApexCustomRoweIndicesUniverse().custom_indices.canada]
        self.universe = sorted(set(energy_infrastructure_ids))
        self.universe_metadata = get_security_metadata(*self.universe)
        self.utilities = utility_ids
        self.canadians = canadian_ids
        self.market_data = apex__adjusted_market_data(*self.universe, parse=True)
        self.availability = ~(self.market_data['px_last'].fillna(method='ffill', limit=25).isnull())
        self.universe_factors = universe_factor_scores(self.universe_name)

    def historical_data(self, field=None):
        assert field is not None
        identifiers = self.universe

        @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace='apex_model_portfolio:' + field, asdict=True)
        def get_security_field_data(*identifiers):
            bbg = ApexBloomberg()
            split = partition_all(25, identifiers)
            result = []
            pool = ThreadPoolExecutor()
            for group in split:
                result.append(bbg.history(group, field))
            result = pd.concat(result, axis=1)
            result.columns = result.columns.droplevel(1)
            result = {x: result[x] for x in identifiers}
            return result
        result = get_security_field_data(*identifiers)
        return pd.concat(result, axis=1)


@nb.jit()
def r2_score(series):
    return np.corrcoef(series, np.arange(len(series)))[0, 1]**2

@nb.jit()
def get_rsi(s, window=10):
    rets = s.pct_change().dropna()
    pos_rets = rets.copy()
    pos_rets[pos_rets < 0] = np.nan
    neg_rets = rets.copy()
    neg_rets[neg_rets >= 0] = np.nan
    neg_rets = neg_rets.abs()

    up_series = pos_rets.ewm(span=window).mean()
    down_series = neg_rets.ewm(span=window).mean()

    rs = (up_series/down_series)
    return 100-100/(1+rs)

def compute_core_portfolio(model_portfolio):
    mkt_caps = model_portfolio.historical_data(field='CUR_MKT_CAP')
    holdings = mkt_caps.rank(axis=1, ascending=False)
    holdings = holdings[holdings < 12].dropna(how='all', axis=1)
    model_portfolio.blackboard['core_holdings'] = holdings

    def compute_signal():
        data = model_portfolio.market_data['returns'].fillna(0)[holdings.columns]
        data = data[model_portfolio.availability[holdings.columns]]
        data = data.fillna(0)
        volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
        signal = data.cumsum().rolling(36*20).apply(r2_score) * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        return signal

    signal = compute_signal().dropna(how='all')
    signal = signal.rank(axis=1, ascending=False, pct=True)
    universe_factors = model_portfolio.universe_factors
    factor_scores = -universe_factors['Leverage'] - universe_factors['Value'] - universe_factors['Earnings Yield'] + universe_factors['Profitability'] + universe_factors['Momentum']
    factor_scores = factor_scores[signal.columns].dropna(how='all', axis=1).fillna(method='ffill', limit=10).rolling(20).mean()
    factor_scores = factor_scores.rank(axis=1, ascending=False, pct=True)
    signal = (signal.fillna(1) + factor_scores.fillna(1)).ewm(span=40).mean()
    signal = signal[model_portfolio.availability].rank(axis=1, ascending=True)
    signal = signal < 10
    signal = signal.astype(int)
    core = signal.divide(signal.sum(axis=1), axis=0)
    return core

def compute_tail_portfolio(model_portfolio, num_holdings=25):
    """
    'Growth': universe_growth_factor,
    'Value': universe_value_factor,
    'Leverage': universe_leverage_factor,
    'CFO Yield': universe_yield_factor,
    'Size': universe_size_factor,
    'Credit': universe_credit_factor,
    'Volatility': universe_vol_factor,
    'Momentum': universe_momentum_factor,
    'Earnings Quality': universe_earnings_quality_factor,
    'Earnings Yield': universe_earnings_yield_factor,
    'Dividend Yield': universe_dividend_yield_factor,
    'Liquidity': universe_liquidity_factor,
    'Operating Efficiency': universe_operating_efficiency_factor,
    'Profitability': universe_profitability_factor,
    """
    mkt_caps = model_portfolio.historical_data(field='CUR_MKT_CAP')
    holdings = mkt_caps.rank(axis=1, ascending=False)
    holdings = holdings[holdings > 12].dropna(how='all', axis=1)
    model_portfolio.blackboard['tail_holdings'] = holdings

    def compute_signal():
        data = model_portfolio.market_data['returns'].fillna(0)[holdings.columns]
        data = data[model_portfolio.availability[holdings.columns]]
        data = data.fillna(0)
        volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
        signal = data.cumsum().rolling(36*20).apply(r2_score) * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        return signal

    signal = compute_signal().dropna(how='all')
    signal = signal.ewm(span=40).mean()[model_portfolio.availability].rank(axis=1, ascending=True)
    # universe_factors = model_portfolio.universe_factors
    # universe_factors = universe_factors.loc[:, ~universe_factors.columns.duplicated()]
    # factor_scores = universe_factors['Credit'] - universe_factors['Leverage'] - universe_factors['Size'] + universe_factors['Profitability'] - universe_factors['Value'] - universe_factors['Earnings Yield']
    # factor_scores = factor_scores[signal.columns].dropna(how='all', axis=1).fillna(method='ffill', limit=10).rolling(60).mean()
    # factor_scores = factor_scores.rank(axis=1, ascending=False)
    signal = signal.loc[:, ~signal.columns.duplicated()]
    signal = signal <= num_holdings
    core = signal.divide(signal.sum(axis=1), axis=0)
    return core



def combine_portfolios(model, core_portfolio, tail_portfolio, max_tail=0.5):
    cols = core_portfolio.columns.tolist() + tail_portfolio.columns.tolist()
    cols = sorted(set(cols))
    portfolio = pd.DataFrame(0, index=core_portfolio.index, columns=cols)
    core_portfolio = core_portfolio.divide(core_portfolio.sum(axis=1), axis=0)
    tail_portfolio = tail_portfolio.divide(tail_portfolio.sum(axis=1), axis=0)
    w_core = core_portfolio.multiply(1 - max_tail, axis=0).fillna(0)
    w_tail = tail_portfolio.multiply(max_tail, axis=0).reindex(w_core.index).fillna(0)
    portfolio.loc[w_core.index, w_core.columns] += w_core
    portfolio.loc[w_tail.index, w_tail.columns] += w_tail
    portfolio = portfolio[model.availability].fillna(limit=5, method='ffill')
    portfolio = portfolio.divide(portfolio.sum(axis=1), axis=0)
    return portfolio

def defensive_overlay_returns(model_portfolio, portfolio, max_hedge=0.5):
    portfolio_returns = (portfolio.shift(1) * model_portfolio.market_data['returns']).sum(axis=1)
    portfolio_cumulative_returns = (portfolio_returns + 1).cumprod()
    slope_200d_mean = portfolio_cumulative_returns.loc['1996':].ewm(span=200).mean()
    zscore = (portfolio_cumulative_returns - slope_200d_mean).ewm(span=20).mean()
    zscore = zscore/zscore.expanding().std()
    zscore[zscore > 0] = 0
    zscore = -zscore.ewm(span=60).mean()
    overlay = (zscore.expanding().max() - zscore)/zscore.expanding().max()
    overlay = np.maximum(overlay, max_hedge).shift(1)
    canadians = ApexCustomRoweIndicesUniverse().custom_indices.utilities
    canadian_returns = apex__adjusted_market_data(*canadians, parse=True)['returns'].fillna(0)
    canadian_returns = canadian_returns.mean(axis=1)
    return portfolio_returns * overlay + canadian_returns * (1-overlay)


def compute_portfolios():
    model_portfolio = ApexModelPortfolio()

    core_portfolio = compute_core_portfolio(model_portfolio)
    tail_portfolio = compute_tail_portfolio(model_portfolio, num_holdings=30)
    tail_portfolio = tail_portfolio.divide(tail_portfolio.sum(axis=1), axis=0)
    portfolio = combine_portfolios(model_portfolio, core_portfolio.fillna(0).ewm(span=60).mean(), tail_portfolio.ewm(span=60).mean(), max_tail=0.2)

    market_data = model_portfolio.market_data

    PORTFOLIOS = pd.concat({
        'Core': core_portfolio,
        'Tail': tail_portfolio,
        'Portfolio': portfolio,
    }, axis=1).fillna(method='ffill', limit=2).fillna(0).ewm(span=3).mean()
    PORTFOLIOS = PORTFOLIOS.divide(PORTFOLIOS.sum(axis=1, level=0), level=0, axis=0)
    PORTFOLIOS_STACKED = PORTFOLIOS.stack().fillna(0).stack().reset_index(drop=False).rename(columns={
        'level_1': 'bloomberg_id',
        'level_2': 'sleeve',
        0: 'weight'
    })
    PORTFOLIOS_STACKED['model_name'] = 'Salient vs AMNA'
    PORTFOLIOS_STACKED['model_version'] = 'v1'
    PORTFOLIOS_STACKED['compute_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    return {
        'portfolios': PORTFOLIOS,
        'current_portfolio': PORTFOLIOS['Portfolio'].iloc[-1],
        'stacked_portfolios': PORTFOLIOS_STACKED,
        'market_data': model_portfolio.market_data,
        'factor_data': model_portfolio.universe_factors
    }

def portfolio_creation_task(ds=None, **ctx):
    result = compute_portfolios()

    # Now submit to mlflow
    portfolios = result['portfolios']
    current_portfolio =  result['current_portfolio']
    market_data =  result['market_data']
    factor_data =  result['factor_data']

