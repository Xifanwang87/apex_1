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
from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg,
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

def rank_signal(result):
    result = result.copy()
    result = result.rank(axis=1, pct=True)
    result = result - 0.5
    return result

def alpha_wrapper(fn):
    def wrapped(market_data):
        return fn(opens=market_data['px_open'].fillna(method='ffill', limit=2),
                  highs=market_data['px_high'].fillna(method='ffill', limit=2),
                  lows=market_data['px_low'].fillna(method='ffill', limit=2),
                  closes=market_data['px_last'].fillna(method='ffill', limit=2),
                  returns=market_data['returns'].fillna(0),
                  volumes=market_data['px_volume'].fillna(method='ffill', limit=2).fillna(0))
    return wrapped

@alpha_wrapper
def alpha_41(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (((high * low)^0.5) - vwap)
    """
    adv = volumes.rolling(20).mean()
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows

    return rank_signal((highs*lows).pow(0.5) - vwaps)

@alpha_wrapper
def alpha_19(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    """
    first_term = -np.sign(closes - closes.shift(7) + closes.diff(7))
    second_term = 1 + rank_signal(1+returns.rolling(250).sum())
    return first_term * second_term

@alpha_wrapper
def alpha_33(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank((-1 * ((1 - (open / close))^1)))
    Why power of 1?
    """
    return rank_signal((-(1-opens/closes)))


@alpha_wrapper
def alpha_21(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     IF TRUE: (sum(close, 8) / 8) + stddev(close, 8) < (sum(close, 2) / 2)
     THEN: -1
     ELSE:
         IF TRUE: (sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))
         THEN: 1
         ELSE:
         IF (1 < (volume / adv20)) OR (volume / adv20) == 1)
         THEN 1
         ELSE -1
    """
    adv = volumes.rolling(20).mean()

    first_if_clause = closes.rolling(8).mean() + closes.rolling(8).std() < closes.rolling(2).mean()
    second_if_clause = closes.rolling(8).mean() - closes.rolling(8).std() > closes.rolling(2).mean()

    result = pd.DataFrame(1, index=closes.index, columns=closes.columns)
    result[second_if_clause] = -1
    result[first_if_clause] = 1

    return result

@lru_cache(maxsize=None)
def compute_alpha_41_signal(universe_name):
    market_data = get_market_data(universe_name)
    a41 = alpha_41(market_data)
    return a41

@lru_cache(maxsize=None)
def compute_alpha_33_signal(universe_name):
    market_data = get_market_data(universe_name)
    a19 = alpha_33(market_data).ewm(span=22).mean()
    return a19

def get_short_term_reversal(market_data):
    returns = market_data['returns']
    monthly_returns = returns.rolling(20).sum()
    signal = monthly_returns.rank(axis=1)
    signal = rank_signal_from_bbg(-signal, cutoff=0)
    return signal


@dataclass
class ApexModel:
    VERSION = "0.1a"
    BENCHMARK: str = field(default='AMNA Index')
    BASE_UNIVERSES: str = field(default_factory=lambda: ['AMNA'])

    @staticmethod
    def fill_nas_series(x):
        last = x.last_valid_index()
        x.loc[:last] = x.loc[:last].ffill()
        return x

    @staticmethod
    def universe_factor_scores(universe, factor, cutoff=0):
        assert factor in UNIVERSE_NEUTRAL_FACTORS
        result = UNIVERSE_NEUTRAL_FACTORS[factor](universe, cutoff=cutoff)
        return result.apply(ApexModel.fill_nas_series)


    def _compute_core_signal(self, window=36*22, with_volatility=True, num_stocks_universe=15, min_days_universe=252):
        core_universe = self.core_universe(num_stocks=num_stocks_universe, min_days=min_days_universe)
        data = self.returns[core_universe.columns]

        base_signal = column_rolling_apply_on_cluster(data.cumsum(), r2_score, window)
        if with_volatility:
            volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        else:
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum())

        signal = signal[core_universe]
        return signal

    def _compute_tail_signal(self, window=36*22, with_volatility=True):
        data = self.returns

        base_signal = column_rolling_apply_on_cluster(data.cumsum(), r2_score, window)
        if with_volatility:
            volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        else:
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum())
        return signal

    def __post_init__(self):
        tickers = set()
        for universe in self.BASE_UNIVERSES:
            tickers.update(APEX_UNIVERSES[universe])
        current_benchmark = get_index_member_weights_on_day(self.BENCHMARK, pd.Timestamp.now() - pd.DateOffset(days=1)).index.tolist()
        current_benchmark = [ApexSecurity.from_id(x).id for x in current_benchmark]
        tickers.update(current_benchmark)
        self.security_ids = sorted(tickers)
        self.securities = [ApexSecurity.from_id(x) for x in self.security_ids]
        self.security_by_ticker = {x.parsekyable_des: x for x in self.securities}
        self.tickers = [x.parsekyable_des for x in self.securities]
        self.alpha_momentum_signal = get_alpha_momentum_signal_daily('AMNA')
        self.invalidate_caches()

    def invalidate_caches(self, core_portfolio_universe=False, core_portfolio_signal=True, **kwargs):
        """
        Used to invalidate caches.
        """
        if core_portfolio_universe:
            STRATEGY_CACHE.clean(['core_portfolio_universe'])

        if core_portfolio_signal:
            STRATEGY_CACHE.clean(['core_portfolio_signal'])

    def historical_data(self, field=None, fillna=True):
        assert field is not None
        identifiers = self.security_ids

        @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace=f'apex_model_portfolio:{field}:{self.VERSION}', asdict=True)
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
        result = pd.concat(result, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        if fillna:
            result = result.apply(self.fill_nas_series)
        return result[self.security_ids]

    @lazyproperty
    def availability(self):
        closes = self.closes.resample('D').pad().apply(ApexModel.fill_nas_series)
        price_filter = closes > 10
        market_cap_filter = self.market_caps.resample('D').pad().apply(ApexModel.fill_nas_series) > 250
        availability = (price_filter & market_cap_filter).fillna(False)
        for bbid, date in AMNA_AVAILABILITY_REMOVALS.items():
            if bbid in availability.columns:
                availability.loc[date:, bbid] = False
        return availability


    @lazyproperty
    def data(self):
        data = apex__adjusted_market_data(*self.security_ids, parse=True)
        data['returns'] = data['returns'].fillna(0)
        data = data.loc[:, ~data.columns.duplicated()]
        return data.apply(self.fill_nas_series).sort_index(axis=1)

    @lazyproperty
    def factor_scores(self):
        universe = 'AMNA'
        RESULT = defaultdict(list)
        for factor_name in UNIVERSE_NEUTRAL_FACTORS:
            try:
                factor = self.universe_factor_scores(universe, factor_name)
                factor = factor.loc[:, ~factor.columns.duplicated()]
                RESULT[factor_name] = factor.apply(ApexModel.fill_nas_series)
            except ValueError:
                print('ValueError in', factor_name)
                continue
        return RESULT

    @lazyproperty
    def closes(self):
        return self.data['px_last'].sort_index(axis=1)

    @lazyproperty
    def returns(self):
        return self.data['returns'].sort_index(axis=1)

    @lazyproperty
    def market_caps(self):
        return self.historical_data(field='CUR_MKT_CAP').apply(self.fill_nas_series).sort_index(axis=1)

    @lazyproperty
    def alpha_signals(self):
        RESULT = defaultdict(list)
        for universe in self.BASE_UNIVERSES:
            alpha_signal = (compute_alpha_41_signal(universe) + compute_alpha_33_signal(universe)) * 0.5
            RESULT[universe] = alpha_signal
            RESULT[universe] = RESULT[universe].apply(ApexModel.fill_nas_series)
        return pd.concat(RESULT, axis=1)


    def core_universe(self, num_stocks=15, min_days=252):
        """
        ORIGINAL CODE:
            alpha_signal = (compute_alpha_41_signal(model_portfolio.universe_name) + compute_alpha_33_signal(model_portfolio.universe_name)) * 0.5
            signal = (signal.fillna(1) - alpha_signal.fillna(1)[holdings.columns][availability])

            signal = signal[model_portfolio.availability].rank(axis=1, ascending=True)
            signal = signal <= 8
            signal = signal.astype(int)
            core = signal.divide(signal.sum(axis=1), axis=0)
            core = core.loc[:, ~core.columns.duplicated()]
            return core
        """
        availability = self.availability
        selection = self.market_caps[self.availability].rank(axis=1, ascending=False)
        selection = selection <= num_stocks
        # If it has been in the group it remains in it
        selection[~selection] = np.nan
        selection = selection.fillna(method='ffill', limit=min_days).apply(self.fill_nas_series)
        selection = selection[self.availability].fillna(False)

        selection_cols = selection.sum() > min_days
        selection_cols = sorted(selection_cols[selection_cols].index.tolist())
        selection = selection[selection_cols]
        return selection.astype(bool)

    def _compute_tail_signal(self, window=36*22, with_volatility=True):
        data = self.returns

        base_signal = column_rolling_apply_on_cluster(data.cumsum(), r2_score, window)
        if with_volatility:
            volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        else:
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum())
        return signal


    def _compute_core_signal(self, window=36*22, with_volatility=True, num_stocks_universe=15, min_days_universe=252):
        core_universe = self.core_universe(num_stocks=num_stocks_universe, min_days=min_days_universe)
        data = self.returns[core_universe.columns]

        base_signal = column_rolling_apply_on_cluster(data.cumsum(), r2_score, window)
        if with_volatility:
            volatility = data.rolling(252).std().rank(axis=1, pct=True, ascending=True)
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum()) * volatility
        else:
            signal = base_signal * (data.rolling(252).sum() - data.rolling(22).sum())

        signal = signal[core_universe]
        return signal

    def core_signal(self, num_stocks=18, cutoff=0.2):
        """
        Core has no alpha signals.

        We'll be using alpha signals on the rest of the portfolio
        """
        core_universe = self.core_universe(num_stocks=num_stocks)

        # The r2 signal
        base_signal = self._compute_core_signal(with_volatility=False, num_stocks_universe=num_stocks).rank(axis=1, ascending=True, pct=True)

        # The Factor Signals
        factors = self.factor_scores
        factor_result = {}
        for factor in ['Growth', 'CFO Yield', 'Liquidity', 'Credit', 'Momentum']:
            factor_result[factor] = factors[factor]
        factor_result['Size'] = -factors['Size']
        factor_result = pd.concat(factor_result, axis=1).mean(axis=1, level=1)
        factor_result[factor_result.abs() < 1e-7] = np.nan # Anything too small
        factor_result = factor_result.rank(axis=1, pct=True, ascending=True)[[x for x in base_signal.columns if x in factor_result.columns]]
        factor_result = factor_result[core_universe]

        # Alpha Momentum
        alpha_momentum_signal = self.alpha_momentum_signal.copy()
        alpha_momentum_signal = alpha_momentum_signal[[x for x in alpha_momentum_signal.columns if x in factor_result.columns]]
        alpha_momentum_signal = alpha_momentum_signal[core_universe].rank(axis=1, pct=True, ascending=True)

        # Short Term Reversal
        short_term_reversal = get_short_term_reversal(self.data)
        short_term_reversal[short_term_reversal < 0] = 0
        short_term_reversal = short_term_reversal[[x for x in short_term_reversal.columns if x in factor_result.columns]][core_universe].fillna(0)

        # Cleanup
        signal = (base_signal[core_universe] + factor_result[core_universe]) + alpha_momentum_signal + short_term_reversal
        signal = signal.ewm(span=7).mean()
        signal = signal.rank(axis=1, pct=True, ascending=True)
        signal = signal[signal > cutoff].fillna(method='ffill', limit=2)
        signal[signal < 1e-5] = 0
        signal = signal.fillna(method='ffill', limit=2)[core_universe]
        return signal.divide(signal.sum(axis=1), axis=0)

    def tail_universe(self, num_stocks_leave_out=15, min_days=252):
        """
        ORIGINAL CODE:
            alpha_signal = (compute_alpha_41_signal(model_portfolio.universe_name) + compute_alpha_33_signal(model_portfolio.universe_name)) * 0.5
            signal = (signal.fillna(1) - alpha_signal.fillna(1)[holdings.columns][availability])

            signal = signal[model_portfolio.availability].rank(axis=1, ascending=True)
            signal = signal <= 8
            signal = signal.astype(int)
            core = signal.divide(signal.sum(axis=1), axis=0)
            core = core.loc[:, ~core.columns.duplicated()]
            return core
        """
        availability = self.availability
        selection = self.market_caps[self.availability].rank(axis=1, ascending=False)
        selection = selection >= num_stocks_leave_out
        selection[~selection] = np.nan
        selection = selection.fillna(method='ffill', limit=min_days).apply(self.fill_nas_series)
        selection = selection[self.availability].fillna(False)
        selection_cols = selection.sum() > min_days
        selection_cols = sorted(selection_cols[selection_cols].index.tolist())
        selection = selection[selection_cols]
        return selection.astype(bool)

    def tail_signal(self, num_stocks_leave_out=18, cutoff=0.5):
        """
        Core has no alpha signals.

        We'll be using alpha signals on the rest of the portfolio
        """
        core_universe = self.tail_universe(num_stocks_leave_out=num_stocks_leave_out)

        # The r2 signal
        base_signal = self._compute_tail_signal(with_volatility=False).rank(axis=1, ascending=True, pct=True)

        # The Factor Signals
        factors = self.factor_scores
        factor_result = {}
        for factor in ['Growth', 'CFO Yield', 'Liquidity', 'Credit', 'Momentum']:
            factor_result[factor] = factors[factor]
        factor_result['Size'] = -factors['Size']
        # factor_result['Leverage'] = -factors['Leverage']
        factor_result = pd.concat(factor_result, axis=1).mean(axis=1, level=1)
        factor_result[factor_result.abs() < 1e-7] = np.nan # Remove anything too small
        factor_result = factor_result.rank(axis=1, pct=True, ascending=True)[[x for x in base_signal.columns if x in factor_result.columns]]
        factor_result = factor_result[core_universe]

        alpha_momentum_signal = self.alpha_momentum_signal.copy()
        alpha_momentum_signal = alpha_momentum_signal[[x for x in alpha_momentum_signal.columns if x in factor_result.columns]].rank(axis=1, pct=True, ascending=True)

        short_term_reversal = get_short_term_reversal(self.data)
        short_term_reversal[short_term_reversal < 0] = 0
        short_term_reversal = short_term_reversal[[x for x in short_term_reversal.columns if x in factor_result.columns]]
        short_term_reversal = short_term_reversal[core_universe].fillna(0)
        # Cleanup
        signal = (base_signal[core_universe] + factor_result[core_universe]+ self.alpha_momentum_signal[core_universe] + short_term_reversal)
        signal = signal.ewm(span=7).mean()
        signal = signal.rank(axis=1, pct=True, ascending=True)
        signal = signal[signal > cutoff].fillna(0)
        signal = signal[core_universe]
        signal[signal < 1e-5] = 0
        signal = signal.fillna(method='ffill', limit=2)
        return signal.divide(signal.sum(axis=1), axis=0)

    def january_effect_signal(self):
        data = self.data
        returns = data.returns

        jan_effect_portfolio_by_year = []
        for year in sorted(set(returns.index.year)):
            signal_ix = pd.date_range(pd.Timestamp(year=year, day=1, month=1), pd.Timestamp(year=year, day=15, month=12))
            holding_period = pd.date_range(pd.Timestamp(year=year, day=16, month=12), pd.Timestamp(year=year+1, day=20, month=2))
            yearly_rets = returns.loc[signal_ix].dropna(how='all', axis=1).dropna(how='all', axis=0).sum().sort_values(ascending=False)
            if len(yearly_rets.index) < 20:
                continue
            yearly_portfolio = yearly_rets.rank(ascending=False)
            yearly_portfolio = yearly_portfolio[yearly_portfolio <= 10]
            if len(yearly_portfolio) == 0:
                continue
            jan_effect_portfolio_by_year.append(pd.DataFrame(1/len(yearly_portfolio), columns=yearly_portfolio.index, index=holding_period))

        jan_effect_portfolio = pd.concat(jan_effect_portfolio_by_year).fillna(0)
        jan_effect_portfolio = jan_effect_portfolio.divide(jan_effect_portfolio.sum(axis=1), axis=0)
        return jan_effect_portfolio[self.availability]


def generate_bloomberg_weight_sheet(weights):
    weights = weights.copy()
    weights = weights[weights.abs() > 1e-5]
    df = pd.DataFrame({'FIXED WEIGHT': weights})
    df['PORTFOLIO NAME'] = 'APEX'
    df['SECURITY_ID'] = [' '.join(x.split(' ')[:2]) for x in fix_security_name_index(df).index.tolist()]
    return df.sort_values(by='FIXED WEIGHT', ascending=False)

def generate_eze_weight_sheet(weights):
    pass

def compute_cash_exposure(max_exposure=0.1):
    data = apex__adjusted_market_data('AMZ Index', 'CL1 Comdty', parse=True)['px_last'].pct_change()
    data = data.rolling(252).std()
    return data

def get_model():
    model = ApexModel()
    tail_factor_signal = model.tail_signal(num_stocks_leave_out=0, cutoff=0.25)
    core_factor_signal = model.core_signal(num_stocks=18, cutoff=0.1)

    factor_signal = core_factor_signal.divide(core_factor_signal.abs().sum(axis=1), axis=0)
    factor_signal = factor_signal.divide(factor_signal.abs().sum(axis=1), axis=0)
    factor_signal = factor_signal.divide(factor_signal.max(axis=1), axis=0) * 0.0985

    total_core_weights = factor_signal.sum(axis=1).shift(1) * 0.8
    total_noncore_weights = 1 - total_core_weights

    noncore_portfolio = tail_factor_signal.multiply(total_noncore_weights, axis=0)
    core_portfolio = factor_signal
    all_assets = noncore_portfolio.columns.tolist() + core_portfolio.columns.tolist()
    all_assets = sorted(set(all_assets))
    noncore_portfolio = noncore_portfolio.T.reindex(all_assets).T.fillna(0)
    core_portfolio = core_portfolio.T.reindex(all_assets).T.fillna(0)
    portfolio = noncore_portfolio.fillna(0) + core_portfolio.fillna(0)
    portfolio[portfolio > 0.1] = 0.095
    portfolio = portfolio.ewm(span=2).mean()
    portfolio = portfolio.divide(portfolio.sum(axis=1), axis=0).loc['1996':]
    return portfolio

