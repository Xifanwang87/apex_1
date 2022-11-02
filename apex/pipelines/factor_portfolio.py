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

import boltons as bs
import dogpile.cache as dc
import funcy as fy
import matplotlib.pyplot as plt
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy as sc
import sklearn as sk
import statsmodels.api as sm
import pendulum

# Other
import toml
import toolz as tz
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as to
import torch.tensor as tt
# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from IPython.core.debugger import set_trace as bp
from scipy.optimize import Bounds, minimize
from toolz import partition_all

import dask_ml
import inflect
import joblib
import pygmo as pg
import pyomo as po
from apex.alpha.market_alphas import MARKET_ALPHAS
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
from joblib import Parallel, delayed, parallel_backend
from queue import Queue
from apex.toolz.reporting import SalientReportWorkbook, SalientDataframeExcelReport, SalientExcelLineChartSheet
from apex.toolz.mutual_information import ApexMutualInformationAnalyzer


def call_market_data_alpha_fn(fn, market_data):
    return fn(opens=market_data['px_open'].fillna(method='ffill'),
              highs=market_data['px_high'].fillna(method='ffill'),
              lows=market_data['px_low'].fillna(method='ffill'),
              closes=market_data['px_last'].fillna(method='ffill'),
              returns=market_data['returns'].fillna(0),
              volumes=market_data['px_volume'].fillna(method='ffill'))

def rolling_mean(df, period=10):
    return df.rolling(period).mean()

def ewm(df, period=10):
    return df.ewm(span=period).mean()

def rolling_median(df, period=10):
    return df.rolling(period).median()

def rolling_sum(df, period=10):
    return df.rolling(period).sum()

def rank(df, subtract_mean=True):
    result = df.rank(axis=1, pct=True)
    if subtract_mean:
        result = result.subtract(result.mean(axis=1), axis=0)
    return result

SMOOTHING_METHODS = {
    'ewm': ewm,
    'median': rolling_median,
    'mean': rolling_mean,
    'sum': rolling_sum,
}

@nb.njit
def fitness_fn(covariance, target_risk_contribution, volatility_target, scores, weights):
    ctw = covariance @ weights
    port_var = weights @ ctw

    mcr_abs = np.abs(np.diag(weights) @ ctw)
    mcr_abs = mcr_abs / port_var
    objective_fn = np.sum(np.power(mcr_abs - target_risk_contribution, 2))

    they_are_wts = np.power(np.sum(np.abs(weights)) - 1., 2)
    return objective_fn, they_are_wts

class risk_budget_weights:
    def __init__(self, dim=None, covariance=None, volatility_target=None, scores=None):
        self.dim = dim
        self.covariance = covariance
        self.volatility_target = volatility_target
        self.scores = scores
        self.target_risk_contribution = np.abs(self.scores)/np.sum(np.abs(self.scores))

        upper_bounds, lower_bounds = [], []
        for x in self.scores:
            if np.sign(x) > 0:
                upper_bounds.append(1)
                lower_bounds.append(0.)
            else:
                upper_bounds.append(0.)
                lower_bounds.append(-1)

        self.ub = upper_bounds
        self.lb = lower_bounds

    def get_bounds(self):
        return (self.lb, self.ub)

    def fitness(self, w):
        return fitness_fn(self.covariance, self.target_risk_contribution, self.volatility_target, self.scores, w)

    def get_name(self):
        return "Risk Budgeted Weights"

    def get_nic(self):
        return 1

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


def opt_risk_budget_problem(problem):
    """
    Problem: maximize sum(abs(signal_i)*log(abs(signal_i)))
    s.t. w'Cw < sigma
    """
    ipt = pg.ipopt()
    ipt.set_string_option("hessian_approximation", "limited-memory")
    ipt.set_numeric_option("tol", 5E-4) # Don't care about anything smaller than 5 basis pts
    ipt_algo = pg.algorithm(ipt)
    prb = pg.problem(problem)
    prb.c_tol = [0.15]

    pop = pg.population(prb, size=1)
    pop = ipt_algo.evolve(pop)
    return pop


def construct_risk_budget_weights(scores, covariance_mtx, volatility_target=0.2):
    problem = risk_budget_weights(dim=scores.values.size, covariance=covariance_mtx.values,
                                  volatility_target=volatility_target,
                                  scores=scores.values)
    algo_res = opt_risk_budget_problem(problem).champion_x
    algo_res = pd.Series(algo_res, scores.index)
    return algo_res

def construct_active_risk_budget_weights(scores, benchmark_weights, covariance_mtx,
                                         volatility_target=0.2):
    combo = pd.DataFrame({'scores': scores, 'bwts': benchmark_weights}).fillna(0)
    scores = combo['scores']
    benchmark_weights = combo['bwts']
    problem = risk_budget_weights(
        dim=scores.values.size, covariance=covariance_mtx.values,
        volatility_target=volatility_target,
        scores=scores.values,
        benchmark=benchmark_weights.values
    )
    algo_res = opt_risk_budget_problem(problem).champion_x
    algo_res = pd.Series(algo_res, scores.index)
    return algo_res


def construct_variance_parity_weights(signal, covariance, volatility_target=0.2):
    """
    weights = sigma/sum(sigma) * sign(signal)
    """
    variance = np.diag(covariance)
    weights = np.sign(signal) * variance / np.sum(variance)
    portfolio_variance = weights.T @ covariance @ weights
    portfolio_vol = np.sqrt(252 * portfolio_variance)
    return weights * volatility_target / portfolio_vol

def construct_dollar_parity_weights(signal, covariance, volatility_target=0.2):
    """
    weights = 1/n * sign(signal)
    """
    return np.sign(signal) * np.full(signal.size, 1/signal.size)

def construct_parsimonious_weights(signal, covariance, volatility_target=0.2):
    """
    Averaging out the 3 different weights because of uncertainty over relative weights.
    """
    result = construct_variance_parity_weights(signal, covariance, volatility_target=0.2)
    result += construct_dollar_parity_weights(signal, covariance, volatility_target=0.2)
    result += construct_risk_budget_weights(signal, covariance, volatility_target=0.2)
    result = result / 3
    return result

def get_covariance_matrices(returns):
    result = {
        'lw': lw_cov(returns),
        'oas': oas_cov(returns),
        'min_cov_det': min_cov_det(returns),
    }
    return result

def construct_portfolio(signal, covariance, methodology='risk_budget', minimum_securities=6, volatility_target=0.2):
    signal = signal[signal != 0].dropna()
    covariance = covariance.loc[signal.index.tolist(), signal.index.tolist()]
    if len(signal.index) < minimum_securities:
        return pd.Series(0, signal.index)
    if methodology == 'risk_budget':
        return construct_risk_budget_weights(signal, covariance, volatility_target=volatility_target)
    elif methodology == 'volatility_parity':
        return construct_variance_parity_weights(signal, covariance, volatility_target=volatility_target)
    elif methodology == 'dollar_parity':
        return construct_dollar_parity_weights(signal, covariance, volatility_target=volatility_target)
    elif methodology == 'parsimonious':
        return construct_parsimonious_weights(signal, covariance, volatility_target=volatility_target)


from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
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


def score_portfolio_construction(dates, signal, market_data, methodology='risk_budget', minimum_securities=6, volatility_target=0.2, key=None):
    result = {}
    weights = None
    all_securities = set(market_data.columns.get_level_values(1).tolist())
    securities_signal = set(signal.loc[list(dates)].dropna(how='all', axis=1).columns.tolist())
    securities = sorted(all_securities.intersection(securities_signal))
    market_data = market_data.copy().loc[:dates[-1]]
    market_data.index = [x.strftime("%Y-%m-%d") for x in market_data.index]
    market_data = market_data.swaplevel(axis=1)[securities].swaplevel(axis=1).sort_index(axis=1)
    closes = market_data['px_last']
    returns = market_data['returns']

    for date in dates:
        dt = date.strftime('%Y-%m-%d')
        day_signal = signal.loc[dt].dropna()
        day_signal = day_signal.reindex(securities).fillna(0)
        day_signal[day_signal.abs() < 1e-5] = np.nan
        day_signal = day_signal.dropna()
        day_returns = returns.iloc[-512:-1][day_signal.index.tolist()]
        covariance = default_covariance(day_returns)
        weights = construct_portfolio(day_signal,
                                    covariance,
                                    methodology=methodology,
                                    minimum_securities=minimum_securities,
                                    volatility_target=volatility_target)
        weights[closes.loc[dt].isnull()] = np.nan
        result[dt] = weights.reindex(all_securities).fillna(0)
    result = pd.DataFrame(result).T
    result[result.abs() < 1e-7] = np.nan
    result = result.fillna(0)
    if key is None:
        key = uuid.uuid4().hex
    filename = f'/mnt/data/experiments/alpha/compute/{key}.ending.{dt}.parquet'
    result.to_parquet(filename)
    return filename


def default_split():
    return [('train', 0.85),  ('pre_validation_holdout', 0.9), ('validation', 0.95), ('post_validation_holdout', 1)]

def min_max_scaler(raw_data, axis=0, feature_range=(-3, 3)):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        return min_max_scaler(raw_data.T, axis=0).T

    data_min = data.min(axis=1)
    data_max = data.max(axis=1)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    scale = (feature_range[1] - feature_range[0])/data_range
    min_scale = feature_range[0] - data_min.multiply(scale)
    result = data.multiply(scale, axis=0).add(min_scale, axis=0)
    return result

def max_abs_scaler(raw_data, axis=0):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        return min_max_scaler(raw_data.T, axis=0).T

    maxabs = data.abs().max(axis=1)
    scale = 1/maxabs
    result = data.multiply(scale, axis=0)
    return result

def zscore_and_windsorize_raw_signal(raw_signal):
    """
    Let's windsorize raw signals. That is, let's cut off extremes on a per-ticker basis.
    """
    result = max_abs_scaler(raw_signal, axis=1)
    result[result.abs() < 0.5] = 0
    return result - 0.5



@dataclass
class ApexMarketAlphaRun:
    """
    Note: you don't need to be efficient, you have the cluster. Whatever if it takes a long time to backtest. You can start
    caching shit later on.
    Hyperparameters:
    {
        'signal_smoothing': integer,
        'smoothing_method': in ('ewm', 'mean', 'median')
        'weight_smoothing': integer,
        'should_rank': boolean,
        'long_short_deciles': {'long': {1, 2}, 'short': {10, 9}},
        'portfolio_construction' in ('risk_budget', 'volatility_parity', 'dollar_parity'),
        'should_normalize': boolean
    }
    """
    name: str
    hyperparameters: typing.Any
    alpha: typing.Callable
    universe_name: str
    universe: list # security universe to compute the factor for.
    data_split: list = field(default_factory=default_split)
    volatility_target: float = field(default=0.2)
    minimum_securities: int = field(default=6)
    _market_data: typing.Any = field(init = False)
    run_id: str = field(default_factory = lambda: uuid.uuid4().hex)
    benchmarks: list = field(default_factory=list)
    def __post_init__(self):
        self._market_data = get_security_market_data(self.universe)
        self.scores = self.create_alpha_scores()
        hyperparameters = self.hyperparameters
        if isinstance(self.hyperparameters, dict):
            hyperparameters = [self.hyperparameters]
        self._hyperparameters = hyperparameters



    def compute_parametrized_score(self, raw_scores):
        ranked_scores = self.rank_signal(raw_scores)
        score_portfolio = self.long_short_deciles(ranked_scores)
        score_portfolio = self.smooth_signal(score_portfolio)
        score_portfolio = score_portfolio[self.validity_table].loc['2005':]

        # Now I need to normalize by maxabs
        total_secs = (~score_portfolio[score_portfolio != 0].isnull()).sum(axis=1)
        total_secs[total_secs < self.minimum_securities] = np.nan
        total_secs = total_secs.dropna()
        score_portfolio = score_portfolio.loc[total_secs.index[0]:]
        return score_portfolio

    def run(self):
        scores = self.scores
        run_name = self.name
        model_name = f'apex:factor_models:market_alpha:{self.name}:model'
        clt = ApexDaskClient()
        s_mkt_dt = clt.scatter(self._market_data)
        for count, hp in enumerate(self._hyperparameters):
            self.hyperparameters = hp
            hp_score = self.compute_parametrized_score(scores)
            s_hp_score = clt.scatter(hp_score)
            results = self.backtest_scores(hp_score.index.tolist(), s_hp_score, s_mkt_dt, clt)
            final_results = {
                'name': self.name,
                'backtest_results': results,
                'run_date': pendulum.now().to_iso8601_string(),
                'parameters': {
                    'hyperparameters': self.hyperparameters,
                    'universe_name': self.universe_name,
                    'volatility_target': self.volatility_target,
                }
            }
            cache_key = f'apex:backtests:alpha_model:results:{self.run_id}:{count}'
            arctic = ArcticApex()
            library = arctic.get_library('apex:backtests:alpha')
            library.write(cache_key, final_results, metadata={'alpha': self.name,
                                        'run_id': self.run_id,
                                        'date': str(pd.Timestamp.now().date())})
            # Now on MLFlow
            self.record_run(results, model_name, run_name, cache_key)
        return final_results

    def market_data(self):
        return self._market_data.copy()

    def rank_signal(self, signal):
        if self.hyperparameters.get('should_rank', False):
            signal = rank(signal)
        return signal

    def smooth_signal(self, signal):
        smoothing = self.hyperparameters.get('signal_smoothing', 1)
        smooth_method = SMOOTHING_METHODS[self.hyperparameters.get('smoothing_method', 'mean')]
        return smooth_method(signal, period=smoothing)

    def record_run(self, alpha_results, model_name, run_name, cache_key):
        client = MLFlowTrackingClient(tracking_uri="http://10.15.201.160:18001")
        run = client.create_run(1,
            run_name=model_name,
        )
        run_id = run.info.run_uuid
        parameters = self.hyperparameters.copy()
        parameters.update({
            'universe_name': self.universe_name,
            'volatility_target': self.volatility_target
        })

        metrics_performance = alpha_results['stats']['performance_stats'].to_dict()
        client.log_param(run_id, 'cache_key', cache_key)
        for param_name, param_value in metrics_performance.items():
            client.log_metric(run_id, param_name, param_value)

        for param_name, param_value in parameters.items():
            client.log_param(run_id, param_name, param_value)
        client.log_param(run_id, 'factor_sign', alpha_results['factor_sign'])
        report_loc = self.create_report_workbook(alpha_results)
        client.log_artifact(run_id, report_loc)

    def create_report_workbook(self, results):
        file_loc = f'/mnt/data/experiments/alpha/reports/{self.name}.{uuid.uuid4().hex}.xlsx'
        benchmarks = self.benchmarks
        report_wb = SalientReportWorkbook()

        ### WEIGHTS
        weights_data = results['weights'].copy()
        weights_data[weights_data == 0.0] = np.nan
        weights_data = weights_data.dropna(how='all')
        weights_sheet = SalientDataframeExcelReport(
            weights_data,
            'Weights',
        )

        ### Performance
        performance_data = pd.DataFrame({'Performance Metrics': results['stats']['performance_stats']})
        stats_sheet = SalientDataframeExcelReport(
            performance_data,
            'Statistics',
        )

        ### Drawdown
        dd_data = results['stats']['drawdown_stats']
        dd_sheet = SalientDataframeExcelReport(
            dd_data,
            'Drawdown',
        )
        ### Chart
        cumulative_returns = (results['returns'].fillna(0) + 1).cumprod()
        cum_rets_sheet = SalientDataframeExcelReport(
            pd.DataFrame({'Cumulative Returns': cumulative_returns}),
            'Cumulative Returns',
        )

        # Add all
        for sheet in [cum_rets_sheet, stats_sheet, dd_sheet, weights_sheet]:
            report_wb.add_sheet(sheet)
        report_wb.save(file_loc)
        return file_loc

    def create_alpha_scores(self):
        market_data = self._market_data
        validity = self.validity_table
        results = call_market_data_alpha_fn(self.alpha, market_data)
        results = results[validity].fillna(limit=2, method='ffill')
        return results

    @property
    def validity_table(self):
        return  (~(self._market_data['px_last'].isnull())).shift(250).fillna(False) #hehe - so that it can compute shit.

    def long_short_deciles(self, scores):
        deciles = (scores.rank(axis=1, pct=True) * 9).fillna(-1).astype(int)
        long_scores = self.hyperparameters.get('long_short_deciles').get('long', set())
        short_scores = self.hyperparameters.get('long_short_deciles').get('short', set())
        if long_scores is None and short_scores is None:
            return scores
        long_deciles = deciles.isin(long_scores)
        short_deciles = deciles.isin(short_scores)
        portfolio = long_deciles.astype(int) - short_deciles.astype(int)
        return portfolio

    def normalize(self, portfolio):
        if self.hyperparameters.get('should_normalize', False):
             return max_abs_scaler(portfolio)
        return portfolio

    def construct_portfolios(self, dates, s_score_portfolio, s_market_data, dask_client):
        batches = partition_all(15, dates)
        futs = []
        for batch in batches:
            futs.append(dask_client.submit(score_portfolio_construction,
                batch, s_score_portfolio, s_market_data,
                methodology=self.hyperparameters['portfolio_construction'],
                minimum_securities=self.minimum_securities,
                volatility_target=self.volatility_target
            ))
        cache_keys = dask_client.gather(futs)
        return cache_keys

    def backtest_scores(self, dates, s_score_portfolio, s_market_data, dask_client):
        result = self.construct_portfolios(dates, s_score_portfolio, s_market_data, dask_client)
        results = [pd.read_parquet(x) for x in result]
        portfolio = pd.concat(results)
        portfolio = portfolio[~portfolio.index.duplicated(keep='first')]
        from apex.toolz.pandas import localize
        portfolio.index = pd.to_datetime(portfolio.index)
        portfolio = localize(portfolio)
        portfolio = self.slow_trading_down(portfolio)
        return self.analyze_portfolio(portfolio, self._market_data['returns'])

    def analyze_portfolio(self, weights, security_returns):
        weights = weights.copy()
        security_returns = security_returns.copy()
        w_valid = weights[weights.abs().sum(axis=1) > 0].index.tolist()
        weights = weights.reindex(w_valid)
        security_returns = security_returns.reindex(w_valid)
        from apex.alpha.analysis import get_sign_adjusted_results, alpha_performance_stats
        weights.columns = [ApexSecurity.from_id(x).parsekyable_des for x in weights.columns]
        security_returns.columns = [ApexSecurity.from_id(x).parsekyable_des for x in security_returns.columns]
        c = set(weights.columns.tolist() + security_returns.columns.tolist())
        c = sorted(c)
        weights = weights.T.reindex(c).fillna(0).T
        security_returns = security_returns.T.reindex(c).fillna(0).T
        sign_adjusted_results = get_sign_adjusted_results(weights.copy(), security_returns)
        sign_adjusted_results['weights']['cash'] = 0.0
        weights, returns = sign_adjusted_results['weights'], sign_adjusted_results['returns']
        stats = alpha_performance_stats(weights, returns)
        result = {
            'weights': weights,
            'attribution': sign_adjusted_results['attribution'],
            'factor_sign': sign_adjusted_results['factor_sign'],
            'returns': returns,
            'stats': stats
        }
        return result

    def slow_trading_down(self, weights):
        """
        This is one way of doing it - it's what rob does.

        I think it might be better to do what I did at CMU:
        Estimate a cross-sectional regression on diff of weights
            -> Given today's diffs, what do we expect tomorrows diffs to be?
        Then trade for the day after portfolio = todays port + tomorrows diffs
        Because of the regression it will smooth out the trading I think - it's like an l2 reg.

        """
        smoothing = self.hyperparameters['weight_smoothing']
        smooth_method = SMOOTHING_METHODS[self.hyperparameters.get('weight_smoothing_method', 'ewm')]
        new_weights = smooth_method(weights, period = smoothing)
        returns = self._market_data['returns'].reindex(new_weights.dropna(how = 'all').index.tolist())
        portfolio_std = new_weights.shift(1) * returns
        portfolio_std = portfolio_std.sum(axis = 1)
        portfolio_std = portfolio_std.std() * np.sqrt(261)
        return new_weights * self.volatility_target/portfolio_std

DEFAULT_HYPERPARAMETERS = {
    'signal_smoothing': 1,
    'smoothing_method': 'ewm',
    'should_rank': False,
    'weight_smoothing': 40,
    'long_short_deciles': {'long': {9, 8, 7}, 'short': {0, 1, 2}},
    'portfolio_construction': 'risk_budget'
}

def get_alpha_model_run(alpha_fn, universe_name, hyperparameters):
    model = ApexMarketAlphaRun(
        name=alpha_fn.__name__,
        alpha=alpha_fn,
        universe=APEX_UNIVERSES[universe_name],
        universe_name=universe_name,
        minimum_securities=12,
        hyperparameters=hyperparameters
    )
    return model

def compute_alpha_model_run(model):
    run_res = model.run()
    return run_res

def create_and_compute_model(alpha_fn, universe_name, hyperparameters):
    model = get_alpha_model_run(alpha_fn, universe_name, hyperparameters)
    return compute_alpha_model_run(model), model


def compute_alpha_reports(alpha, universe='AMNA'):
    clt = ApexDaskClient()
    BASE_HP = {
        'signal_smoothing': 1,
        'smoothing_method': 'ewm',
        'should_rank': False,
        'weight_smoothing': 1,
        'long_short_deciles': {'long': {9, 8, 7, 6, 5}, 'short': {0, 1, 2, 3, 4}},
        'portfolio_construction': 'risk_budget',
    }
    results = []
    model = None
    for signal_smoothing in [1, 3, 5, 10, 20]:
        for weight_smoothing in [1, 2, 3, 4, 5, 10, 15, 20, 40, 60]:
            hp = BASE_HP.copy()
            hp['signal_smoothing'] = signal_smoothing
            hp['weight_smoothing'] = weight_smoothing
            if model is None:
                model = get_alpha_model_run(alpha, universe, hp)
                continue
            model._hyperparameters.append(hp)
    r = model.run()
    results.append(r)
    return results