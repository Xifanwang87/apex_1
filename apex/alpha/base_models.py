import math
import time
import typing
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


import funcy
import pandas as pd
from dataclasses import dataclass, field
from distributed import fire_and_forget
from scipy.optimize import Bounds, minimize
from toolz import partition_all
import numpy as np
import joblib
import pygmo as pg
import pyomo as po
from apex.data.access import get_security_market_data
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.toolz.arctic import ArcticApex
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.dicttools import keys, values
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.universe import APEX_UNIVERSES
from joblib import Parallel, delayed, parallel_backend


def call_market_data_alpha_fn(fn, market_data):
    return fn(opens=market_data['px_open'].fillna(method='ffill'),
              highs=market_data['px_high'].fillna(method='ffill'),
              lows=market_data['px_low'].fillna(method='ffill'),
              closes=market_data['px_last'].fillna(method='ffill'),
              returns=market_data['returns'].fillna(0),
              volumes=market_data['px_volume'].fillna(0))

def rolling_mean(df, period=10):
    return df.rolling(period).mean()

def ewm(df, period=10):
    return df.ewm(span=period).mean()

def rolling_median(df, period=10):
    return df.rolling(period).median()

def rank(df, subtract_mean=True):
    result = df.rank(axis=1, pct=True)
    if subtract_mean:
        result = result.subtract(result.mean(axis=1), axis=0)
    return result

SMOOTHING_METHODS = {
    'ewm': ewm,
    'median': rolling_median,
    'mean': rolling_mean
}

class risk_budget_weights:
    def __init__(self, dim=None, covariance=None, volatility_target=None, scores=None):
        self.dim = dim
        self.covariance = covariance
        self.volatility_target = volatility_target
        self.scores = scores

    def fitness(self, weights):
        port_var = weights.T @ self.covariance @ weights
        mcr_abs = np.abs(np.diag(weights) @ self.covariance @ weights)
        mcr_abs = mcr_abs/port_var
        objective_fn = np.sum(np.power(mcr_abs - np.abs(self.scores)/np.sum(np.abs(self.scores)), 2))
        they_are_wts = (np.sum(np.abs(weights)) - 5) # At most 5 turns
        they_are_wts_two = 0 - np.sum(np.abs(weights)) # At most 0% cash
        and_we_vol_target = (np.sqrt(weights.T @ self.covariance @ weights) * np.sqrt(252) - self.volatility_target)
        and_signs_weights_and_scores_match = np.sum(np.sign(weights) != np.sign(self.scores)).astype(np.float64)
        return [objective_fn, and_signs_weights_and_scores_match, and_we_vol_target,  they_are_wts, they_are_wts_two]

    def get_bounds(self):
        # Bounds are based on the scores.
        upper_bounds, lower_bounds = [], []
        for x in self.scores:
            if x > 0:
                upper_bounds.append(3)
                lower_bounds.append(0)
            else:
                upper_bounds.append(0)
                lower_bounds.append(-3)

        return (lower_bounds, upper_bounds)

    def get_name(self):
        return "Risk Budgeted Weights"

    def get_nic(self):
        return 2

    def get_nec(self):
        return 2

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


def construct_risk_budget_weights(signal, covariance, volatility_target=0.1):
    """
    Problem: maximize sum(abs(signal_i)*log(abs(signal_i)))
    s.t. w'Cw < sigma
    """
    risk_budget_sma = risk_budget_weights(dim=signal.values.size, covariance=covariance.values,
                                          volatility_target=volatility_target,
                                          scores=signal.values)
    nl = pg.ipopt()
    nl.set_string_option("hessian_approximation", "limited-memory")
    nl.set_numeric_option("tol", 1E-5)
    algo = pg.algorithm(nl)
    pop = pg.population(risk_budget_sma, 3)
    pop = algo.evolve(pop)
    result = pd.Series(pop.champion_x, signal.index)
    return result


def construct_variance_parity_weights(signal, covariance, volatility_target=0.2):
    """
    weights = sigma/sum(sigma) * sign(signal)
    """
    variance = np.diag(covariance)
    weights = np.sign(signal) * variance / np.sum(variance)
    portfolio_variance = weights.T @ np.diag(variance) @ weights
    portfolio_vol = np.sqrt(252 * portfolio_variance)
    return weights * volatility_target/portfolio_vol

def construct_dollar_parity_weights(signal, covariance, volatility_target=0.2):
    """
    weights = 1/n * sign(signal)
    """
    return np.sign(signal) * np.full(signal.size, 1/signal.size)

def construct_parsimonious_weights(signal, covariance, volatility_target=0.2):
    """
    Averaging out the 3 different weights because of uncertainty over relative weights.
    """
    result = construct_volatility_parity_weights(signal, covariance, volatility_target=0.2)
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
        return construct_risk_budget_weights(signal, covariance, volatility_target)
    elif methodology == 'volatility_parity':
        return construct_variance_parity_weights(signal, covariance, volatility_target)
    elif methodology == 'dollar_parity':
        return construct_dollar_parity_weights(signal, covariance, volatility_target)
    elif methodology == 'parsimonious':
        return construct_parsimonious_weights(signal, covariance, volatility_target)

def batch_portfolio_construction(signal_batch, market_data, methodology='risk_budget', minimum_securities=6, volatility_target=0.2):
    results = {}
    signal_batch[signal_batch == 0.0] = np.nan
    signal_batch = signal_batch.dropna(how='all').dropna(how='all', axis=1).fillna(0)
    returns = market_data['returns'][signal_batch.columns].rolling(2).mean().iloc[-250:].fillna(0)
    covariance = (lw_cov(returns) + oas_cov(returns))/2
    for date in signal_batch.index:
        results[date] = construct_portfolio(signal_batch.loc[date],
                                            covariance,
                                            methodology=methodology,
                                            minimum_securities=minimum_securities,
                                            volatility_target=volatility_target)
    return results


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
        'portfolio_construction' in ('risk_budget', 'volatility_parity', 'dollar_parity')
    }

    TODO
    a) split data
    b) long_short_deciles
    c)
    """
    name: str
    hyperparameters: dict
    alpha: typing.Callable
    universe: list # security universe to compute the factor for.
    data_split: list = field(default_factory=default_split)
    volatility_target: float = field(default=0.2)
    minimum_securities: int = field(default=6)
    _market_data: typing.Any = field(init=False)
    dask_clt: typing.Any = field(init=False)
    def __post_init__(self):
        self._market_data = get_security_market_data(self.universe)
        self.dask_clt = ApexDaskClient()

    def run(self):
        scores = self.create_alpha_scores()
        results = self.backtest_scores(scores, self._market_data)
        final_results = {
            'name': self.name,
            'backtest_results': results,
            'parameters': {
                'hyperparameters': self.hyperparameters,
                'universe': self.universe,
                'volatility_target': self.volatility_target
            }
        }
        return final_results

    def create_alpha_scores(self):
        market_data = self._market_data
        validity = ~(market_data['px_last'].isnull())
        results = call_market_data_alpha_fn(self.alpha, market_data)
        results = results[validity].fillna(limit=2, method='ffill')
        return results

    def long_short_deciles(self, scores):
        deciles = (scores.rank(axis=1, pct=True) * 9).fillna(-1).astype(int)
        long_scores = self.hyperparameters.get('long_short_deciles').get('long', {9, 8})
        short_scores = self.hyperparameters.get('long_short_deciles').get('short', {1, 2}
        if long_scores is None and short_scores is None:
            return scores
        long_deciles = deciles.isin(long_scores)
        short_deciles = deciles.isin(short_scores)
        portfolio = long_deciles.astype(int) - short_deciles.astype(int)
        return portfolio

    def backtest_scores(self, scores, market_data):
        smoothed_scores = self.smooth_signal(scores)
        ranked_scores = self.rank_signal(smoothed_scores)
        score_portfolio = self.long_short_deciles(ranked_scores)
        # Now I need to normalize by maxabs
        score_portfolio = max_abs_scaler(score_portfolio)
        total_secs = (~score_portfolio[score_portfolio != 0].isnull()).sum(axis=1)
        total_secs[total_secs < self.minimum_securities] = np.nan
        total_secs = total_secs.dropna()
        portfolio = {}
        batches = list(partition_all(5, total_secs.index.tolist()))
        batch_inputs = [score_portfolio.loc[batch[0]:batch[-1] + pd.DateOffset(days=1)] for batch in batches]
        dask_clt = self.dask_clt
        dask = ApexDaskClient()
        with joblib.parallel_backend("dask", scatter=[market_data]):
            result = Parallel(n_jobs=75)(delayed(batch_portfolio_construction)(
                    batch,
                    market_data,
                    methodology='risk_budget',
                    minimum_securities=self.minimum_securities,
                    volatility_target=self.volatility_target) for batch in batch_inputs)
        v = [pd.concat(x) for x in result]
        portfolio = pd.concat(v)
        portfolio = portfolio[~portfolio.index.duplicated(keep='first')]
        portfolio = portfolio.unstack().fillna(0).sort_index()
        slowed_trading_portfolio = self.slow_trading_down(portfolio)
        results = {
            'raw_optimization_results': result,
            'base_portfolio': portfolio,
            'slowed_down': slowed_trading_portfolio,
            'score_portfolio': score_portfolio,
            'ranked_scores': ranked_scores,
            'smoothed_scores': smoothed_scores,
            'scores': scores,
            'stats': {
                'base': self.compute_stats(portfolio),
                'slowed': self.compute_stats(slowed_trading_portfolio)
            }
        }
        return results


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

    def slow_trading_down(self, weights):
        """
        This is one way of doing it - it's what rob does.

        I think it might be better to do what I did at CMU:
        Estimate a cross-sectional regression on diff of weights
            -> Given today's diffs, what do we expect tomorrows diffs to be?
        Then trade for the day after portfolio = todays port + tomorrows diffs
        Because of the regression it will smooth out the trading I think - it's like an l2 reg.

        """
        smoothing = self.hyperparameters.get('weight_smoothing', 1)
        smooth_method = SMOOTHING_METHODS[self.hyperparameters.get('weight_smoothing_method', 'ewm')]
        return smooth_method(weights, period = smoothing)
