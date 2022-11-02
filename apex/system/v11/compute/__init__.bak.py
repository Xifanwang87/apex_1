import dask
import dask.dataframe as dd
import numba as nb
import numpy as np
import pandas as pd
from dask import delayed
from dask.distributed import get_client
from toolz import partial, valmap

import xarray as xr
from apex.pipelines.volatility import (garman_klass_vol_estimate,
                                       parkinson_vol_estimate,
                                       rogers_satchell_yoon_vol_estimate)
from apex.system.v11.data import APEX__MARKET_DATA_FIELDS, ds_to_df
from apex.toolz.dask import ApexDaskClient
from apex.toolz.itertools import flatten
from apex.system.v11.compute.smart_beta import apex__compute_smart_beta_factors, SMART_BETAS
from apex.system.v11.compute.market_alpha import MARKET_ALPHAS
from dask.delayed import Delayed


TRANSFORMS = {
    'identity': lambda x: x,
    'delayed': lambda x: x.shift(1),

    'ewm2': lambda x: x.ewm(halflife=2).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm50': lambda x: x.ewm(halflife=50).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
}

DELAYED_TRANSFORMS = {x: delayed(TRANSFORMS[x]) for x in TRANSFORMS}

@dask.delayed
def rank_max_abs_scaler(raw_data, axis=0):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        data = data.T
        maxval = data.expanding().max()
        minval = data.expanding().min()
        data = data + minval
        data = data/maxval
        result = data.T
    else:
        data = data.rank(axis=1) - 1
        maxval = data.max(axis=1)
        scale = 1/maxval
        result = data.multiply(scale, axis=0)
    return result


@dask.delayed
def apex__compute_vol_weighted_portfolio(market_data, availability, portfolio, halflife=40):
    returns = market_data['returns']
    vol = returns.ewm(halflife=halflife).std()
    portfolio = portfolio * vol
    portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)[availability]
    return portfolio

@dask.delayed
def apex__compute_inverse_vol_weighted_portfolio(market_data, availability, portfolio, halflife=40):
    returns = market_data['returns']
    vol = returns.ewm(halflife=halflife).std()
    inv_vol = 1/vol
    portfolio = portfolio * inv_vol
    portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)[availability]
    return portfolio


@nb.njit
def _apex__nb_compute_low_turnover_weights(returns, availability, portfolio, blend_multiplier):
    """
    Blend every day at rate = blend multiplier / 2
    """
    new_port = np.zeros_like(portfolio)
    new_port[0] = portfolio[0]
    rets = returns + 1
    for day in range(1, len(portfolio)):
        curr_pos = new_port[day - 1] * rets[day]
        curr_val = np.sum(curr_pos)
        curr_wt = curr_pos/curr_val
        new_pos = curr_wt + (portfolio[day] - curr_wt) * blend_multiplier[day]
        new_pos[~availability[day]] = 0
        new_port[day] = new_pos/np.sum(np.abs(new_pos))
    return new_port

@dask.delayed
def apex__compute_drifting_turnover_constrained_portfolio(market_data, availability, portfolio, turnover_period, variance_halflife=20, maximum_gap_close=0.5):
    returns = market_data['returns'].fillna(0)
    var = returns.ewm(halflife=variance_halflife).var()[availability]
    var = var.median(axis=1).dropna()
    var_per_day = var.expanding().median()

    # compute number of median days equivalent to rebalancing days
    turns_per_variance_day = 1/turnover_period
    blend_multiplier = var/var_per_day * turns_per_variance_day

    # Now let's set up everything for numpy
    np_port = portfolio.fillna(0).values.copy()

    np_returns = returns.reindex(portfolio.index)[portfolio.columns].fillna(0).values
    np_availability = availability.reindex(portfolio.index)[portfolio.columns].values
    np_blend_multiplier = np.minimum(blend_multiplier.reindex(portfolio.index).fillna(0), maximum_gap_close).values
    result = _apex__nb_compute_low_turnover_weights(np_returns, np_availability, np_port, np_blend_multiplier)
    result = pd.DataFrame(result, index=portfolio.index, columns=portfolio.columns)
    return result[availability]

def apex__long_only_transform_pipeline(market_data, availability, raw_alpha):
    """
    1. Transform
    2. Compute portfolio construction (inv vol weighted, vol weighetd, etc)
    """
    base_alpha = rank_max_abs_scaler(raw_alpha)
    results = []
    for cutoff in [0, 0.25, 0.5]:
        for transform_name, transform_fn in DELAYED_TRANSFORMS.items():
            tdata = transform_fn(base_alpha[base_alpha >= cutoff])[availability]
            normalized_tdata = rank_max_abs_scaler(tdata)[availability]
            normalized_tdata = normalized_tdata.divide(normalized_tdata.sum(axis=1), axis=0)
            results.append(normalized_tdata)
    return results

def apex__long_short_transform_pipeline(market_data, availability, raw_alpha):
    """
    1. Transform
    2. Compute portfolio construction (inv vol weighted, vol weighetd, etc)
    """
    base_alpha = rank_max_abs_scaler(raw_alpha)
    results = []
    for cutoff in [0, 0.25, 0.5]:
        for transform_name, transform_fn in DELAYED_TRANSFORMS.items():
            tdata = transform_fn(base_alpha[base_alpha >= cutoff])[availability]
            normalized_tdata = rank_max_abs_scaler(tdata)[availability]
            normalized_tdata = normalized_tdata.subtract(normalized_tdata.mean(axis=1), axis=0)
            normalized_tdata = normalized_tdata.divide(normalized_tdata.abs().sum(axis=1), axis=0)
            results.append(normalized_tdata)
    return results

def apex__create_turnover_constrained_portfolios(market_data, availability, base_portfolio):
    turnover_periods = [10, 20, 40, 70]
    fn = partial(apex__compute_drifting_turnover_constrained_portfolio, market_data, availability, base_portfolio)
    results = [fn(x) for x in turnover_periods]
    return results

def apex__clean_portfolio(x):
    return x.dropna(how='all', axis=0).fillna(0)

def apex__compute_long_only_pipeline(market_data, availability, raw_alpha):
    """
    Steps:
    1. Transform raw alphas
    2. Compute portfolios
    """
    transformed_portfolios = apex__long_only_transform_pipeline(market_data, availability, raw_alpha)

    inv_vol_wt_fn = partial(apex__compute_inverse_vol_weighted_portfolio, market_data, availability)
    # vol_wt_fn = partial(apex__compute_vol_weighted_portfolio, market_data, availability)

    inv_vol_weighted_portfolios = [inv_vol_wt_fn(x) for x in transformed_portfolios]
    # vol_weighted_portfolios = [vol_wt_fn(x) for x in transformed_portfolios]

    turnover_constrain = partial(apex__create_turnover_constrained_portfolios, market_data, availability)
    portfolios = transformed_portfolios + inv_vol_weighted_portfolios
    portfolios = [apex__clean_portfolio(x) for x in portfolios]
    portfolios = flatten([turnover_constrain(x) for x in portfolios])
    return portfolios


def apex__compute_long_short_pipeline(market_data, availability, raw_alpha):
    """
    Steps:
    1. Transform raw alphas
    2. Compute portfolios
    """
    transformed_portfolios = apex__long_short_transform_pipeline(market_data, availability, raw_alpha)

    inv_vol_wt_fn = partial(apex__compute_inverse_vol_weighted_portfolio, market_data, availability)
    # vol_wt_fn = partial(apex__compute_vol_weighted_portfolio, market_data, availability)

    inv_vol_weighted_portfolios = [inv_vol_wt_fn(x) for x in transformed_portfolios]
    # vol_weighted_portfolios = [vol_wt_fn(x) for x in transformed_portfolios]

    turnover_constrain = partial(apex__create_turnover_constrained_portfolios, market_data, availability)
    portfolios = transformed_portfolios + inv_vol_weighted_portfolios
    portfolios = flatten([turnover_constrain(x) for x in portfolios])
    return portfolios


def apex__alpha_wrapper(fn):
    def wrapped(apex_data, availability=None):
        if isinstance(apex_data, xr.Dataset):
            market_data = ds_to_df(apex_data[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns', 'default_availability']])
        else:
            market_data = apex_data[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns', 'default_availability']]

        if availability is None:
            availability = market_data['default_availability'].fillna(False)
        return fn(opens=market_data['px_open'],
                  highs=market_data['px_high'],
                  lows=market_data['px_low'],
                  closes=market_data['px_last'],
                  returns=market_data['returns'],
                  volumes=market_data['px_volume'])[availability]
    return wrapped


def apex__compute_portfolios_with_alpha(market_data, availability, raw_alpha, long_only=True, long_short=True):
    portfolios = {}
    if long_only:
        portfolios['long_only'] = []
        portfolios['long_only'] += apex__compute_long_only_pipeline(market_data, availability, raw_alpha)
        portfolios['long_only'] += apex__compute_long_only_pipeline(market_data, availability, -raw_alpha)
    if long_short:
        portfolios['long_short'] = []
        portfolios['long_short'] += apex__compute_long_short_pipeline(market_data, availability, raw_alpha)
        portfolios['long_short'] += apex__compute_long_short_pipeline(market_data, availability, -raw_alpha)
    return portfolios

def apex__market_alpha_compute_pipeline(alpha_d, market_data, availability, long_only=True, long_short=True):
    """
    Pipeline for market alphas
    """
    portfolios = apex__compute_portfolios_with_alpha(market_data, availability, alpha_d, long_only=long_only, long_short=long_short)
    return portfolios

def apex__market_alpha_pipeline(market_data, availability, long_only=True, long_short=True):
    """
    Market alphas (all)
    """
    assert isinstance(market_data, Delayed)
    assert isinstance(availability, Delayed)
    portfolios = {}
    for alpha_name, alpha_fn in MARKET_ALPHAS.items():
        raw_alpha_d = apex__alpha_wrapper(delayed(alpha_fn))(market_data)
        portfolios[alpha_name] = apex__market_alpha_compute_pipeline(raw_alpha_d,
            market_data, availability, long_only=long_only, long_short=long_short)

    return portfolios

def apex__smart_beta_pipeline(market_data, availability, dataset, long_only=True, long_short=True):
    """
    Pipeline for smart betas (all)
    """
    assert isinstance(market_data, Delayed)
    assert isinstance(availability, Delayed)
    assert isinstance(dataset, Delayed)
    portfolios = {}
    for factor, smart_beta_fn in SMART_BETAS.items():
        factor_data = delayed(smart_beta_fn)(dataset)
        portfolios[factor] = apex__compute_portfolios_with_alpha(market_data, availability,
            factor_data, long_only=long_only, long_short=long_short)
    return portfolios

def apex__alpha_pipeline(dataset, long_only=True, long_short=True):
    """
    Dataset is simply the universe dataset.
    """

    if isinstance(dataset, xr.Dataset):
        dataset = ds_to_df(dataset[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns', 'default_availability']])
    market_data = dataset[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns', 'default_availability']]
    availability = dataset['default_availability']

    market_data = dask.delayed(market_data)
    availability = dask.delayed(availability)
    dataset = dask.delayed(dataset)

    return {
        'market_alpha': apex__market_alpha_pipeline(market_data, availability, long_only=long_only, long_short=long_short),
        'smart_beta': apex__smart_beta_pipeline(market_data, availability, dataset, long_only=long_only, long_short=long_short),
    }