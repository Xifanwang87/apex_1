import pickle
import shutil
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd
import redis
from dataclasses import dataclass, field
from funcy import zipdict
from joblib import Parallel, delayed, parallel_backend
from toolz import curry, merge, partition_all

import xarray as xr
#from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import ApexBloomberg
from apex.toolz.bloomberg import apex__adjusted_market_data as apex__amd
from apex.toolz.bloomberg import (get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.caches import (FUNDAMENTAL_DATA_CACHING, MARKET_DATA_CACHING,
                               UNIVERSE_DATA_CACHING)
from apex.toolz.deco import lazyproperty, retry
from apex.toolz.functools import isnotnone
from apex.toolz.deco import lazyproperty, retry
from functools import lru_cache
import logging
from functools import lru_cache

APEX__FUNDAMENTAL_FIELDS_TO_LOAD = [
    'asset_turnover',
    'average_dividend_yield',
    'book_val_per_sh',
    'bs_lt_borrow',
    'bs_tot_asset',
    'cf_cash_from_oper',
    'cf_free_cash_flow',
    'cur_ratio',
    'ebitda_growth',
    'ebitda',
    'enterprise_value',
    'eqy_dps',  # dividend per share
    'ev_to_t12m_ebitda',
    'gross_margin',
    'is_comp_eps_adjusted',
    'is_comp_sales',
    'is_adjusted_gross_profit',
    'is_tot_oper_exp',
    'return_on_inv_capital',
    'is_inc_bef_xo_item',
    'trail_12m_net_sales',
    'trail_12m_free_cash_flow',
    'short_and_long_term_debt',
    'free_cash_flow_equity',
    'cash_and_marketable_securities',
    'is_dil_eps_cont_ops',
]

APEX__HISTORICAL_FIELDS_TO_LOAD = [
    'cur_mkt_cap',
    'rsk_bb_implied_cds_spread',
    'shareholder_yield',
    'short_int_ratio',
    'earn_yld_hist',
    'eeps_nxt_yr',
]

APEX__MARKET_DATA_FIELDS = ['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns']

APEX__SMART_BETAS = [
    'enterprise_value',
    'ebitda_growth_to_mkt_cap',
    'ev_to_t12m_ebitda',
    'cashflow_yield',
    'cashflow_yield_t12m',
    'earnings_quality',
    'dividend_yield',
    'dividend_yield_t12m',
    'dividend_growth_12m',
    'profitability_factor',
    'debt_to_tot_assets',
    'debt_to_ebitda',
    'debt_to_mkt_val',
    'debt_to_cur_ev',
    'earnings_yield_t12m',
    'earnings_yield_curr',
    'px_to_free_cashflow',
    'px_to_t12m_free_cashflow',
    'pe_ratio',
    'px_to_book_ratio',
    'sales_growth',
    'volatility_factor',
    'momentum_12-1',
    'leverage_and_liquidity',
    'berry_ratio',
    'cashflow_roe',
    'roic_change',
    'operating_efficiency_factor',
    'asset_growth',
    'earnings_revision_chg',
    'size_factor',
    'credit_factor',
    'credit_factor_alt',
    'ewm_vol_40d_hl'
]

def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

###
### DATA PREPROCESSING/CLEANING
###
def df_to_xr(df):
    """
    df = dataframe with index = time and columns = tickers
    """
    return xr.DataArray(df, dims=['time', 'ticker'])

def ds_to_df(dataset):
    """
    dataset has dims ticker and time
    """
    return dataset.to_dataframe().unstack('ticker')

def df_to_ds(df):
    cols = set(df.columns.get_level_values(0))
    result = {}
    for c in cols:
        result[c] = df_to_xr(df[c])
    return xr.Dataset(data_vars=result)

def apex__default_availability(dataset, min_market_cap=250, min_price=10, min_dollar_volume=1e6, min_universe_stocks=6):
    """
    Default availability for simplification in future
    """
    close_prices = dataset['px_last']
    market_cap = dataset['cur_mkt_cap']
    dollar_volume = close_prices * dataset['px_volume']

    median_price_filter = close_prices.rolling(time=250).median() > min_price
    market_cap_filter = market_cap.rolling(time=250).median() > min_market_cap
    dollar_volume_filter = dollar_volume.rolling(time=250).median() > min_dollar_volume

    availability = median_price_filter & market_cap_filter & dollar_volume_filter
    num_stocks_filter = availability.sum(axis=1) > min_universe_stocks
    availability = availability * num_stocks_filter

    availability = availability.where(availability == True)
    return availability


@UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace='apex:v1.1:fundamental_data', asdict=True)
def _apex__fundamental_data(*tickers):
    bbg = ApexBloomberg()
    data = bbg.fundamentals(tickers, APEX__FUNDAMENTAL_FIELDS_TO_LOAD.copy())
    data.index = data.index + pd.DateOffset(days=1)
    tickers_w_data = sorted(set(data.columns.get_level_values(0)))
    result = {x: data[x] for x in tickers_w_data}
    for ticker in tickers:
        if ticker not in result:
            result[ticker] = None
    return result

def apex__fundamental_data(tickers):
    data = _apex__fundamental_data(*tickers)
    return pd.concat(data, axis=1).swaplevel(axis=1).sort_index(axis=1)

@UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace='apex:v1.1:fundamental_data', asdict=True)
def _apex__historical_data(*tickers):
    bbg = ApexBloomberg()
    data = bbg.history(tickers, APEX__HISTORICAL_FIELDS_TO_LOAD.copy())
    tickers_w_data = sorted(set(data.columns.get_level_values(0)))
    result = {x: data[x] for x in tickers_w_data}
    for ticker in tickers:
        if ticker not in result:
            result[ticker] = None
    return result

def apex__historical_data(tickers):
    data = _apex__historical_data(*tickers)
    return pd.concat(data, axis=1).swaplevel(axis=1).sort_index(axis=1)

def apex__market_data(tickers):
    data = apex__amd(*tickers, parse=True)
    return data

def apex__create_raw_data(tickers, ds=None):
    """
    Dataset creation logic
    """
    # Datapoints
    fundamental_data = apex__fundamental_data(tickers)
    historical_data = apex__historical_data(tickers)
    market_data = apex__market_data(tickers)

    # To xarray dataset
    dataset = [fundamental_data, historical_data, market_data]
    as_field_df = lambda df: {x: df[x] for x in set(df.columns.get_level_values(0))}
    dataset = merge(*[as_field_df(x) for x in dataset])
    dataset = xr.Dataset(data_vars={x: xr.DataArray(dataset[x], dims=['time', 'ticker']) for x in dataset})
    if ds is not None:
        dataset = dataset.sel(time=slice(ds, None))
    return dataset



def parkinson_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices['px_high']
    lows = undl_prices['px_low']

    result = 1.0 / (4 * np.log(2))
    result *= ((np.log(highs / lows)**2))
    return result

def rogers_satchell_yoon_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices['px_high']
    lows = undl_prices['px_low']
    closes = undl_prices['px_last']
    opens = undl_prices['px_open']

    first_term = np.log(highs / closes)
    second_term = np.log(highs / opens)
    third_term = np.log(lows / closes)
    fourth_term = np.log(lows / opens)

    return (first_term * second_term + third_term * fourth_term)


def garman_klass_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices['px_high']
    lows = undl_prices['px_low']
    closes = undl_prices['px_last']
    closes_shifted = undl_prices['px_last'].shift(time=1)

    first_term = (0.5 * (np.log(highs / lows) ** 2))
    second_term = ((2 * np.log(2) - 1) * (np.log(closes / closes_shifted) ** 2))

    return first_term - second_term

def default_volatility(undl_prices, vol_days=252):
    gk_vol = garman_klass_vol_estimate(undl_prices)
    rsy_vol = rogers_satchell_yoon_vol_estimate(undl_prices)
    p_vol = parkinson_vol_estimate(undl_prices)
    gk_vol = gk_vol.where(gk_vol > 0)
    rsy_vol = rsy_vol.where(rsy_vol > 0)
    p_vol = p_vol.where(p_vol > 0)

    gk_vol = xr.DataArray(gk_vol.to_pandas().fillna(method='ffill').ewm(halflife=vol_days).mean())
    rsy_vol = xr.DataArray(rsy_vol.to_pandas().fillna(method='ffill').ewm(halflife=vol_days).mean())
    p_vol = xr.DataArray(p_vol.to_pandas().fillna(method='ffill').ewm(halflife=vol_days).mean())
    return (((gk_vol + rsy_vol + p_vol)/3.0) ** (0.5)) * np.sqrt(252)



def crossectional_rank_scaler(data):
    """
    Axis=1 means computing it in time-series way.
    """
    data = data.rank('ticker') - 1
    maxval = data.max('ticker')
    scale = 1/maxval
    return data * scale

def earnings_quality_factor(dataset):
    data = dataset['is_comp_eps_adjusted'].to_pandas()
    availability = dataset['universe:basic'].to_pandas()
    data = data.pct_change()
    data[data == 0.0] = np.nan
    results = {}
    for c in data.columns:
        results[c] = data[c].dropna().ewm(span=8).std()
    results = pd.concat(results, axis=1).reindex(availability.index).fillna(method='ffill') * availability
    return df_to_xr(results)


def profitability_factor(dataset):
    income_xo = dataset['is_inc_bef_xo_item']
    assets_to_normalize_by = dataset['bs_tot_asset']
    cfo = dataset['cf_cash_from_oper']
    roa = income_xo/assets_to_normalize_by
    droa = roa - roa.shift(time=252)
    cfo = cfo / assets_to_normalize_by
    f_accruals = cfo - roa
    result = crossectional_rank_scaler(f_accruals)
    result += crossectional_rank_scaler(cfo)
    result += crossectional_rank_scaler(roa)
    result += crossectional_rank_scaler(droa)
    return result

def leverage_and_liquidity_signals(dataset):
    long_term_borrow = dataset['bs_lt_borrow']
    current_ratio = dataset['cur_ratio']
    assets_to_normalize_by = dataset['bs_tot_asset']

    leverage = long_term_borrow/assets_to_normalize_by
    idleverage = crossectional_rank_scaler(leverage - leverage.shift(time=252))
    idliquid = crossectional_rank_scaler(current_ratio - current_ratio.shift(time=252))
    return (idliquid + idleverage)

def compute_financial_indicators(ds):
    ds['enterprise_value'] = ds['cur_mkt_cap'] + ds['bs_lt_borrow'] - ds['cash_and_marketable_securities']
    ds['ebitda_growth_to_mkt_cap'] = (ds['ebitda'] - ds['ebitda'].shift(time=252))/ds['cur_mkt_cap'].shift(time=252)
    ds['ev_to_t12m_ebitda'] = ds['ebitda'].rolling(time=252).mean()/ds['enterprise_value']
    ds['cashflow_yield'] = ds['cf_free_cash_flow']/ds['enterprise_value']
    ds['cashflow_yield_t12m'] = ds['cf_free_cash_flow'].rolling(time=252).mean()/ds['enterprise_value']
    ds['earnings_quality'] = earnings_quality_factor(ds)
    ds['dividend_yield'] = ds['eqy_dps']/ds['px_last']
    ds['dividend_yield_t12m'] = ds['eqy_dps'].rolling(time=252).mean()/ds['px_last']
    ds['dividend_growth_12m'] = (ds['eqy_dps'] - ds['eqy_dps'].shift(time=252))/ds['px_last']
    ds['profitability_factor'] = profitability_factor(ds)
    ds['debt_to_tot_assets'] = ds['bs_lt_borrow'] / ds['bs_tot_asset']
    ds['debt_to_ebitda'] = ds['bs_lt_borrow'] / ds['ebitda']
    ds['debt_to_mkt_val'] = ds['bs_lt_borrow'] / ds['cur_mkt_cap']
    ds['debt_to_cur_ev'] = ds['bs_lt_borrow'] / ds['enterprise_value']
    ds['earnings_yield_t12m'] =  ds['ebitda'].rolling(time=252).mean() / ds['cur_mkt_cap']
    ds['earnings_yield_curr'] =  ds['ebitda'] / ds['cur_mkt_cap']
    ds['px_to_free_cashflow'] = ds['cf_free_cash_flow'] / ds['cur_mkt_cap']
    ds['px_to_t12m_free_cashflow'] = ds['cf_free_cash_flow'].rolling(time=252).mean() / ds['cur_mkt_cap']
    ds['pe_ratio'] = ds['ebitda']/ds['cur_mkt_cap']
    ds['px_to_book_ratio'] = ds['book_val_per_sh'] / ds['px_last']
    ds['sales_growth'] = (ds['is_comp_sales'] - ds['is_comp_sales'].shift(time=252))/ds['is_comp_sales'].shift(time=252)
    ds['volatility_factor'] = ds['returns'].rolling(time=252).std()
    ds['momentum_12-1'] = ds['returns'].rolling(time=252).sum() - ds['returns'].rolling(time=20).sum()
    ds['leverage_and_liquidity'] = leverage_and_liquidity_signals(ds)
    ds['berry_ratio'] = ds['is_adjusted_gross_profit'] / ds['is_tot_oper_exp']
    ds['cashflow_roe'] = ds['free_cash_flow_equity']/ds['cur_mkt_cap'] # to see how cheap it is getting
    ds['roic_change'] = ds['return_on_inv_capital'] - ds['return_on_inv_capital'].shift(time=252)
    ds['operating_efficiency_factor'] = ds['gross_margin'] * ds['asset_turnover']
    ds['asset_growth'] = (ds['bs_tot_asset'] - ds['bs_tot_asset'].shift(time=252))/ds['bs_tot_asset'].shift(time=252)
    ds['earnings_revision_chg'] = (ds['eeps_nxt_yr'] - ds['eeps_nxt_yr'].shift(time=22*3))
    ds['size_factor'] = ds['cur_mkt_cap']
    ds['credit_factor'] = ds['rsk_bb_implied_cds_spread']
    ds['credit_factor_alt'] = crossectional_rank_scaler(ds['rsk_bb_implied_cds_spread']) + crossectional_rank_scaler(ds['debt_to_cur_ev'])


    returns = ds['returns'].to_pandas()
    vol40 = df_to_xr(returns.ewm(halflife=40).std()) * ds['universe:basic']
    vol20 = df_to_xr(returns.ewm(halflife=20).std()) * ds['universe:basic']
    vol10 = df_to_xr(returns.ewm(halflife=10).std()) * ds['universe:basic']
    vol100 = df_to_xr(returns.ewm(halflife=100).std()) * ds['universe:basic']

    ds['ewm_vol_40d_hl'] = vol40
    ds['ewm_vol_10d_hl'] = vol10
    ds['ewm_vol_20d_hl'] = vol20
    ds['ewm_vol_100d_hl'] = vol100
    ds['volatility'] = default_volatility(ds, vol_days=20)
    return ds



def apex__data_cleanup_availability(raw_data):
    """
    For data cleaning the only thing we are looking for is non-null prices.
    """
    px_last = raw_data['px_last'].to_dataframe().unstack('ticker')['px_last']
    px_last = px_last.apply(fill_nas_series)
    availability = ~px_last.isnull()
    return availability

def apex__data_cleanup_pipeline(raw_data):
    """
    0. Input: xarray dataset generated by apex__build_universe_dataset
    1. compute availability with nans on px_last after fill_nas_series
    """
    returns = raw_data['returns'].copy()
    dataset = raw_data.ffill('time')
    availability = apex__data_cleanup_availability(raw_data)
    df = ds_to_df(dataset)
    ds = {}
    for var in dataset.data_vars:
        ds[var] = df[var][availability]
    ds['returns'] = returns.to_pandas()
    ds = xr.Dataset(data_vars=ds)
    return dataset

def apex__postprocess_dataset(data, end_date=None, start_date='1990-01-01'):
    if end_date is None:
        today = pd.Timestamp.now(tz='America/Chicago')
        if today.hour < 17:
            today = today - pd.DateOffset(days=1)
        end_date = today.strftime('%Y-%m-%d')
    data = data.astype(float)
    availability = apex__data_cleanup_availability(data)
    df = ds_to_df(data.ffill('time'))
    ds = {}
    for var in data.data_vars:
        ds[var] = df[var][availability]

    availability[~availability] = np.nan
    ds['universe:basic'] = availability
    data = xr.Dataset(data_vars=ds)
    data['universe:default'] = apex__default_availability(data)
    data = data.sel(time=slice(start_date, end_date))
    data = compute_financial_indicators(data)
    return data

def apex__create_universe_dataset(tickers):
    raw_data = apex__create_raw_data(tickers)
    processed_data = apex__postprocess_dataset(raw_data)
    return processed_data

import pickle
import shutil
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path

import numpy as np
import numba as nb
import pandas as pd
from dataclasses import dataclass, field
from toolz import curry, merge, partition_all

import xarray as xr
import logging
from functools import lru_cache

#####################
###### STRATEGY STUFF
#####################

import numpy as np
import numba as nb
import empyrical as ep
import pyfolio as pf
import inflection

def _nb_fill_nas_mtx(arr, inplace=False):
    if not isinstance(arr, np.ndarray):
        arr = arr.values
    if inplace:
        out = arr
    else:
        out = arr.copy()
    arr_last_ix = arr.shape[0] - 1
    last_valid_indices = arr_last_ix - (~np.isnan(arr[::-1])).argmax(axis=0) - 1 ## Not isnan, reversed, argmax. Meaning first index? yep. exactly.
    for row_idx in range(1, out.shape[0]):
        for col_idx in range(0, out.shape[1]):
            if np.isnan(out[row_idx, col_idx]) and last_valid_indices[col_idx] > row_idx:
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    return out

def zion__forward_fill_nulls(data, inplace=False):
    import numpy as np
    import numba as nb

    if isinstance(data, np.ndarray):
        return _nb_fill_nas_mtx(data, inplace=inplace)
    elif isinstance(data, pd.DataFrame):
        return pd.DataFrame(_nb_fill_nas_mtx(data.values, inplace=inplace), index=data.index, columns=data.columns)
    elif isinstance(data, xr.DataArray):
        return xr.DataArray(_nb_fill_nas_mtx(data.values, inplace=inplace), dims=['time', 'ticker'], coords=[data.time, data.ticker])
    elif isinstance(data, xr.Dataset):
        return data.apply(_nb_fill_nas_mtx, inplace=inplace)

@nb.njit
def zion_nb__exponential_smoothing(series, alpha):
    out_arr = series.copy()

    n, m = out_arr.shape

    for c_ix in range(m):
        for day in range(1, n):
            if np.isnan(out_arr[day - 1, c_ix]):
                continue
            out_arr[day, c_ix] = alpha[c_ix] * series[day, c_ix] + out_arr[day - 1, c_ix] * (1-alpha[c_ix])
    return out_arr

def zion__exponential_smoothing_forecast_fit(input_data, forecast_period):
    """
    Fits exponential smoothing for particular forecast period.
    """
    from scipy.optimize import minimize
    assert isinstance(input_data, (xr.DataArray, pd.DataFrame))
    input_data = zion__forward_fill_nulls(input_data)
    if isinstance(input_data, xr.DataArray):
        target_data = input_data.values
        in_data = input_data.shift(time=forecast_period).values
        alpha_names = input_data.ticker.values.tolist()
    elif isinstance(input_data, pd.DataFrame):
        target_data = input_data.values
        in_data = input_data.shift(forecast_period).values
        alpha_names = input_data.columns.tolist()

    def loss_fn(alpha):
        smoothed = zion_nb__exponential_smoothing(in_data, alpha)
        errors = target_data - smoothed
        errors = np.nansum(errors ** 2)
        return errors

    n, m = input_data.shape
    start_alpha_guess = np.zeros(m) + 0.5

    alphas = minimize(loss_fn, start_alpha_guess, method='L-BFGS-B', bounds=[(0, 1)] * m).x
    result = zion_nb__exponential_smoothing(input_data.values, alphas)

    if isinstance(input_data, xr.DataArray):
        result = xr.DataArray(result, dims=['time', 'ticker'], coords=[input_data.time, input_data.ticker])
    elif isinstance(input_data, pd.DataFrame):
        result = pd.DataFrame(result, index=input_data.index, columns=input_data.columns)
    return {
        'data': result,
        'alphas': pd.Series(alphas, index=alpha_names)
    }

def zion__exponential_smoothing(input_data, halflife=None, alpha=None):
    if halflife is not None:
        assert halflife > 0
        alpha = 1 - np.exp(np.log(0.5)/halflife)
    result = zion_nb__exponential_smoothing(input_data.values, [alpha] * input_data.shape[1])

    if isinstance(input_data, xr.DataArray):
        result = xr.DataArray(result, dims=['time', 'ticker'], coords=[input_data.time, input_data.ticker])
    elif isinstance(input_data, pd.DataFrame):
        result = pd.DataFrame(result, index=input_data.index, columns=input_data.columns)
    return result

PORTFOLIO_STAT_FNS = {
    'cumulative_returns': ep.cum_returns_final,
    'sharpe_ratio': ep.sharpe_ratio,
    'calmar_ratio': ep.calmar_ratio,
    'omega_ratio': ep.omega_ratio,
    'sortino_ratio': ep.sortino_ratio,
}


def zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns'].fillna(0)
    portfolio = signal_dataset.shift(time=1).fillna(0) # Addl lag

    portfolio_drifted = (portfolio.shift(time=1) * (1 + returns))
    portfolio_drifted = portfolio_drifted / np.abs(portfolio_drifted.sum('ticker'))

    trades = np.abs(portfolio.fillna(0) - portfolio_drifted.fillna(0))
    strategy_returns = (portfolio.shift(time=1) * returns - trades * transaction_costs * 0.0001).sum('ticker')
    del trades
    del portfolio_drifted
    del returns
    return strategy_returns

def zion__performance_stats(weights, portfolio_returns):
    weights = weights.copy()
    weights['cash'] = 0
    perf_stats = pf.timeseries.perf_stats(portfolio_returns, positions=weights)
    perf_stats.index = map(inflection.underscore, perf_stats.index.str.replace(' ', '_'))
    del weights
    return perf_stats

def zion_xr__backtest_portfolio_weights(market_data, signal_dataset, transaction_costs=15):
    """
    Signal dataset is an xr Dataset with vars = portfolios
    """
    signal_returns = zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs)
    if isinstance(signal_dataset, xr.Dataset):
        signals = list(signal_dataset.data_vars.keys())
        stats = pd.DataFrame({x: zion__performance_stats(signal_dataset[x].to_pandas(), signal_returns[x].to_pandas()) for x in signals}).T
    elif isinstance(signal_dataset, xr.DataArray):
        stats = zion__performance_stats(signal_dataset.to_pandas(), signal_returns.to_pandas())
    # Now need to do this...
    return {
        'stats': stats,
        'returns': signal_returns,
    }

def zion__compute_rolling_returns_stats(returns, window=252):
    stats = {}
    for ix, day in enumerate(returns.index):
        if ix < window:
            start_ix = 0
        else:
            start_ix = ix - window
        stats[day] = pf.timeseries.perf_stats(returns.iloc[start_ix:ix])
    result = pd.DataFrame(stats)
    result.index = map(inflection.underscore, result.index.str.replace(' ', '_'))
    return result.T



def zion__performance_stats(weights, portfolio_returns):
    weights = weights.copy()
    weights['cash'] = 0
    perf_stats = pf.timeseries.perf_stats(portfolio_returns, positions=weights)
    perf_stats.index = map(inflection.underscore, perf_stats.index.str.replace(' ', '_'))
    del weights
    return perf_stats

def zion__compute_strategy_returns(market_data, portfolio, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns']
    portfolio = portfolio.shift(1) # Addl lag

    portfolio_drifted = (portfolio.shift(1) * (1 + returns))
    portfolio_drifted = portfolio_drifted.divide(portfolio_drifted.abs().sum(axis=1), axis=0)

    trades = portfolio - portfolio_drifted
    return (portfolio.shift(1) * returns - trades.abs() * transaction_costs * 0.0001).sum(axis=1)


def zion__backtest_portfolio_weights(market_data, portfolio, transaction_costs=15):
    """
    Simple backtest with transaction costs and slippage
    """
    strategy_returns = zion__compute_strategy_returns(market_data, portfolio, transaction_costs).reindex(portfolio.index)
    stats = zion__performance_stats(portfolio, strategy_returns)
    return {
        'stats': stats,
        'returns': strategy_returns,
    }

from functools import partial
import numpy as np


def _nb_fill_nas_mtx(arr, inplace=False):
    if not isinstance(arr, np.ndarray):
        arr = arr.values
    if inplace:
        out = arr
    else:
        out = arr.copy()
    arr_last_ix = arr.shape[0] - 1
    last_valid_indices = arr_last_ix - (~np.isnan(arr[::-1])).argmax(axis=0) - 1 ## Not isnan, reversed, argmax. Meaning first index? yep. exactly.
    for row_idx in range(1, out.shape[0]):
        for col_idx in range(0, out.shape[1]):
            if np.isnan(out[row_idx, col_idx]) and last_valid_indices[col_idx] > row_idx:
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    return out

def zion__forward_fill_nulls(data, inplace=False):
    import numpy as np
    import numba as nb

    if isinstance(data, np.ndarray):
        return _nb_fill_nas_mtx(data, inplace=inplace)
    elif isinstance(data, pd.DataFrame):
        return pd.DataFrame(_nb_fill_nas_mtx(data.values, inplace=inplace), index=data.index, columns=data.columns)
    elif isinstance(data, xr.DataArray):
        return xr.DataArray(_nb_fill_nas_mtx(data.values, inplace=inplace), dims=['time', 'ticker'], coords=[data.time, data.ticker])
    elif isinstance(data, xr.Dataset):
        return data.apply(_nb_fill_nas_mtx, inplace=inplace)


@nb.njit
def zion_nb__exponential_smoothing(series, alpha):
    out_arr = series.copy()

    n, m = out_arr.shape

    for c_ix in range(m):
        for day in range(1, n):
            if np.isnan(out_arr[day - 1, c_ix]):
                continue
            out_arr[day, c_ix] = alpha[c_ix] * series[day, c_ix] + out_arr[day - 1, c_ix] * (1-alpha[c_ix])
    return out_arr

def zion__exponential_smoothing_forecast_fit(input_data, forecast_period):
    """
    Fits exponential smoothing for particular forecast period.
    """
    from scipy.optimize import minimize
    assert isinstance(input_data, (xr.DataArray, pd.DataFrame))
    input_data = zion__forward_fill_nulls(input_data)
    if isinstance(input_data, xr.DataArray):
        target_data = input_data.values
        in_data = input_data.shift(time=forecast_period).values
        alpha_names = input_data.ticker.values.tolist()
    elif isinstance(input_data, pd.DataFrame):
        target_data = input_data.values
        in_data = input_data.shift(forecast_period).values
        alpha_names = input_data.columns.tolist()

    def loss_fn(alpha):
        smoothed = zion_nb__exponential_smoothing(in_data, alpha)
        errors = target_data - smoothed
        errors = np.nansum(errors ** 2)
        return errors

    n, m = input_data.shape
    start_alpha_guess = np.zeros(m) + 0.5

    alphas = minimize(loss_fn, start_alpha_guess, method='L-BFGS-B', bounds=[(0, 1)] * m).x
    result = zion_nb__exponential_smoothing(input_data.values, alphas)

    if isinstance(input_data, xr.DataArray):
        result = xr.DataArray(result, dims=['time', 'ticker'], coords=[input_data.time, input_data.ticker])
    elif isinstance(input_data, pd.DataFrame):
        result = pd.DataFrame(result, index=input_data.index, columns=input_data.columns)
    return {
        'data': result,
        'alphas': pd.Series(alphas, index=alpha_names)
    }

def zion__exponential_smoothing(input_data, halflife=None, alpha=None):
    if halflife is not None:
        assert halflife > 0
        alpha = 1 - np.exp(np.log(0.5)/halflife)
    result = zion_nb__exponential_smoothing(input_data.values, [alpha] * input_data.shape[1])

    if isinstance(input_data, xr.DataArray):
        result = xr.DataArray(result, dims=['time', 'ticker'], coords=[input_data.time, input_data.ticker])
    elif isinstance(input_data, pd.DataFrame):
        result = pd.DataFrame(result, index=input_data.index, columns=input_data.columns)
    return result

PORTFOLIO_STAT_FNS = {
    'cumulative_returns': ep.cum_returns_final,
    'sharpe_ratio': ep.sharpe_ratio,
    'calmar_ratio': ep.calmar_ratio,
    'omega_ratio': ep.omega_ratio,
    'sortino_ratio': ep.sortino_ratio,
}


def zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns'].fillna(0)
    portfolio = signal_dataset.shift(time=1).fillna(0) # Addl lag

    portfolio_drifted = (portfolio.shift(time=1) * (1 + returns))
    portfolio_drifted = portfolio_drifted / np.abs(portfolio_drifted.sum('ticker'))

    trades = np.abs(portfolio.fillna(0) - portfolio_drifted.fillna(0))
    strategy_returns = (portfolio.shift(time=1) * returns - trades * transaction_costs * 0.0001).sum('ticker')
    del trades
    del portfolio_drifted
    del returns
    return strategy_returns

def zion__performance_stats(weights, portfolio_returns):
    weights = weights.copy()
    weights['cash'] = 0
    perf_stats = pf.timeseries.perf_stats(portfolio_returns, positions=weights)
    perf_stats.index = map(inflection.underscore, perf_stats.index.str.replace(' ', '_'))
    del weights
    return perf_stats

def zion_xr__backtest_portfolio_weights(market_data, signal_dataset, transaction_costs=15):
    """
    Signal dataset is an xr Dataset with vars = portfolios
    """
    signal_returns = zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs)
    if isinstance(signal_dataset, xr.Dataset):
        signals = list(signal_dataset.data_vars.keys())
        stats = pd.DataFrame({x: zion__performance_stats(signal_dataset[x].to_pandas(), signal_returns[x].to_pandas()) for x in signals}).T
    elif isinstance(signal_dataset, xr.DataArray):
        stats = zion__performance_stats(signal_dataset.to_pandas(), signal_returns.to_pandas())
    # Now need to do this...
    return {
        'stats': stats,
        'returns': signal_returns,
    }

def zion__compute_rolling_returns_stats(returns, window=252):
    stats = {}
    for ix, day in enumerate(returns.index):
        if ix < window:
            start_ix = 0
        else:
            start_ix = ix - window
        stats[day] = pf.timeseries.perf_stats(returns.iloc[start_ix:ix])
    result = pd.DataFrame(stats)
    result.index = map(inflection.underscore, result.index.str.replace(' ', '_'))
    return result.T


def zion__compute_strategy_returns(market_data, portfolio, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns']
    portfolio = portfolio.shift(1) # Addl lag

    portfolio_drifted = (portfolio.shift(1) * (1 + returns))
    portfolio_drifted = portfolio_drifted.divide(portfolio_drifted.abs().sum(axis=1), axis=0)

    trades = portfolio - portfolio_drifted
    return (portfolio.shift(1) * returns - trades.abs() * transaction_costs * 0.0001).sum(axis=1)


def zion__backtest_portfolio_weights(market_data, portfolio, transaction_costs=15):
    """
    Simple backtest with transaction costs and slippage
    """
    strategy_returns = zion__compute_strategy_returns(market_data, portfolio, transaction_costs).reindex(portfolio.index)
    stats = zion__performance_stats(portfolio, strategy_returns)
    return {
        'stats': stats,
        'returns': strategy_returns,
    }

def df_to_xr(df):
    """
    df = dataframe with index = time and columns = tickers
    """
    return xr.DataArray(df, dims=['time', 'ticker'])

def ds_to_df(dataset):
    """
    dataset has dims ticker and time
    """
    return dataset.to_dataframe().unstack('ticker').sort_index().sort_index(axis=1)

def df_to_ds(df):
    cols = set(df.columns.get_level_values(0))
    result = {}
    for c in cols:
        result[c] = df_to_xr(df[c])
    return xr.Dataset(data_vars=result)

TS_TRANSFORMS = {
    'identity': lambda x: x,

    'ewm1': partial(zion__exponential_smoothing, halflife=1),
    'ewm3': partial(zion__exponential_smoothing, halflife=3),
    'ewm5': partial(zion__exponential_smoothing, halflife=5),
    'ewm10': partial(zion__exponential_smoothing, halflife=10),
    'ewm20': partial(zion__exponential_smoothing, halflife=20),
    'ewm40': partial(zion__exponential_smoothing, halflife=40),
    'ewm80': partial(zion__exponential_smoothing, halflife=80),
    'ewm160': partial(zion__exponential_smoothing, halflife=160),
    'ewm200': partial(zion__exponential_smoothing, halflife=200),
}

TS_DIFFED_TRANSFORMS = {
#     'ewm5_diffed': lambda x: x.ewm(halflife=5).mean().diff(),
#     'ewm20_diffed': lambda x: x.ewm(halflife=5).mean().diff(),
#     'ewm50_diffed': lambda x: x.ewm(halflife=40).mean().diff(),
    'ewm200_diffed': lambda x: partial(zion__exponential_smoothing, halflife=200)(x).diff('time'),
}

@nb.njit
def zion__nb_compute_low_turnover_weights(returns, availability, portfolio, blend_multiplier):
    """
    Blend every day at rate = blend multiplier / 2

    portfolio: np.ndarray shaped (n_time, n_securities)
    """
    new_port = np.zeros_like(portfolio)
    rets = returns + 1
    initialized = False
    for day in range(1, len(portfolio)):
        if not initialized:
            if np.nansum(np.abs(portfolio[day - 1])) < 1e-5:
                continue
            else:
                new_port[day - 1] = portfolio[day - 1]/np.nansum(np.abs(portfolio[day - 1]))
                initialized = True

        if np.sum(availability[day]) == 0:
            new_port[day] = new_port[day - 1]
            continue

        curr_pos = new_port[day - 1] * rets[day]
        curr_pos[~availability[day]] = 0
        curr_val = np.nansum(np.abs(curr_pos))
        curr_wt = curr_pos/curr_val

        new_pos = curr_wt + (portfolio[day] - curr_wt) * blend_multiplier[day]
        new_pos[~availability[day]] = 0
        port_val = np.nansum(np.abs(new_pos))
        port_day = new_pos / port_val
        new_port[day] = port_day
    return new_port

def zion__blend_transformer(dataset, universe, portfolio_pipeline, blend_reference_quantile=0.99, blend_var_halflife=1, blend_turnover_days=1):
    """
    Transforms the portfolios in pipeline into blended ones

        blending_options = {
            'blend_reference_quantile': 0.99,
            'blend_var_halflife': 1,
            'blend_turnover_days': 1
        }
    """
    blend_var_halflife = 1
    var = xr.DataArray(dataset.returns.to_pandas().ewm(halflife=blend_var_halflife).var())
    var = var.median('ticker')
    var_exp_median = xr.DataArray(var.to_pandas().expanding().quantile(blend_reference_quantile))

    # compute number of median days equivalent to rebalancing days
    turns_per_variance_day = 1/blend_turnover_days
    blend_multiplier = np.minimum(var/var_exp_median * turns_per_variance_day, 1)

    np_rets = dataset.returns.fillna(0).values
    np_av = dataset[universe].fillna(False).astype(bool).values
    np_bld = blend_multiplier.values
    results = []
    for p in portfolio_pipeline:
        try:
            np__port = zion__nb_compute_low_turnover_weights(np_rets, np_av, p.fillna(0).values, np_bld)
        except:
            print(p.shape, np_av.shape, blend_multiplier.shape)
            raise
        results.append(xr.DataArray(np__port, dims=['time', 'ticker'], coords=[dataset.time, dataset.ticker]))
    return results

def zion__signal_expander(dataset, signal, cutoff=0, long_short=False, universe='universe:base', prefix=None, blending_options=None,
                         post_cutoff=False):
    """
    The timeseries expander takes a signal, expands it across time, and yields back all the expansions
    This signal must be normalized to center at zero per stock

    It is always long-only
    """
    vol = dataset['volatility']
    market_cap = dataset['cur_mkt_cap']
    signal = dataset[signal]
    availability = dataset[universe]
    signal = signal.rank('ticker')

    signal = (signal - signal.min('ticker')) / (signal.max('ticker') - signal.min('ticker'))

    if long_short:
        signal = signal * 2 - 1
        signal = signal.where(np.abs(signal) > cutoff) # (cutoff, 1) or (-1, -cutoff) and (cutoff, 1)
        signal = signal - signal.mean('ticker')
    else:
        signal = signal.where(np.abs(signal) > cutoff) # (cutoff, 1) or (-1, -cutoff) and (cutoff, 1)

    signal_ds = {}

    MULTIPLIERS = {
        'base': 1,
        'inv_vol': 1/(vol * 100),
        'mkt_cap': market_cap,
    }


    def compute_derived_portfolios(t_signal):
        t_signals = []
        for multiplier_name, multiplier in MULTIPLIERS.items():
            base_port_t = (t_signal * multiplier)
            t_signals.append(base_port_t)
        return t_signals


    transforms = merge(TS_TRANSFORMS, TS_DIFFED_TRANSFORMS)
    transformed_signals = []
    for transform_name, transform in transforms.items():
        """
        Now you need to optimize this.
        Put all t_signals on a single dataset (everything right before normalization/multiplying by multipliers)
        """
        # Transform
        t_signal = transform(signal)
        t_signal = xr.DataArray(t_signal, dims=['time', 'ticker'])
        t_signal = (t_signal - t_signal.min('ticker')) / (t_signal.max('ticker') - t_signal.min('ticker'))
        transformed_signals += compute_derived_portfolios(t_signal)
        t_signal = -t_signal
        t_signal = (t_signal - t_signal.min('ticker')) / (t_signal.max('ticker') - t_signal.min('ticker'))
        transformed_signals += compute_derived_portfolios(t_signal)


    t_ds_dict = {f't_signal={ix}': data for ix, data in enumerate(transformed_signals)}
    t_ds = xr.Dataset(data_vars=t_ds_dict)

    t_ds = (t_ds - t_ds.min('ticker'))/(t_ds.max('ticker') - t_ds.min('ticker'))
    t_ds = t_ds / np.abs(t_ds).sum('ticker')
    t_ds_dvs = [f'{prefix}_{x}' for x in t_ds.data_vars.keys()]
    t_ds_dvs = dict(zip(t_ds.data_vars.keys(), t_ds_dvs))
    t_ds_dict = {t_ds_dvs[x]: t_ds[x] for x in t_ds_dict}
    t_ds.rename(name_dict=t_ds_dvs, inplace=True)

    if post_cutoff:
        t_ds = (t_ds - t_ds.min('ticker')) /(t_ds.max('ticker') - t_ds.min('ticker'))

        if long_short:
            t_ds = t_ds * 2 - 1
            t_ds = t_ds.where(np.abs(t_ds) > cutoff) # (cutoff, 1) or (-1, -cutoff) and (cutoff, 1)
            t_ds = t_ds - t_ds.mean('ticker')
        else:
            t_ds = t_ds.where(np.abs(t_ds) > cutoff) # (cutoff, 1) or (-1, -cutoff) and (cutoff, 1)

    blending_options = {
        'blend_reference_quantile': 0.99,
        'blend_var_halflife': 1,
        'blend_turnover_days': 10
    }
    blended_portfolios = zion__blend_transformer(dataset, universe, t_ds_dict.values(), **blending_options)
    t_ds_bld = dict(zip(t_ds_dict.keys(), blended_portfolios))
    t_ds_bld = {x + '_blended': t_ds_bld[x] for x in t_ds_dict}
    t_ds_bld = xr.Dataset(data_vars=t_ds_bld) * availability
    t_ds_bld = t_ds_bld / np.abs(t_ds_bld).sum('ticker')

    #portfolios = xr.merge([t_ds_bld, t_ds])
    return t_ds_bld

def zion__compute_expansions(dataset, universe, signal, cutoffs=[0, 50, 75], cutoff_locations=[False], long_short=False):
    portfolios = []
    for cutoff in cutoffs:
        for post_cutoff in cutoff_locations:
            if long_short:
                prefix = f'ls_cutoff={cutoff}__post_cutoff={post_cutoff}'
            else:
                prefix = f'lo_cutoff={cutoff}__post_cutoff={post_cutoff}'

            portfolios.append(zion__signal_expander(dataset, signal, universe=universe,
                                                    prefix=prefix, post_cutoff=post_cutoff,
                                                    cutoff=cutoff/100, long_short=long_short))
    portfolios = xr.merge(portfolios)
    return portfolios.fillna(0)

def zion__create_simple_optimal_portfolio(dataset, portfolios, pct_portfolios=0.05):
    portfolios = portfolios.fillna(0)
    portfolios = portfolios / np.abs(portfolios).sum('ticker')
    num_portfolios = len(portfolios.data_vars) * pct_portfolios
    portfolio_returns = zion_xr__compute_strategy_returns(dataset, portfolios, 10).to_dataframe()
    cumprod_strategies = (1+portfolio_returns).cumprod() # Expanding
    cumprod_strategies[cumprod_strategies < 1] = np.nan
    weights = (cumprod_strategies.rank(axis=1, ascending=False) <= num_portfolios).astype(float)
    weights = weights.divide(weights.sum(axis=1), axis=0).ewm(halflife=20).mean()
    final_results = {}
    for column in portfolios.data_vars:
        final_results[column] = portfolios[column] * xr.DataArray(weights[column], dims=['time'])
    final_results = xr.Dataset(final_results)
    final_portfolio = sum(final_results[x].fillna(0) for x in final_results.data_vars)
    final_portfolio = final_portfolio / final_portfolio.sum('ticker')
    return final_portfolio

def zion__build_universes(dataset):
    def large_caps_availability(dataset, percentile=0.9):
        """
        Default availability for simplification in future
        """
        availability = dataset['universe:default']
        market_cap = dataset['cur_mkt_cap'] * availability
        availability_mkt_cap = market_cap.rolling(time=250).median().rank('ticker', pct=True) >= percentile
        availability = availability * availability_mkt_cap
        return availability.where(availability > 0)

    def mid_caps_availability(dataset, bottom_percentile=0.25, top_percentile=0.9):
        """
        Default availability for simplification in future
        """
        availability = dataset['universe:default']
        market_cap = dataset['cur_mkt_cap'] * availability
        availability_bot = market_cap.rolling(time=250).median().rank('ticker', pct=True) >= bottom_percentile
        availability_top = market_cap.rolling(time=250).median().rank('ticker', pct=True) <= top_percentile
        availability = availability * availability_top * availability_bot
        return availability.where(availability > 0)


    def small_caps_availability(dataset, percentile=0.25):
        """
        Default availability for simplification in future
        """
        availability = dataset['universe:default']
        market_cap = dataset['cur_mkt_cap'] * availability
        availability_mkt_cap = market_cap.rolling(time=250).median().rank('ticker', pct=True) <= percentile
        availability = availability * availability_mkt_cap
        return availability.where(availability > 0)

    def ev_to_ebitda_universes(dataset):
        default_availability = dataset['universe:default']
        ev_to_ebitda = dataset['ev_to_t12m_ebitda'].rank('ticker')
        ev_to_ebitda = ev_to_ebitda - ev_to_ebitda.min('ticker')
        ev_to_ebitda = ev_to_ebitda/(ev_to_ebitda.max('ticker') - ev_to_ebitda.min('ticker'))
        high_ev_to_ebitda = ~(ev_to_ebitda.where(ev_to_ebitda > 0.75).isnull())
        low_ev_to_ebitda = ~(ev_to_ebitda.where(ev_to_ebitda <= 0.25).isnull())

        high_ev_to_ebitda = high_ev_to_ebitda * default_availability
        low_ev_to_ebitda = low_ev_to_ebitda * default_availability
        return high_ev_to_ebitda, low_ev_to_ebitda

    def dividend_yield_universes(dataset):
        default_availability = dataset['universe:default']
        dividend_yield_t12m = dataset['dividend_yield_t12m'].rank('ticker')
        dividend_yield_t12m = dividend_yield_t12m - dividend_yield_t12m.min('ticker')
        dividend_yield_t12m = dividend_yield_t12m/(dividend_yield_t12m.max('ticker') - dividend_yield_t12m.min('ticker'))
        high_div_yld = ~(dividend_yield_t12m.where(dividend_yield_t12m > 0.75).isnull())
        low_div_yld = ~(dividend_yield_t12m.where(dividend_yield_t12m <= 0.25).isnull())

        high_div_yld = high_div_yld * default_availability
        low_div_yld = low_div_yld * default_availability
        return high_div_yld, low_div_yld

    def price_to_earnings_universes(dataset):
        default_availability = dataset['universe:default']
        pe_ratio = dataset['pe_ratio'].rank('ticker')
        pe_ratio = pe_ratio - pe_ratio.min('ticker')
        pe_ratio = pe_ratio/(pe_ratio.max('ticker') - pe_ratio.min('ticker'))
        high_pe_ratio = ~(pe_ratio.where(pe_ratio > 0.75).isnull())
        low_pe_ratio = ~(pe_ratio.where(pe_ratio <= 0.25).isnull())

        high_pe_ratio = high_pe_ratio * default_availability
        low_pe_ratio = low_pe_ratio * default_availability
        return high_pe_ratio, low_pe_ratio

    def leverage_universes(dataset):
        default_availability = dataset['universe:default']
        leverage = dataset['debt_to_tot_assets'].rank('ticker')
        leverage = leverage - leverage.min('ticker')
        leverage = leverage/(leverage.max('ticker') - leverage.min('ticker'))
        high_leverage = ~(leverage.where(leverage > 0.75).isnull())
        low_leverage = ~(leverage.where(leverage <= 0.25).isnull())

        high_leverage = high_leverage * default_availability
        low_leverage = low_leverage * default_availability
        return high_leverage, low_leverage

    def momentum_universes(dataset):
        default_availability = dataset['universe:default']
        momentum = dataset['momentum_12-1'].rank('ticker')
        momentum = momentum - momentum.min('ticker')
        momentum = momentum/(momentum.max('ticker') - momentum.min('ticker'))
        high_momentum = ~(momentum.where(momentum > 0.75).isnull())
        low_momentum = ~(momentum.where(momentum <= 0.25).isnull())

        high_momentum = high_momentum * default_availability
        low_momentum = low_momentum * default_availability
        return high_momentum, low_momentum

    def subuniverses_availability(dataset):
        from apex.toolz.bloomberg import ApexBloomberg
        bbg = ApexBloomberg()
        tickers = dataset.ticker
        tickers = tickers.to_pandas().index.tolist()
        sub_industries = bbg.reference(tickers, 'gics_sub_industry_name').fillna('Other')
        sub_industries.index.name = 'ticker'
        sub_industries = sub_industries.rename(columns={'gics_sub_industry_name': 'sub_industry_name'}).reset_index()
        sub_industries['sub_industry_name'] = sub_industries['sub_industry_name'].map(inflection.parameterize)
        sub_industries['universe_name'] = 'universe:' + sub_industries['sub_industry_name'].str.replace('-','_')
        sub_industry_count = sub_industries.groupby('universe_name').count()['ticker']
        sub_industry_count = sub_industry_count[sub_industry_count > 10]
        sub_industry_subuniverses = sub_industry_count.index.tolist()
        sub_industries = sub_industries.set_index('ticker')
        results = {}
        for subuniverse_name in sub_industry_subuniverses:
            subuniverse = sub_industries[sub_industries['universe_name'] == subuniverse_name]
            tickers_in_subuniverse = sorted(set(subuniverse.index.tolist()))
            tickers_not_in_subuniverse = [x for x in tickers if x not in tickers_in_subuniverse]
            subuniverse = dataset['universe:default'].copy()
            subuniverse.loc[{'ticker': subuniverse.ticker.isin(tickers_not_in_subuniverse)}] = np.nan
            subuniverse.name = subuniverse_name
            results[subuniverse_name] = subuniverse
        return results, sub_industry_subuniverses


    def create_strategy_subuniverses(dataset):
        large_caps_universe = large_caps_availability(dataset)
        mid_caps_universe = mid_caps_availability(dataset)
        small_caps_universe = small_caps_availability(dataset)

        high_leverage, low_leverage = leverage_universes(dataset)
        high_pe_ratio, low_pe_ratio = price_to_earnings_universes(dataset)
        high_div_yld, low_div_yld = dividend_yield_universes(dataset)
        high_ev_to_ebitda, low_ev_to_ebitda = ev_to_ebitda_universes(dataset)
        high_momentum, low_momentum = momentum_universes(dataset)

        dataset, subuniverse_names = subuniverses_availability(dataset)
        dataset['universe:large_cap'] = large_caps_universe
        dataset['universe:mid_cap'] = mid_caps_universe
        dataset['universe:small_cap'] = small_caps_universe
        dataset['universe:high_leverage'] = high_leverage
        dataset['universe:high_pe_ratio'] = high_pe_ratio
        dataset['universe:high_div_yld'] = high_div_yld
        dataset['universe:high_ev_to_ebitda'] = high_ev_to_ebitda
        dataset['universe:high_momentum'] = high_momentum
        dataset['universe:low_leverage'] = low_leverage
        dataset['universe:low_pe_ratio'] = low_pe_ratio
        dataset['universe:low_div_yld'] = low_div_yld
        dataset['universe:low_ev_to_ebitda'] = low_ev_to_ebitda
        dataset['universe:low_momentum'] = low_momentum

        subuniverse_names = subuniverse_names + [
            'universe:large_cap',
            'universe:mid_cap',
            'universe:small_cap',
            'universe:high_leverage',
            'universe:high_pe_ratio',
            'universe:high_div_yld',
            'universe:high_ev_to_ebitda',
            'universe:high_momentum',
            'universe:low_leverage',
            'universe:low_pe_ratio',
            'universe:low_div_yld',
            'universe:low_ev_to_ebitda',
            'universe:low_momentum',
        ]
        # Now let's build the industry universes
        dataset = xr.Dataset(dataset)
        dataset = dataset.where(dataset > 0)
        return dataset.astype(float), subuniverse_names
    subuniverses, subuniverse_names = create_strategy_subuniverses(dataset)
    return xr.merge([dataset, subuniverses]), subuniverse_names


smart_betas = [
    'enterprise_value',
    'ebitda_growth_to_mkt_cap',
    'ev_to_t12m_ebitda',
    'cashflow_yield',
    'cashflow_yield_t12m',
    'earnings_quality',
    'dividend_yield',
    'dividend_yield_t12m',
    'dividend_growth_12m',
    'profitability_factor',
    'debt_to_tot_assets',
    'debt_to_mkt_val',
    'debt_to_cur_ev',
    'earnings_yield_t12m',
    'earnings_yield_curr',
    'px_to_free_cashflow',
    'px_to_t12m_free_cashflow',
    'pe_ratio',
    'px_to_book_ratio',
    'sales_growth',
    'volatility_factor',
    'momentum_12-1',
    'leverage_and_liquidity',
    'berry_ratio',
    'cashflow_roe',
    'roic_change',
    'operating_efficiency_factor',
    'asset_growth',
    'earnings_revision_chg',
    'size_factor',
    'credit_factor',
    'credit_factor_alt',
]