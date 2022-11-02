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
from apex.toolz.bloomberg import ApexBloomberg, should_cache_market_data_fn
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
from apex.toolz.functools import isnotnone


APEX__FUNDAMENTAL_FIELDS_TO_LOAD = [
    'asset_turnover',
    'average_dividend_yield',
    'book_val_per_sh',
    'bs_lt_borrow',
    'bs_tot_asset',
    'bs_tot_val_of_shares_repurchased',
    'cash_and_marketable_securities',
    'cf_cash_from_oper',
    'cf_free_cash_flow',
    'cur_ratio',
    'ebitda_growth',
    'ebitda',
    'enterprise_value',
    'eqy_dps',  # dividend per share
    'ev_to_t12m_ebitda',
    'extern_eqy_fnc',
    'free_cash_flow_equity',
    'gross_margin',
    'is_adjusted_gross_profit',
    'is_adjusted_operating_income',
    'is_comp_eps_adjusted',
    'is_comp_sales',
    'is_dil_eps_cont_ops',
    'is_inc_bef_xo_item',
    'is_oper_inc',
    'is_tot_oper_exp',
    'net_operating_assets',
    'return_on_inv_capital',
    'short_and_long_term_debt',
    'tang_book_val_per_sh',
    'trail_12m_free_cash_flow',
    'trail_12m_net_sales',
]

APEX__HISTORICAL_FIELDS_TO_LOAD = [
    'cur_mkt_cap',
    'rsk_bb_implied_cds_spread',
    'shareholder_yield',
    'short_int_ratio',
    'earn_yld_hist',
    'eeps_nxt_yr',
    'announcement_dt',
    'eqy_sh_out',
    'eqy_float',
    'eqy_free_float_pct'
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
    'ewm_vol_40d_hl',
    'px_to_sales',
    'buyback_yield',
    'buyback_yield_alt',
    'cashflow_to_debt',
    'price_to_52-week_high',
    'price_to_52-week_low',
    '50_200_moving_avgs',
    '1y_pct_change_debt',
    '1y_pct_change_oper_assets',
    'net_oper_assets_to_ev',
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

def apex__default_availability(dataset, min_market_cap=50, min_price=1, min_dollar_volume=1e5, min_universe_stocks=2):
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

    availability = availability.where(availability)
    return availability


def apex__fundamental_data(tickers, namespace='apex:v1.5:fundamental_data'):
    @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace=namespace, asdict=True, should_cache_fn=should_cache_market_data_fn)
    def _apex__fundamental_data(*tickers):
        from toolz import partition_all
        from apex.toolz.dask import ApexDaskClient
        split = partition_all(10, tickers)
        result = []
        pool = ApexDaskClient()

        @retry(Exception, tries=3)
        def get_data(tickers):
            bbg = ApexBloomberg()
            return bbg.fundamentals(tickers, APEX__FUNDAMENTAL_FIELDS_TO_LOAD.copy())

        for group in split:
            result.append(pool.submit(get_data, group))
        result = pool.gather(result)

        data = pd.concat(result, axis=1)
        data.index = data.index + pd.DateOffset(days=1)
        tickers_w_data = sorted(set(data.columns.get_level_values(0)))
        result = {x: data.get(x, None) for x in tickers}
        for ticker in tickers:
            if ticker not in result:
                result[ticker] = None
        return result

    data = _apex__fundamental_data(*tickers)
    return pd.concat(data, axis=1).swaplevel(axis=1).sort_index(axis=1)

def create_historical_data_fn(namespace='apex:v1.5:historical_data'):
    @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace=namespace, asdict=True, should_cache_fn=should_cache_market_data_fn)
    def historical_data(*tickers):
        from apex.toolz.dask import ApexDaskClient
        split = partition_all(10, tickers)
        result = []
        pool = ApexDaskClient()

        @retry(Exception, tries=3)
        def get_data(tickers):
            bbg = ApexBloomberg()
            return bbg.history(tickers, APEX__HISTORICAL_FIELDS_TO_LOAD.copy())

        for group in split:
            result.append(pool.submit(get_data, group))
        result = pool.gather(result)
        data = pd.concat(result, axis=1)

        tickers_w_data = sorted(set(data.columns.get_level_values(0)))
        result = {x: data[x] for x in tickers_w_data}
        for ticker in tickers:
            if ticker not in result:
                result[ticker] = None
        return result
    return historical_data

def apex__historical_data(tickers, namespace='apex:v1.5:historical_data'):
    from apex.toolz.dask import ApexDaskClient
    split = partition_all(50, tickers)
    pool = ApexDaskClient()

    @retry(Exception, tries=3)
    def get_data(tickers):
        _apex__historical_data = create_historical_data_fn(namespace=namespace)
        return _apex__historical_data(*tickers)

    result = []
    for group in split:
        future = pool.submit(get_data, group)
        future_res = future.result()
        result.append(pd.concat(future_res, axis=1).swaplevel(axis=1).sort_index(axis=1))
    data = pd.concat(result, axis=1)
    return data

def apex__market_data(tickers):
    data = apex__amd(*tickers, parse=True)
    return data

def apex__create_raw_data(tickers, ds=None, fundamentals=True, historical_data=True, market_data=True):
    """
    Dataset creation logic
    """
    # Datapoints
    dataset = []
    if fundamentals:
        dataset.append(apex__fundamental_data(tickers))
    if historical_data:
        dataset.append(apex__historical_data(tickers))
    if market_data:
        dataset.append(apex__market_data(tickers))

    # To xarray dataset
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
    ds['short_term_mr'] = -ds['returns'].rolling(time=20).sum()
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
    ds['px_to_sales'] = ds['enterprise_value']/ds['is_comp_sales']
    ds['buyback_yield'] = ds['bs_tot_val_of_shares_repurchased'] / ds['enterprise_value']
    ds['buyback_yield_alt'] = ds['bs_tot_val_of_shares_repurchased'] / ds['cur_mkt_cap']
    ds['cashflow_to_debt'] = ds['is_inc_bef_xo_item'] / ds['bs_lt_borrow']
    ds['price_to_52-week_high'] = ds['px_last']/ds['px_last'].rolling(time=252).max()
    ds['price_to_52-week_low'] = ds['px_last']/ds['px_last'].rolling(time=252).min()
    ds['50_200_moving_avgs'] = ds['px_last'].rolling(time=50).mean()/ds['px_last'].rolling(time=200).mean()
    ds['1y_pct_change_debt'] = ds['bs_lt_borrow']/ds['bs_lt_borrow'].shift(time=252)
    ds['1y_pct_change_oper_assets'] = (ds['net_operating_assets'] - ds['net_operating_assets'].shift(time=252))/ds['enterprise_value']
    ds['net_oper_assets_to_ev'] = ds['net_operating_assets']/ds['enterprise_value']

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
    from apex.toolz.bloomberg import ApexBloomberg
    bbg = ApexBloomberg()
    currently_available = bbg.reference(px_last.columns.tolist(), 'COMPOSITE_LISTING_STATUS')['COMPOSITE_LISTING_STATUS'] == 'Y'
    tickers_trading = currently_available[currently_available].index.tolist()
    px_last[tickers_trading] = px_last[tickers_trading].fillna(method='ffill')

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
        ds[var] = df[var].fillna(method='ffill')[availability]
    ds['returns'] = returns.to_pandas().fillna(0)
    ds = xr.Dataset(data_vars=ds)
    return ds

def apex__postprocess_dataset(data, end_date=None, start_date='1990-01-01'):
    if end_date is None:
        today = pd.Timestamp.now(tz='America/Chicago')
        if today.hour < 17:
            today = today - pd.DateOffset(days=1)
        end_date = today.strftime('%Y-%m-%d')
    data = data.astype(float)
    ds = apex__data_cleanup_pipeline(data)
    del data
    availability = apex__data_cleanup_availability(ds)
    ds['universe:basic'] = availability
    ds['universe:default'] = apex__default_availability(ds)
    ds = ds.sel(time=slice(start_date, end_date))
    ds = compute_financial_indicators(ds)
    return ds

def apex__create_universe_dataset(tickers, postprocess=True):
    batches = partition_all(15, tickers)
    raw_data = []
    from apex.toolz.dask import ApexDaskClient
    pool = ApexDaskClient()
    for batch in batches:
        batch_data = pool.submit(apex__create_raw_data, batch)
        raw_data.append(batch_data)
    raw_data = pool.gather(raw_data)
    raw_data = xr.merge(raw_data)
    if postprocess:
        processed_data = apex__postprocess_dataset(raw_data)
        return processed_data
    else:
        return raw_data

def apex__postprocess_dataset(data, end_date=None, start_date='1990-01-01'):
    if end_date is None:
        today = pd.Timestamp.now(tz='America/Chicago')
        if today.hour < 17:
            today = today - pd.DateOffset(days=1)
        end_date = today.strftime('%Y-%m-%d')
    data = data.astype(float)
    ds = apex__data_cleanup_pipeline(data)
    del data
    availability = apex__data_cleanup_availability(ds)
    ds['universe:basic'] = availability
    ds['universe:default'] = apex__default_availability(ds)
    ds = ds.sel(time=slice(start_date, end_date))
    try:
        ds = compute_financial_indicators(ds)
    except:
        # In this case, there is probably no data of a few types. Just return what we have
        # already.
        return ds
    return ds

def apex__create_universe_dataset(tickers, postprocess=True, fundamentals=False, historical_data=True, market_data=True):
    batches = partition_all(10, tickers)
    from apex.toolz.dask import ApexDaskClient
    pool = ApexDaskClient()
    raw_data = []
    for batch in batches:
        batch_data = pool.submit(apex__create_raw_data, batch, fundamentals=fundamentals,
            historical_data=historical_data, market_data=market_data)
        raw_data.append(batch_data)
    raw_data = pool.gather(raw_data)
    raw_data = xr.merge(raw_data)
    if postprocess:
        processed_data = apex__postprocess_dataset(raw_data)
        return processed_data
    else:
        return raw_data