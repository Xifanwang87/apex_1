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
from apex.toolz.arctic import ArcticApex
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
from apex.toolz.functools import isnotnone


APEX__HISTORICAL_FIELDS_TO_LOAD = [
    'eqy_sh_out',
    'fund_net_asset_val',
    'short_int',
    'cur_mkt_cap'
]

APEX__MARKET_DATA_FIELDS = ['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns']


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

def apex__default_availability(dataset, min_market_cap=25, min_price=3, min_dollar_volume=1e5, min_universe_stocks=5):
    """
    Default availability for simplification in future
    """
    close_prices = dataset['px_last']
    dollar_volume = close_prices * dataset['px_volume']

    median_price_filter = close_prices.rolling(time=250).median() > min_price
    dollar_volume_filter = dollar_volume.rolling(time=250).median() > min_dollar_volume

    availability = median_price_filter & dollar_volume_filter
    return availability


@UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace='apex:flow:v1.1:historical_data', asdict=True, should_cache_fn=isnotnone)
def _apex__historical_data(*tickers):
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

def apex__historical_data(tickers):
    from apex.toolz.dask import ApexDaskClient
    split = partition_all(50, tickers)
    pool = ApexDaskClient()

    @retry(Exception, tries=3)
    def get_data(tickers):
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

def apex__create_raw_data(tickers, ds=None):
    """
    Dataset creation logic
    """
    # Datapoints
    historical_data = apex__historical_data(tickers)
    market_data = apex__market_data(tickers)

    # To xarray dataset
    dataset = [historical_data, market_data]
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


def compute_financial_indicators(ds):
    ds['volatility_factor'] = ds['returns'].rolling(time=252).std()
    ds['momentum_12-1'] = ds['returns'].rolling(time=252).sum() - ds['returns'].rolling(time=20).sum()
    ds['short_term_mr'] = -ds['returns'].rolling(time=20).sum()
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
    ds['universe:default'] = availability
    ds = ds.sel(time=slice(start_date, end_date))
    ds = compute_financial_indicators(ds)
    return ds

def apex__create_universe_dataset(tickers, postprocess=True):
    batches = partition_all(100, tickers)
    raw_data = []
    for batch in batches:
        batch_data = apex__create_raw_data(batch)
        raw_data.append(batch_data)
    raw_data = xr.merge(raw_data)
    if postprocess:
        processed_data = apex__postprocess_dataset(raw_data)
        return processed_data
    else:
        return raw_data
