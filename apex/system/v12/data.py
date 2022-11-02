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

@dataclass
class ApexBlackboard:
    """
    Lazy dataset loader for memory conservation.
    """
    name: str = field(default_factory=lambda: uuid.uuid4().hex)
    output_type: str = field(default='xr')
    prefix: str = field(default='apex:v12:dataset')
    clean: bool = field(default=False)
    temporary: bool = field(default=False)
    update: bool = field(default=False)
    ds: str = field(default=None)
    def __post_init__(self):
        if self.temporary:
            self.prefix = 'apex:temporary_blackboard'
        if self.clean:
            self.clear_cache()
        if self.update == 'if_empty':
            if not self.variables:
                self.update_cache()
        elif self.update:
            self.update_cache()

    def clear_cache(self):
        lib = self.cache
        pool = ThreadPoolExecutor()
        f = []
        for var in self.variables:
            f.append(pool.submit(lib.delete, var))

        return len(lib.list_symbols()) == 0

    @property
    @retry(Exception)
    def cache(self):
        arc = ArcticApex()
        lib = arc.get_library(self.library_name)

        sess = arc.session
        sess.set_quota(self.library_name, 100 * 1024 * 1024 * 1024)

        return lib

    def update_cache(self):
        raise NotImplementedError

    @property
    def library_name(self):
        return f'{self.prefix}:{self.name}'

    @property
    def variables(self):
        return self.cache.list_symbols()

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key: str, data: typing.Any):
        return self.write(key, data)

    @retry(Exception)
    def write(self, key, data, metadata=None):
        # Save format always xr
        if isinstance(data, (xr.DataArray, pd.DataFrame)):
            data = xr.DataArray(data, dims=['time', 'ticker'])

        return self.cache.write(key, data, metadata=metadata)

    @retry(Exception)
    def read_fields(self, fields: list, output_type=None):
        pool = ThreadPoolExecutor()

        def cache_read_fn(library_name, fld):
            cache = ArcticApex().get_library(library_name)
            return cache.read(fld).data

        results = pool.map(cache_read_fn, [self.library_name for x in fields], fields)
        results = zipdict(fields, results)
        try:
            return self._format_data_item(xr.Dataset(data_vars=results), output_type=output_type)
        except:
            return results

    @retry(Exception)
    def read(self, key, output_type=None):
        data = self.cache.read(key).data
        """
        TODO: ADD CLOCK TO BLACKBOARD
        ds = self.ds
        if ds:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.loc[:ds]
            elif isinstance(data, xr.DataArray):
                data = data.
        """
        return self._format_data_item(data, output_type=output_type)

    def get(self, key, output_type=None):
        if isinstance(key, str):
            return self.read(key, output_type=output_type)
        else:
            return self.read_fields(key, output_type=output_type)

    @retry(Exception)
    def get_metadata(self, key):
        data = self.cache.read(key).metadata
        return data

    def get_matching(self, regex, output_type=None):
        symbols = self.cache.list_symbols(regex=regex)
        if symbols:
            return self.read_fields(symbols, output_type=output_type)

    def _format_data_item(self, data, output_type=None):
        if output_type is None:
            output_type = self.output_type

        if output_type == 'pd':
            if isinstance(data, xr.DataArray):
                return data.to_pandas()
            elif isinstance(data, xr.Dataset):
                return ds_to_df(data)
            else:
                return data
        elif output_type == 'xr':
            return data
        elif output_type == 'npy':
            return data.values
        return data

    def close(self):
        if self.temporary:
            cache = ArcticApex()
            arc = cache.session
            arc.delete_library(self.library_name)

    def __del__(self):
        self.close()


@dataclass
class ApexTemporaryBlackboard(ApexBlackboard):
    clean: bool = field(default=True)
    temporary: bool = field(default=True)
    time_to_live: int = field(default=60*60*24) # seconds, 1 day
    def __post_init__(self):
        super().__post_init__()
        ttl = pd.Timestamp.now() + pd.DateOffset(seconds=self.time_to_live)
        self.write('__ttl', ttl)

@dataclass
class ApexUniverseBlackboard(ApexBlackboard):
    prefix: str = field(default='apex:v12:universe_dataset')

    def update_cache(self):
        universe = self.name
        cache = self.cache
        dataset = apex__dataset(universe, postprocess=True)
        tickers = dataset.ticker.values.tolist()
        cache.write('universe_metadata', {
            'universe_name': universe,
            'tickers': tickers,
            'last_updated': pd.Timestamp.now(),
        })
        for c, d in dataset.data_vars.items():
            cache.write(c, d, metadata={
                'last_updated': pd.Timestamp.now(),
            })

        del dataset

    @property
    def market_data(self):
        return self.read_fields(APEX__MARKET_DATA_FIELDS)

    @property
    def smart_betas(self):
        return self.read_fields(APEX__SMART_BETAS)

    @property
    def raw_fields(self):
        return self.read_fields(APEX__MARKET_DATA_FIELDS + APEX__HISTORICAL_FIELDS_TO_LOAD + APEX__FUNDAMENTAL_FIELDS_TO_LOAD)

    @classmethod
    def build_universe_dataset(cls, universe_name, tickers):
        r = apex__build_universe_dataset(universe_name, tickers)
        if r:
            return cls(universe=universe_name)


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

def apex__portfolio_dataset(data, portfolio):
    if isinstance(data, pd.DataFrame):
        data = df_to_ds(data.reindex(portfolio.index))
    dataset = data[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns', 'default_availability']]
    dataset['portfolio'] = portfolio
    dataset = dataset.reindex(time=portfolio.index)
    return dataset

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

def apex__create_and_save_raw_data(tickers, filename, ds=None):
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

    # save
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in dataset.data_vars}
    dataset.to_netcdf(filename, encoding=encoding)
    return True

def _read_netcdfs_dataset(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(files)
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

def apex__read_netcdf_dataset(directory, parallel=False, **kwargs):
    directory_glob = directory.rstrip('/') + '/*.nc'
    directory = Path(directory.rstrip('/'))
    files = [str(x) for x in directory.glob('*.nc')]
    if len(files) > 0:
        try:
            if 'autoclose' not in kwargs:
                kwargs['autoclose'] = True
            return xr.open_mfdataset(directory_glob,
                                     parallel=parallel,
                                     concat_dim='ticker',
                                     **kwargs)
        except:
            import traceback
            traceback.print_exc()
            return _read_netcdfs_dataset(files, 'ticker')

def apex__read_dataset(base_directory, group='all'):
    base_directory = Path(base_directory)
    nc_dir = Path(base_directory / 'netcdf')
    z_dir = Path(base_directory / 'zarr')
    if nc_dir.exists():
        return apex__read_netcdf_dataset(str(nc_dir)).load()
    elif z_dir.exists():
        try:
            data = xr.open_zarr(str(z_dir), group=group).load()
            data.coords['ticker'] = data.ticker.astype(str)
            return data
        except:
            print('error opening zarr')
            return apex__read_netcdf_dataset(str(nc_dir)).load()
    else:
        raise NotImplementedError

def apex__save_zarr_groups(data, folder):
    data = data.load()
    data.coords['ticker'] = data.ticker.astype(str)
    data.to_zarr(folder, group='all', mode='w')
    return True

def apex__cleanup_universe_folder(universe_name):
    save_dir = Path(f'/apex.data/apex.universes/{universe_name}/')
    if save_dir.exists():
        shutil.rmtree(f'/apex.data/apex.universes/{universe_name}')
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'raw_data' / 'netcdf').mkdir(parents=True, exist_ok=True)
    else:
        (save_dir / 'raw_data' / 'netcdf').mkdir(parents=True, exist_ok=True)
    return True


def apex__build_universe_dataset(universe_name, tickers, ds=None):
    try:
        from apex.toolz.dask import ApexDaskClient
        split = partition_all(25, tickers)
        apex__cleanup_universe_folder(universe_name)

        def create_chunk_dataset(chunk, ds=None):
            save_dir = Path(f'/apex.data/apex.universes/{universe_name}/raw_data/netcdf')
            chunk_name = uuid.uuid4().hex
            filename = str(save_dir / (chunk_name + '.nc'))
            return apex__create_and_save_raw_data(chunk, filename, ds=ds)

        pool = ApexDaskClient()
        result = []
        for chunk in split:
            result.append(pool.submit(create_chunk_dataset, chunk, ds=ds))

        result = pool.gather(result)
    except:
        import traceback
        traceback.print_exc()
        return False
    return True

def apex__build_zarr_dataset(universe_name):
    data = apex__dataset(universe_name, postprocess=False)
    apex__save_zarr_groups(data, f'/apex.data/apex.universes/{universe_name}/raw_data/zarr')
    return True


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

    availability = apex__data_cleanup_availability(data)
    df = ds_to_df(data.ffill('time'))
    ds = {}
    for var in data.data_vars:
        ds[var] = df[var][availability]
    ds['basic_availability'] = availability
    data = xr.Dataset(data_vars=ds)
    data['default_availability'] = apex__default_availability(data)
    data = data.sel(time=slice(start_date, end_date))
    data = compute_financial_indicators(data)
    return data

def apex__dataset(universe_name, subuniverse=None, group='all', postprocess=False, end_date=None, start_date='1990-01-01'):
    base_loc = f'/apex.data/apex.universes/{universe_name}/raw_data/'
    if subuniverse is not None:
        assert isinstance(subuniverse, str)
        base_loc = base_loc + f'/subuniverse_data/{subuniverse}'
    if postprocess:
        data = apex__read_dataset(base_loc, group='all')
        data = apex__postprocess_dataset(data)
    else:
        data = apex__read_dataset(base_loc, group=group)
    return data

def apex__create_subuniverse_dataset(universe_name, subuniverse_name, tickers):
    """
    Simply takes a subset of universe.
    """
    data = apex__dataset(universe_name)
    data = data.sel(ticker=tickers)
    data.coords['ticker'] = data.ticker.astype(str)
    save_dir = f'/apex.data/apex.universes/{universe_name}/subuniverse_data/{subuniverse_name}/input/zarr'
    # save
    apex__save_zarr_groups(data, save_dir)
    return True



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
    availability = dataset['basic_availability'].to_pandas()
    data = data.pct_change()
    data[data == 0.0] = np.nan
    results = {}
    for c in data.columns:
        results[c] = data[c].dropna().ewm(span=8).std()
    results = pd.concat(results, axis=1).reindex(availability.index).fillna(method='ffill')[availability]
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
    ds = ds.copy()
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
    vol40 = df_to_xr(returns.ewm(halflife=40).std()[ds['basic_availability'].to_pandas()])
    vol20 = df_to_xr(returns.ewm(halflife=20).std()[ds['basic_availability'].to_pandas()])
    vol10 = df_to_xr(returns.ewm(halflife=10).std()[ds['basic_availability'].to_pandas()])
    vol100 = df_to_xr(returns.ewm(halflife=100).std()[ds['basic_availability'].to_pandas()])

    ds['ewm_vol_40d_hl'] = vol40
    ds['ewm_vol_10d_hl'] = vol10
    ds['ewm_vol_20d_hl'] = vol20
    ds['ewm_vol_100d_hl'] = vol100
    return ds
