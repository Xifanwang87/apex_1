from .deco import retry
import requests
from tiingo import TiingoClient
from concurrent.futures import ThreadPoolExecutor
from .functools import dictmap
from .threading import threaded__dictmap
from functools import wraps
from .pandas import timeseries_df_stacked
import pandas as pd
from toolz import keymap
import funcy

__TIINGO_CONFIG = {}
__TIINGO_CONFIG['session'] = True
__TIINGO_CONFIG['api_key'] = '0b5cf1b19e96775ae780b349f69b4a33b1c44a24'

TIINGO_CLIENT = TiingoClient(__TIINGO_CONFIG)

def stack_and_localize_tiingo_df(data, tz='America/Chicago'):
    data = data.copy()
    data.index.rename('date', inplace=True)
    try:
        data = data.tz_localize(tz)
    except:
        data = data.tz_convert(tz)
    data = data.stack().reset_index(drop=False).rename(columns={'level_1': 'field', 0: 'value'}).dropna()
    return data

def process_tiingo_market_data_df(data, tz='America/Chicago'):
    adjusted_data = data[['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']].rename(
        columns={
            'adjClose': 'px_last',
            'adjHigh': 'px_high',
            'adjLow': 'px_low',
            'adjOpen': 'px_open',
            'adjVolume': 'px_volume',
        }
    )
    adjusted_data['returns'] = adjusted_data['px_last'].pct_change()
    adjusted_data = stack_and_localize_tiingo_df(adjusted_data)
    adjusted_data['adjusted'] = True
    unadjusted_data = data[['open', 'high', 'low', 'close', 'volume', 'adjClose']].rename(columns={
        'adjClose': 'returns',
        'open': 'px_open',
        'close': 'px_close',
        'low': 'px_low',
        'high': 'px_high',
        'volume': 'px_volume',
    })
    unadjusted_data['returns'] = unadjusted_data['returns'].pct_change()
    unadjusted_data = stack_and_localize_tiingo_df(unadjusted_data)
    unadjusted_data['adjusted'] = False
    data = pd.concat([unadjusted_data, adjusted_data]).sort_index().reset_index(drop=True).sort_values(by=['date', 'adjusted', 'field'])
    data = data[['date', 'adjusted', 'field', 'value']]
    return data

def get_ticker_news(tickers, start_date=None, end_date=None, sources=[], tags=[], limit=1000, offset=0):
    today = pd.Timestamp.now()
    if start_date is None:
        start_date ='1/1/2018'
    if end_date is None:
        end_date = today

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    return client.get_news(tickers=tickers,
                            tags=tags,
                            sources=sources,
                            startDate=start_date,
                            endDate=end_date,
                            limit=limit,
                            offset=offset,
                            sortBy='publishedDate')

@retry(Exception, tries=3)
def get_ticker_market_data(ticker):
    data = TIINGO_CLIENT.get_dataframe(ticker, startDate='1900-01-01')
    data = process_tiingo_market_data_df(data)
    data['identifier'] = ticker
    return data

@retry(Exception, tries=3)
def get_ticker_metadata(ticker):
    return TIINGO_CLIENT.get_ticker_metadata(ticker)

def get_multi_ticker_metadata(*tickers):
    return dictmap(get_ticker_metadata, tickers)

def get_multi_ticker_market_data(*tickers):
    result = dictmap(get_ticker_market_data, tickers)
    final_result = []
    for identifier, data in result.items():
        data['identifier'] = identifier
        final_result.append(data)
    return pd.concat(final_result).reset_index(drop=True)

def get_multi_ticker_metadata_threaded(*tickers):
    def pull_fn(ticker):
        client = TiingoClient(__TIINGO_CONFIG)
        try:
            return client.get_ticker_metadata(ticker)
        except:
            return None
    return threaded__dictmap(pull_fn, tickers, tickers)

def get_multi_ticker_market_data_threaded(*tickers):
    def pull_fn(ticker):
        client = TiingoClient(__TIINGO_CONFIG)
        try:
            data = client.get_dataframe(ticker, startDate='1900-01-01')
        except:
            data = False
        if data is not False:
            return process_tiingo_market_data_df(data)
        else:
            return None
    pool = ThreadPoolExecutor()

    result = funcy.zipdict(tickers, pool.map(pull_fn, tickers))
    return result