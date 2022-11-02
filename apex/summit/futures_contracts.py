import asyncio

import joblib as jl
import numpy as np
import pandas as pd
import quandl
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from apex.toolz.deco import lazyproperty
import trio

quandl.ApiConfig.api_key = 'DRRNen1z-D21zsRzunvQ'


FUTURES_CONTRACTS = [['EXCHANGE', 'SYMBOL', 'NAME'],
    ['CME', 'TY', 'CBOT 10-year US Treasury Note'],
    ['CME', 'FV', 'CBOT 5-year US Treasury Note'],
    ['CME', 'TU', 'CBOT 2-year US Treasury Note'],
    ['CME', 'FF', 'CBOT 30-day Federal Funds'],
    ['CME', 'US', 'CBOT 30-year US Treasury Bond'],
    ['CME', 'C', 'CBOT Corn'],
    ['CME', 'DJ', 'CBOT Dow Jones Ind Avg (DJIA)'],
    ['CME', 'SM', 'CBOT Soybean Meal'],
    ['CME', 'BO', 'CBOT Soybean Oil'],
    ['CME', 'S', 'CBOT Soybeans'],
    ['CME', 'W', 'CBOT Wheat'],
    ['CME', 'AD', 'CME Australian Dollar AUD'],
    ['CME', 'BP', 'CME British Pound GBP'],
    ['CME', 'CD', 'CME Canadian Dollar CAD'],
    ['CME', 'EC', 'CME Euro FX'],
    ['CME', 'ED', 'CME Eurodollar'],
    ['CME', 'JY', 'CME Japanese Yen JPY'],
    ['CME', 'KW', 'CME Kansas City Wheat'],
    ['CME', 'LN', 'CME Lean Hogs'],
    ['CME', 'LC', 'CME Live Cattle'],
    ['CME', 'NQ', 'CME NASDAQ 100 Index Mini'],
    ['CME', 'NE', 'CME New Zealand Dollar NZD'],
    ['CME', 'NK', 'CME Nikkei 225'],
    ['CME', 'MD', 'CME S&P 400 Midcap Index'],
    ['CME', 'SP', 'CME S&P 500 Index'],
    ['CME', 'ES', 'CME S&P 500 Index E-Mini'],
    ['CME', 'SF', 'CME Swiss Franc CHF'],
    ['CME', 'RB', 'NYMEX Gasoline'],
    ['CME', 'GC', 'NYMEX Gold'],
    ['CME', 'HO', 'NYMEX Heating Oil'],
    ['CME', 'NG', 'NYMEX Natural Gas'],
    ['CME', 'PA', 'NYMEX Palladium'],
    ['CME', 'PL', 'NYMEX Platinum'],
    ['CME', 'SI', 'NYMEX Silver'],
    ['CME', 'CL', 'NYMEX WTI Crude Oil'],
    ['ICE', 'B', 'ICE Brent Crude Oil'],
    ['ICE', 'MP', 'ICE British Pound GBP'],
    ['ICE', 'CC', 'ICE Cocoa'],
    ['ICE', 'KC', 'ICE Coffee C'],
    ['ICE', 'CT', 'ICE Cotton'],
    ['ICE', 'G', 'ICE Gasoil'],
    ['ICE', 'O', 'ICE Heating Oil'],
    ['ICE', 'OJ', 'ICE Orange Juice'],
    ['ICE', 'ATW', 'ICE Rotterdam Coal'],
    ['ICE', 'RF', 'ICE Russell 1000 Index Mini'],
    ['ICE', 'TF', 'ICE Russell 2000 Index Mini'],
    ['ICE', 'SB', 'ICE Sugar No. 11'],
    ['ICE', 'M', 'ICE UK Natural Gas'],
    ['ICE', 'DX', 'ICE US Dollar Index'],
    ['ICE', 'T', 'ICE WTI Crude Oil']]
FUTURES_CONTRACTS = pd.DataFrame(FUTURES_CONTRACTS)
FUTURES_CONTRACTS.columns = FUTURES_CONTRACTS.iloc[0]
FUTURES_CONTRACTS = FUTURES_CONTRACTS.iloc[1:].reset_index(drop=True)


async def get_quandl_data(ticker):
    loop = asyncio.get_event_loop()
    try:
        e = ThreadPoolExecutor()
        data = await loop.run_in_executor(quandl.get, ticker)
    except:
        return None
    data = data.reset_index()
    data['eventtime'] = pd.to_datetime(data['Date'])
    data['contract'] = ticker
    if 'Prev. Day Open Interest' in data.columns:
        data['open_interest'] = data['Prev. Day Open Interest'].fillna(0.0).shift(-1)
    elif 'Previous Day Open Interest' in data.columns:
        data['open_interest'] = data['Previous Day Open Interest'].fillna(0.0).shift(-1)

    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Settle': 'settle',
        'Volume': 'volume',
    })

    for col in ['open', 'high', 'low', 'settle', 'volume', 'open_interest']:
        if col not in data.columns:
            data[col] = np.nan

    data.replace(to_replace='None', value=np.nan, inplace=True)
    data.fillna(value=np.nan, inplace=True)
    data['close'] = data['settle'].copy()
    data = data[['open', 'high', 'low', 'close', 'settle', 'volume', 'open_interest', 'eventtime', 'contract']].copy()
    return data.set_index('eventtime')


async def get_quandl_data_multiple(tickers):
    async with trio.open_nursery() as nursery:
        for ticker in tickers:
            nursery.start_soon(get_quandl_data, ticker)
    result = {x: get_quandl_data(x) for x in tickers}
    return result

@dataclass
class FutureInstrument:
    exchange: str
    symbol: str
    contract_codes: str = field(default='FGHJKMNQUVXZ')

    @lazyproperty
    def _contracts(self):
        start_year = 1966
        symbols = []
        for contract_code in self.contract_codes:
            symbols += [f'{self.exchange}_{self.symbol}{contract_code}{year}' for year in range(start_year, pd.Timestamp.now().year)]
        return sorted(set(symbols))

    @lazyproperty
    def contracts(self):
        data = self.data
        return data['close'].columns.tolist()

    @lazyproperty
    def data(self):
        possible_contracts = self._contracts
        result = trio.run(get_quandl_data_multiple, [f'SRF/{x}' for x in possible_contracts])
        print(result)
        data = pd.concat(result, axis=1)
        return data.swaplevel(0, axis=1).sort_index(axis=1)

    @lazyproperty
    def diffs(self):
        return self.data['close'].diff()

    def contract_by_month(self, month=None, contract_code=None):
        if month is not None:
            contract_codes = dict(zip(range(1, 13), self.contract_codes))
            contract_code = contract_codes[month]
