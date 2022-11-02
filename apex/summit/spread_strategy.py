import quandl
import pandas as pd
import numpy as np
import joblib as jl
import asyncio

from dataclasses import dataclass, field
from apex.toolz.deco import lazyproperty


quandl.ApiConfig.api_key = 'DRRNen1z-D21zsRzunvQ'

loop = asyncio.get_event_loop()

async def get_quandl_data(ticker):
    try:
        data = quandl.get(ticker)
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
        fn = jl.delayed(get_quandl_data)
        result = jl.Parallel(n_jobs=32, prefer="threads")(fn(f'SRF/{x}') for x in possible_contracts)
        data = dict(zip(possible_contracts, result))
        data = pd.concat(data, axis=1)
        return data.swaplevel(0, axis=1).sort_index(axis=1)

    @lazyproperty
    def diffs(self):
        return self.data['close'].diff()

    def contract_by_month(self, month=None, contract_code=None):
        if month is not None:
            contract_codes = dict(zip(range(1, 13), self.contract_codes))
            contract_code = contract_codes[month]

