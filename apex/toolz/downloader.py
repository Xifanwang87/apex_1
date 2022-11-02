import datetime
import math
import pprint
import re
from pathlib import Path
from pprint import pprint
from typing import Container

import pandas as pd
from apex.toolz.dicttools import keys, values
from apex.toolz.itertools import flatten

import funcy
import toolz
from dataclasses import dataclass

from .bloomberg import get_index_member_weights_multiday as bbg_index_members
from .bloomberg import get_security_fundamental_data as bbg_fundamentals
from .bloomberg import get_security_historical_data as bbg_historical_data
from .bloomberg import get_security_market_data as bbg_market_data
from .bloomberg import get_security_metadata as bbg_metadata
from .bloomberg import get_security_metadata_full as bbg_metadata_full
from .dask import ApexDaskClient, compute_delayed
from .pandas import localize
from .tiingo import get_multi_ticker_market_data as tiingo_market_data
from .tiingo import get_multi_ticker_market_data_threaded as tiingo_market_data
from .tiingo import get_multi_ticker_metadata as tiingo_metadata
from .tiingo import get_multi_ticker_metadata_threaded as tiingo_metadata
from .storage import ApexMinio
from .universe import (
    ApexUniverseIndices,
    ApexUniverseSecurities,
    ApexCustomRoweIndicesUniverse,
    ApexUniverseBenchmarks,
    ApexUniverseAMUS,
    ApexUniverseAMNA
)
from .pandas import sort_stacked_data
from apex.security import ApexSecurity


def fix_raw_ticker_us(x):
    values_to_replace = (
        ' UW ',
        ' UN ',
        ' UQ ',
        ' UA ',
        ' UV ',
        ' UR ',
    )
    u = ' US '
    for v in values_to_replace:
        x = x.replace(v, u)
    return x


def fix_raw_ticker_cn(x):
    values_to_replace = (
        ' CT ',
    )
    u = ' CN '
    for v in values_to_replace:
        x = x.replace(v, u)
    return x


@dataclass(frozen=True)
class ApexDataDownloader:
    @property
    def universe_midstream(self):
        """
        Returns universe for every day. It has been pre-computed from 1996 and that's the list
        above.
        """
        universe = ApexUniverseSecurities()
        # what I want is tickers
        return set(universe.energy.infrastructure.midstream.tickers)

    @property
    def universe_energy_infrastructure(self):
        """
        Returns universe for every day. It has been pre-computed from 1996 and that's the list
        above.
        """
        universe = ApexUniverseSecurities()
        # what I want is tickers
        universe = set(universe.energy.infrastructure.tickers)
        return universe

    def universe_energy(self, north_america=True):
        """
        Returns universe for every day. It has been pre-computed from 1996 and that's the list
        above.
        """
        universe = ApexUniverseSecurities()
        # what I want is tickers
        res = set(universe.energy.tickers)
        res = filter(lambda x: ('US' in set(x.split(' '))) or ('CN' in set(x.split(' '))), res)
        return sorted(res)

    @property
    def universe_indices(self):
        """
        Returns important indices for lyf.

        In order of bloomberg popularity, so we'll be returning a tuple.
        """
        universe = ApexUniverseIndices().tickers
        return tuple(universe)

    @property
    def universe_rowe_custom(self):
        """
        Returns important indices for lyf.

        In order of bloomberg popularity, so we'll be returning a tuple.
        """
        universe = ApexCustomRoweIndicesUniverse()
        return universe.tickers

    def metadata(self, securities, source='bloomberg', full=True):
        if isinstance(securities, str):
            securities = [securities]
        if source == 'bloomberg':
            if full:
                data = bbg_metadata_full(*securities)
            else:
                data = bbg_metadata(*securities)
        elif source == 'tiingo':
            data = self.tiingo_metadata(securities)
        else:
            raise NotImplementedError("Source not found.")
        return data

    def tiingo_metadata(self, securities):
        tiingo_tickers = self.tiingo_tickers(securities)
        metadata = tiingo_metadata(*values(tiingo_tickers))
        metadata = toolz.keymap(lambda x: funcy.flip(tiingo_tickers)[x], metadata)
        return metadata

    def market_data(self, securities, source='bloomberg', sort_values=False):
        if isinstance(securities, str):
            securities = [securities]
        if source == 'bloomberg':
            data = bbg_market_data(*securities)
        elif source == 'tiingo':
            data = self.tiingo_market_data(securities)
        else:
            raise NotImplementedError("Source not found.")

        data['source'] = source
        data = data[['source', 'adjusted', 'date', 'identifier', 'field', 'value']]
        if sort_values:
            return sort_stacked_data(data)
        else:
            return data

    def tiingo_market_data(self, securities):
        tiingo_tickers = self.tiingo_tickers(securities)
        tiingo_pull_tickers = values(tiingo_tickers)
        data = tiingo_market_data(*tiingo_pull_tickers)
        data = toolz.valfilter(lambda x: x is not None, data)
        tiingo_to_bbg = funcy.flip(tiingo_tickers)
        data = toolz.keymap(lambda x: tiingo_to_bbg[x], data)
        for k, v in data.items():
            v['identifier'] = k
        result = pd.concat(values(data))
        result = result[['adjusted', 'date', 'identifier', 'field', 'value']]
        try:
            result['date'] = pd.DatetimeIndex(pd.to_datetime(result['date'])).tz_localize('America/Chicago', errors='coerce')
        except:
            result['date'] = pd.DatetimeIndex(pd.to_datetime(result['date'])).tz_convert('America/Chicago')

        return result

    def __distributed_fundamental_data(self, tickers: list):
        """
        Tickers is a map from raw ticker to fundamental ticker.
        """
        dask = ApexDaskClient()
        # Lets split it in 20 groups.
        group_size = max(math.ceil(len(tickers) / 20), 1)
        groups = list(toolz.partition_all(group_size, tickers))
        futures = []
        for group in groups:
            futures.append(dask.submit(bbg_fundamentals, *group))
        results = compute_delayed(futures, exceptions='ignore')
        return results

    def fundamental_data(self, securities, sort_values=True):
        if isinstance(securities, str):
            securities = [securities]

        results = bbg_fundamentals(*securities)
        results = results[['date', 'identifier', 'field', 'value']]

        if sort_values:
            results = results.sort_values(by=['date', 'identifier', 'field']).reset_index(drop=True)
        return {
            'results': results,
            'missing': missing,
            'security_to_fundamental_ticker_map': sec_to_ticker
        }

    def historical_data(self, securities, columns, sort_values=True):
        if isinstance(securities, str):
            securities = [securities]

        if isinstance(columns, str):
            securities = [securities]

        results = bbg_historical_data(*securities, fields=columns)
        return results

    def index_member_weights(self, index, start_date=None, end_date=None, freq='Q'):
        return bbg_index_members(index, start_date=start_date, end_date=end_date, freq=freq)

    def index_members(self, index, start_date=None, end_date=None, freq='Q'):
        result = self.index_member_weights(index, start_date=start_date, end_date=end_date, freq=freq)
        result = sorted(set(pd.DataFrame(result).T.columns))
        return result

    def tiingo_tickers(self, tickers):
        tiingo_inputs = {}
        metadata = self.metadata(tickers)
        for ticker, t_metadata in metadata.items():
            if t_metadata['id_exch_symbol'] == '':
                tiingo_inputs[ticker] = None
            else:
                tiingo_inputs[ticker] = t_metadata['id_exch_symbol']
        tiingo_inputs = toolz.valfilter(lambda x: x is not None, tiingo_inputs)
        return tiingo_inputs

    def get_fundamentals_ticker(self, securities):
        if isinstance(securities, str):
            securities = [securities]
        metadata = self.metadata(securities)
        available_securities_metadata = set(metadata.keys())
        missing_securities = set(securities).difference(available_securities_metadata)

        fund_tickers = {}
        for ticker in available_securities_metadata:
            sec_metadata = metadata[ticker]
            if sec_metadata['bpipe_reference_security_class'] != 'Equity':
                continue
            try:
                if sec_metadata['eqy_fund_ticker'] == '':
                    fund_tickers[ticker] = ticker
                else:
                    fundamental_ticker = sec_metadata.get('eqy_fund_ticker', 0) + ' ' + sec_metadata.get('market_sector_des', None)
                    fund_tickers[ticker] = fundamental_ticker
            except:
                missing_securities.add(ticker)
        return sorted(set(fund_tickers.values())), missing_securities, fund_tickers

    @property
    def equity_universe(self):
        """
        Returns universe for every day. It has been pre-computed from 1996 and that's the list
        above.

        It is only equities, however.
        """
        universe = set(self.universe_energy())
        universe = universe.union(ApexUniverseAMUS().tickers)
        universe = universe.union(ApexUniverseAMNA().tickers)
        universe = universe.union(self.universe_rowe_custom)
        universe = sorted(set(universe))
        universe = self.fix_ticker_changes(universe)
        return universe

    def fix_ticker_changes(self, ticker_list):
        universe = ticker_list.copy()
        metadata = self.metadata(universe, full=False)
        ticker_changes = {x for x in metadata if metadata[x]['market_status'] == 'TKCH'}
        for ticker in ticker_changes:
            metadata = bbg_metadata(ticker)[ticker]
            new_ticker = metadata['primary_security_ticker']
            universe.remove(ticker)
            universe.append(new_ticker)
        return universe

    def equity_data_update(self, tickers):
        """
        Very important function.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        securities = [ApexSecurity.from_id(x) for x in tickers]
        result = []
        for security in securities:
            id_for_bbg = security.id_bb_global
            id_for_tiingo = security.parsekyable_des
            column_order = ['update_datetime', 'source', 'adjusted', 'date', 'identifier', 'field', 'value']
            try:
                fundamental_data = self.fundamental_data(id_for_bbg)['results']
            except:
                fundamental_data = None

            try:
                tiingo_mkt_data = self.market_data(id_for_tiingo, source='tiingo')
            except:
                tiingo_mkt_data = None

            try:
                bloomberg_mkt_data = self.market_data(id_for_bbg, source='bloomberg')
            except:
                bloomberg_mkt_data = None

            market_data = []

            if tiingo_mkt_data is not None:
                market_data.append(tiingo_mkt_data)

            if bloomberg_mkt_data is not None:
                market_data.append(bloomberg_mkt_data)

            if len(market_data) > 0:
                market_data = pd.concat(market_data)
                market_data = sort_stacked_data(market_data)
                market_data['update_datetime'] = pd.Timestamp.now(tz='America/Chicago')
                market_data = market_data[column_order]
                market_data['identifier'] = id_for_bbg
            else:
                market_data = None

            if fundamental_data is not None and len(fundamental_data) > 0:
                fundamental_data['source'] = 'bloomberg'
                fundamental_data['adjusted'] = False
                fundamental_data_copy = fundamental_data.copy()
                fundamental_data_copy['adjusted'] = True
                fundamental_data = pd.concat([fundamental_data, fundamental_data_copy])
                fundamental_data['date'] = pd.DatetimeIndex(fundamental_data['date'], tz='America/Chicago')
                fundamental_data['update_datetime'] = pd.Timestamp.now(tz='America/Chicago')
                fundamental_data = fundamental_data[column_order]
                fundamental_data['identifier'] = id_for_bbg
            else:
                fundamental_data = None

            dataset = [market_data, fundamental_data]
            dataset = [x for x in dataset if x is not None]
            if len(dataset) > 0:
                data = pd.concat(dataset).sort_values(by=['adjusted', 'date', 'field', 'source']).reset_index(drop=True)
            else:
                data = None
            result.append(data)
        result = [x for x in result if x is not None]
        if len(result) > 0:
            return pd.concat(result).reset_index(drop=True)
        else:
            return None
