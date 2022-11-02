from itertools import chain
from numbers import Number
from typing import AnyStr, Mapping, Sequence

import pandas as pd
from dataclasses import dataclass
from dogpile.cache import make_region
from toolz import unique

from apex.toolz.bloomberg import ApexBloomberg, fix_security_name_index
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.security import ApexSecurity, get_security
from apex.toolz.deco import lazyproperty
from concurrent.futures import ThreadPoolExecutor
from apex.toolz.caches import PORTFOLIO_DB_CACHE, PORTFOLIO_DB_SHORT_TERM_CACHE
from toolz import partition_all
import typing
from apex.toolz.bloomberg import get_index_positions_on_day
import numpy as np


PORTFOLIO_ID_TO_ACCOUNT = {
    'U15262333-13': 'SMLPX',
    'U15262333-14': 'SMM',
    'U15262333-18': 'LP',
    'U15262333-20': 'TR TE',
    'U15262333-22': 'HEB',
    'U15262333-23': 'TR',
    'U15262333-34': 'BAYLOR',
    'U15262333-35': 'LPF AND KHF TRUST',
    'U15262333-37': 'MICHIGAN DIVERSIFIED INCOME FUND',
    'U15262333-38': 'STRATEGIC DIVERSIFIED INCOME FUND',
    # 'U15262333-40': 'COUNTY OF ALLEGHENY',
    # 'U15262333-41': 'UNIVERSITY OF SOUTHERN CALIFORNIA',
    'U15262333-42': 'SMA',
    'U15262333-43': 'PSERS',
    'U15262333-44': 'OHIO POLICE AND FIRE PENSION FUND',
    'U15262333-46': 'KAISER FDN HLTH PLANS HOSPITAL',
    'U15262333-47': 'POLICE AND FIRE RET CITY OF DETROIT',
    'U15262333-48': 'MEMORIAL HERMANN HEALTH SYSTEM',
    'U15262333-50': 'JPMC RETIREMENT PLAN',
    'U15262333-51': 'SHELBY COUNTY RETIREMENT SYSTEM',
    'U15262333-52': 'CITY OF EL PASO EMPLOYEES RETIREMENT TRUST',
    'U15262333-53': 'TEXAS AM FOUNDATION',
    'U15262333-54': 'MEMORIAL HERMANN HEALTH SYSTEM PENSION PLAN AND TR',
    'U15262333-55': 'SHELBY COUNTY',
    'U15262333-56': 'BCBS OF ARIZONA',
    'U15262333-58': 'GETTY FOUNDATION',
    'U15262333-59': 'HIGH CONVICTION',
    'U15262333-61': 'MN',
    # 'U15262333-64': 'HOUSTON MUNICIPAL EPS',
    # 'U15262333-65': 'MEADOWS FOUNDATION',
    'U15262333-330': 'STATE OF UTAH',
}

ACCOUNT_TO_PORTFOLIO_ID = {k: v for v, k in PORTFOLIO_ID_TO_ACCOUNT.items()}
ACCOUNTS = set(ACCOUNT_TO_PORTFOLIO_ID.keys())

PORTFOLIO_ID_TO_INCEPTION_DATE = {
    'U15262333-61': '07/26/2017',
    'U15262333-25': '10/18/2016',
    'U15262333-22': '10/12/2016',
    'U15262333-13': '09/20/2015',
    'U15262333-14': '12/31/2015',
    'U15262333-23': '09/30/2015',
    'U15262333-20': '10/12/2016',
    'U15262333-42': '12/31/2015',
    'U15262333-59': '12/31/2015',
    'U15262333-18': '10/12/2016',
}

@PORTFOLIO_DB_CACHE.cache_on_arguments()
def get_bloomberg_portfolio_data_for_date(portfolio_id, date):
    bbg = ApexBloomberg()
    date = pd.to_datetime(date).strftime('%Y%m%d')
    res = bbg.portfolio_data(portfolio_id + ' Client', date)
    res.columns = res.columns.str.lower()
    return res.set_index('security')['position']

def p_bloomberg_portfolio_data_for_dates(portfolio_id, dates):
    pool = ThreadPoolExecutor()
    dates = [x.strftime('%Y%m%d') for x in dates]
    pids = [portfolio_id + ' Client' for x in dates]
    holdings = dict(zip(dates, pool.map(get_bloomberg_portfolio_data_for_date, pids, dates)))
    holdings = pd.DataFrame(holdings).T
    holdings.index = pd.to_datetime(holdings.index, format='%Y%m%d')
    return holdings


def get_bloomberg_portfolio_data_by_id(portfolio_id,
                                 start_date=None,
                                 end_date=None,
                                 freq='B'):
    if start_date is None:
        start_date = pd.to_datetime(PORTFOLIO_ID_TO_INCEPTION_DATE.get(portfolio_id, '06/01/2019'), format='%m/%d/%Y').date()
    if end_date is None:
        end_date = pd.Timestamp.now().date()
    date_range = pd.date_range(start_date, end_date, freq=freq)
    # Lets make some batches (1Y)
    date_batches = partition_all(128, date_range)
    result = []
    pool = ThreadPoolExecutor()
    for batch in date_batches:
        result.append(pool.submit(p_bloomberg_portfolio_data_for_dates, portfolio_id, batch))
    result = pd.concat(compute_delayed(result))
    result.index = result.index.droplevel(0)
    return result

def get_account_holdings_by_date(acct, date):
    acct_bbg_id = ACCOUNT_TO_PORTFOLIO_ID[acct]
    res = fix_security_name_index(get_bloomberg_portfolio_data_for_date(
        acct_bbg_id, pd.Timestamp(date).strftime('%Y-%m-%d')))

    securities = set(res.index)
    if 'MSSALMLP Index' in securities:
        position_sign = np.sign(res['MSSALMLP Index'])
        index_positions = position_sign * get_index_positions_on_day('MSSALMLP Index', date)['Actual Weight']
        res = res.drop('MSSALMLP Index')
        res = pd.concat([res, index_positions])
        res = res.groupby(res.index).sum()
    return res

def get_account_weights_by_date(acct, date):
    holdings = get_account_holdings_by_date(acct, date)
    from apex.toolz.bloomberg import apex__adjusted_market_data
    closes = pd.concat(apex__adjusted_market_data(*holdings.index.tolist()).values(), axis=1).swaplevel(axis=1)['px_last']
    closes = closes.loc[:date].fillna(method='ffill', limit=3).iloc[-1]
    weights = (holdings * closes)
    weights = weights/weights.abs().sum()
    return weights

def get_account_historical_holdings(acct):
    acct_bbg_id = ACCOUNT_TO_PORTFOLIO_ID[acct]
    start_day = PORTFOLIO_ID_TO_INCEPTION_DATE.get(acct_bbg_id, '06/01/2019')
    res = get_bloomberg_portfolio_data_by_id(
        acct_bbg_id, start_day, pd.Timestamp.now())
    return res

def parsed_portfolio_data_by_id(portfolio_id, freq='B'):
    start_date = pd.to_datetime(PORTFOLIO_ID_TO_INCEPTION_DATE.get(portfolio_id, '06/01/2019'), format='%m/%d/%Y').date()
    end_date = pd.Timestamp.now().date()

    holdings = get_bloomberg_portfolio_data_by_id(portfolio_id, start_date, end_date, freq=freq)
    try:
        holdings = holdings.drop(columns=['USD Curncy'])  # No need...
    except:
        pass
    holdings.columns = [ApexSecurity.from_id(x) for x in holdings.columns]
    from apex.toolz.bloomberg import apex__adjusted_market_data
    market_data = apex__adjusted_market_data(*list(holdings.columns), parse=True)
    closes = market_data['px_last']
    weights = closes * holdings
    weights = weights.divide(weights.abs().sum(axis=1), axis=0)
    return weights

def parsed_account_data(account, freq='B'):
    portfolio_id = ACCOUNT_TO_PORTFOLIO_ID[account]
    return parsed_portfolio_data_by_id(portfolio_id)

@dataclass
class ApexSalientPortfolio:
    """
    Just a simple data class to get the data.
    """
    account: str
    portfolio_id: str

    @staticmethod
    def get_account_data(account, day):
        holdings = get_account_historical_holdings(account)
        holdings = pd.DataFrame(compute_delayed(holdings)).T
        holdings = holdings.drop(columns=['USD Curncy']) # No need...
        holdings.columns = [get_security(ticker=x) for x in holdings.columns]
        return holdings

    @classmethod
    def from_account(cls, account, date):
        data = ApexSalientPortfolio.get_account_data(account)
        return cls(account=account,
                   portfolio_id=ACCOUNT_TO_PORTFOLIO_ID[account],
                   inception_date=min(data))

    @classmethod
    def from_portfolio_id(cls, portfolio_id):
        return ApexSalientPortfolio.from_account(PORTFOLIO_ID_TO_ACCOUNT[portfolio_id])

    @property
    def weights(self):
        closes = pd.DataFrame(self.security_frame.close)
        weights = {}
        for day, portfolio in self.data.items():
            portfolio_pd = portfolio.to_pandas()
            closes_day = closes[portfolio_pd.index.tolist()].loc[day - pd.DateOffset(days=8):day].fillna(method='ffill').iloc[-1]
            day_value = portfolio_pd * closes_day
            day_weights = day_value / day_value.abs().sum()
            weights[day] = day_weights
        return pd.DataFrame(weights).T
