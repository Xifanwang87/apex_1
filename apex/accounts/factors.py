import pandas as pd
from toolz import valmap

from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.pipelines.covariance import lw_cov, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__adjusted_market_data,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_bloomberg_field_score,
                                         get_market_data, rank_signal_from_bbg)
from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg,
                                  fix_security_name_index,
                                  get_index_member_weights_on_day)
from funcy import isnone
from apex.pipelines.volatility import default_volatility
import numpy as np


@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='v0.1', should_cache_fn=isnone)
def apex__account_bloomberg_fundamental_field(account, ds, field_name):
    w = get_account_weights_by_date(account, ds)
    tickers = sorted(w.index.tolist())
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    bbg = ApexBloomberg()
    result = bbg.fundamentals(tickers, field_name)
    result.columns = result.columns.droplevel(1)
    result = result.reindex(market_data.index).fillna(method='ffill')
    result = result[market_data['px_last'].columns]
    result = result[~market_data['px_last'].isnull()]
    return result[tickers]


@UNIVERSE_DATA_CACHING.cache_on_arguments()
def account_bloomberg_field_factor(account, ds, field_name):
    w = get_account_weights_by_date(account, ds)
    universe = sorted(w.index.tolist())
    bbg = ApexBloomberg()
    result = bbg.history(universe, field_name)
    result.columns = result.columns.droplevel(1)
    return result[universe]

def apex__account_bloomberg_field_score(account, ds, field_name, cutoff=0.6):
    data = account_bloomberg_field_factor(account, ds, field_name)
    return rank_signal_from_bbg(data, cutoff=cutoff)

def apex__account_bloomberg_fundamental_field_score(account, ds, field_name, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, field_name)
    return rank_signal_from_bbg(data, cutoff=cutoff)

def account_growth_factor(account, ds, cutoff=0.6):
    return apex__account_bloomberg_fundamental_field_score(account, ds, 'EBITDA_GROWTH')

def account_value_factor(account, ds, cutoff=0.6):
    return apex__account_bloomberg_fundamental_field_score(account, ds, 'EV_TO_T12M_EBITDA')

def account_size_factor(account, ds, cutoff=0.6):
    return apex__account_bloomberg_field_score(account, ds, 'CUR_MKT_CAP')

def account_credit_factor(account, ds, cutoff=0.6):
    return apex__account_bloomberg_field_score(account, ds, 'RSK_BB_IMPLIED_CDS_SPREAD')

def account_leverage_factor(account, ds, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, 'BS_LT_BORROW') / \
        apex__account_bloomberg_fundamental_field(account, ds, 'EBITDA')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def account_yield_factor(account, ds, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, 'CF_FREE_CASH_FLOW') / \
        apex__account_bloomberg_fundamental_field(account, ds, 'ENTERPRISE_VALUE')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def account_vol_factor(account, ds, cutoff=0.6):
    w = get_account_weights_by_date(account, ds)
    universe = sorted(w.index.tolist())
    market_data = apex__adjusted_market_data(*universe, parse=True)
    market_data['returns'] = market_data['returns'].fillna(0)
    market_data = market_data.fillna(method='ffill')
    volatility = default_volatility(market_data.iloc[-600:].dropna(how='all'))
    return rank_signal_from_bbg(volatility, cutoff=cutoff)

def account_momentum_factor(account, ds, cutoff=0.6):
    w = get_account_weights_by_date(account, ds)
    universe = sorted(w.index.tolist())
    market_data = apex__adjusted_market_data(*universe, parse=True)
    returns = market_data['returns'].fillna(0)
    momentum = returns.rolling(252).sum() - returns.rolling(21).sum()
    return rank_signal_from_bbg(momentum, cutoff=cutoff)

def account_earnings_quality_factor(account, ds, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, 'IS_COMP_EPS_ADJUSTED')
    data = data.pct_change()
    data[data == 0.0] = np.nan
    results = {}
    for c in data.columns:
        results[c] = data[c].dropna().expanding().std()
    results = pd.DataFrame(results).fillna(method='ffill')

    tickers = sorted(results.columns.tolist())
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    results = results.reindex(market_data.index)
    results = results.fillna(method='ffill')
    valid = ~market_data['px_last'].isnull()
    result = rank_signal_from_bbg(results[valid], cutoff=cutoff)
    return result

def account_earnings_yield_factor(account, ds, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, 'EARN_YLD_HIST')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def account_dividend_yield_factor(account, ds, cutoff=0.6):
    data = apex__account_bloomberg_fundamental_field(account, ds, 'AVERAGE_DIVIDEND_YIELD')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def account_profitability_factor(account, ds, cutoff=0.6):
    income_xo = apex__account_bloomberg_fundamental_field(account, ds, 'IS_INC_BEF_XO_ITEM')
    assets_to_normalize_by = apex__account_bloomberg_fundamental_field(account, ds, 'BS_TOT_ASSET')
    cfo = apex__account_bloomberg_fundamental_field(account, ds, 'CF_CASH_FROM_OPER')
    roa = income_xo/assets_to_normalize_by
    droa = roa.diff(252)
    cfo = cfo / assets_to_normalize_by
    f_accruals = cfo - roa
    result = rank_signal_from_bbg(f_accruals, cutoff=cutoff)
    result += rank_signal_from_bbg(cfo, cutoff=cutoff)
    result += rank_signal_from_bbg(roa, cutoff=cutoff)
    result += rank_signal_from_bbg(droa, cutoff=cutoff)
    return rank_signal_from_bbg(result, cutoff=cutoff)

def account_liquidity_factor(account, ds, cutoff=0.6):
    current_ratio = apex__account_bloomberg_fundamental_field(account, ds, 'CUR_RATIO')
    return rank_signal_from_bbg(current_ratio, cutoff=cutoff)

def account_operating_efficiency_factor(account, ds, cutoff=0.6):
    gross_margin = apex__account_bloomberg_fundamental_field(account, ds, 'GROSS_MARGIN')
    turnover_ratio = apex__account_bloomberg_fundamental_field(account, ds, 'ASSET_TURNOVER')
    result = rank_signal_from_bbg(turnover_ratio, cutoff=0) + rank_signal_from_bbg(gross_margin, cutoff=0)
    return rank_signal_from_bbg(result, cutoff=cutoff)


ACCOUNT_NEUTRAL_FACTORS = {
    'Growth': account_growth_factor,
    'Value': account_value_factor,
    'Leverage': account_leverage_factor,
    'CFO Yield': account_yield_factor,
    'Size': account_size_factor,
    'Credit': account_credit_factor,
    'Volatility': account_vol_factor,
    'Momentum': account_momentum_factor,
    'Earnings Quality': account_earnings_quality_factor,
    'Earnings Yield': account_earnings_yield_factor,
    'Dividend Yield': account_dividend_yield_factor,
    'Liquidity': account_liquidity_factor,
    'Operating Efficiency': account_operating_efficiency_factor,
    'Profitability': account_profitability_factor,
}

