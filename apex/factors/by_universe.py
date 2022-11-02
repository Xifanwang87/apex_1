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
from apex.universe import APEX_UNIVERSES

@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='v0.1', should_cache_fn=isnone)
def apex__universe_bloomberg_fundamental_field(universe, field_name):
    tickers = APEX_UNIVERSES[universe]
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    bbg = ApexBloomberg()
    result = bbg.fundamentals(tickers, field_name)
    result.columns = result.columns.droplevel(1)
    result = result.reindex(market_data.index).fillna(method='ffill')
    result = result[market_data['px_last'].columns]
    result = result[~market_data['px_last'].isnull()]
    return result[tickers]


def apex__universe_market_data(universe):
    tickers = APEX_UNIVERSES[universe]
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    return market_data

@UNIVERSE_DATA_CACHING.cache_on_arguments()
def apex__universe_bloomberg_field_factor(universe, field_name):
    tickers = sorted(APEX_UNIVERSES[universe])
    bbg = ApexBloomberg()
    result = bbg.history(tickers, field_name)
    result.columns = result.columns.droplevel(1)
    return result[tickers]

def apex__account_bloomberg_field_score(universe, field_name, cutoff=0.6):
    data = apex__universe_bloomberg_field_factor(universe, field_name)
    return rank_signal_from_bbg(data, cutoff=cutoff)

def apex__universe_bloomberg_fundamental_field_score(universe, field_name, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, field_name)
    return rank_signal_from_bbg(data, cutoff=cutoff)

def universe_growth_factor(universe, cutoff=0.6):
    return apex__universe_bloomberg_fundamental_field_score(universe, 'EBITDA_GROWTH', cutoff=cutoff)

def universe_value_factor(universe, cutoff=0.6):
    return apex__universe_bloomberg_fundamental_field_score(universe, 'EV_TO_T12M_EBITDA', cutoff=cutoff)

def universe_size_factor(universe, cutoff=0.6):
    return apex__account_bloomberg_field_score(universe, 'CUR_MKT_CAP', cutoff=cutoff)

def universe_credit_factor(universe, cutoff=0.6):
    return apex__account_bloomberg_field_score(universe, 'RSK_BB_IMPLIED_CDS_SPREAD', cutoff=cutoff)

def universe_leverage_factor(universe, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, 'BS_LT_BORROW') / \
        apex__universe_bloomberg_fundamental_field(universe, 'EBITDA')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def universe_yield_factor(universe, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, 'CF_FREE_CASH_FLOW') / \
        apex__universe_bloomberg_fundamental_field(universe, 'ENTERPRISE_VALUE')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def universe_vol_factor(universe, cutoff=0.6):
    tickers = APEX_UNIVERSES[universe]
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    market_data['returns'] = market_data['returns'].fillna(0)
    market_data = market_data.fillna(method='ffill')
    volatility = default_volatility(market_data.iloc[-600:].dropna(how='all'))
    return rank_signal_from_bbg(volatility, cutoff=cutoff)

def universe_momentum_factor(universe, cutoff=0.6):
    tickers = APEX_UNIVERSES[universe]
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    returns = market_data['returns'].fillna(0)
    momentum = returns.rolling(252).sum() - returns.rolling(21).sum()
    return rank_signal_from_bbg(momentum, cutoff=cutoff)

def universe_earnings_quality_factor(universe, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, 'IS_COMP_EPS_ADJUSTED')
    data = data.pct_change()
    data[data == 0.0] = np.nan
    results = {}
    for c in data.columns:
        results[c] = data[c].dropna().expanding().std()
    results = pd.concat(results, axis=1).fillna(method='ffill')

    tickers = sorted(results.columns.tolist())
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    results = results.reindex(market_data.index)
    results = results.fillna(method='ffill')
    valid = ~market_data['px_last'].isnull()
    result = rank_signal_from_bbg(results[valid], cutoff=cutoff)
    return result

def universe_earnings_yield_factor(universe, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, 'EARN_YLD_HIST')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def universe_dividend_yield_factor(universe, cutoff=0.6):
    data = apex__universe_bloomberg_fundamental_field(universe, 'AVERAGE_DIVIDEND_YIELD')
    return rank_signal_from_bbg(data, cutoff=cutoff)

def universe_profitability_factor(universe, cutoff=0.6):
    income_xo = apex__universe_bloomberg_fundamental_field(universe, 'IS_INC_BEF_XO_ITEM')
    assets_to_normalize_by = apex__universe_bloomberg_fundamental_field(universe, 'BS_TOT_ASSET')
    cfo = apex__universe_bloomberg_fundamental_field(universe, 'CF_CASH_FROM_OPER')
    roa = income_xo/assets_to_normalize_by
    droa = roa.diff(252)
    cfo = cfo / assets_to_normalize_by
    f_accruals = cfo - roa
    result = rank_signal_from_bbg(f_accruals, cutoff=cutoff)
    result += rank_signal_from_bbg(cfo, cutoff=cutoff)
    result += rank_signal_from_bbg(roa, cutoff=cutoff)
    result += rank_signal_from_bbg(droa, cutoff=cutoff)
    return rank_signal_from_bbg(result, cutoff=cutoff)

def universe_liquidity_factor(universe, cutoff=0.6):
    current_ratio = apex__universe_bloomberg_fundamental_field(universe, 'CUR_RATIO')
    return rank_signal_from_bbg(current_ratio, cutoff=cutoff)

def universe_operating_efficiency_factor(universe, cutoff=0.6):
    gross_margin = apex__universe_bloomberg_fundamental_field(universe, 'GROSS_MARGIN')
    turnover_ratio = apex__universe_bloomberg_fundamental_field(universe, 'ASSET_TURNOVER')
    result = rank_signal_from_bbg(turnover_ratio, cutoff=0) + rank_signal_from_bbg(gross_margin, cutoff=0)
    return rank_signal_from_bbg(result, cutoff=cutoff)

UNIVERSE_NEUTRAL_FACTORS = {
    'Growth': universe_growth_factor,
    'Value': universe_value_factor,
    'Leverage': universe_leverage_factor,
    'CFO Yield': universe_yield_factor,
    'Size': universe_size_factor,
    'Credit': universe_credit_factor,
    'Volatility': universe_vol_factor,
    'Momentum': universe_momentum_factor,
    'Earnings Quality': universe_earnings_quality_factor,
    'Earnings Yield': universe_earnings_yield_factor,
    'Dividend Yield': universe_dividend_yield_factor,
    'Liquidity': universe_liquidity_factor,
    'Operating Efficiency': universe_operating_efficiency_factor,
    'Profitability': universe_profitability_factor,
}

