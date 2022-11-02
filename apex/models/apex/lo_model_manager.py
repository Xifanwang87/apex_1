import pickle
import typing
import uuid

import funcy
import numpy as np
import pandas as pd
import pyfolio as pf
import redis
from dataclasses import dataclass, field
from joblib import Parallel, delayed, parallel_backend
from toolz import curry

from apex.analytics import apex_metric__max_dd
from apex.pipelines.volatility import default_volatility
from apex.toolz.arctic import ApexArctic
from apex.toolz.bloomberg import ApexBloomberg
from apex.toolz.bloomberg import apex__adjusted_market_data as apex__amd
from apex.toolz.bloomberg import (get_security_fundamental_data,
                                  apex__adjusted_market_data,
                                  get_security_metadata)
from apex.toolz.caches import FUNDAMENTAL_DATA_CACHING
from apex.toolz.dask import ApexDaskClient


BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS = [
    'current_ev_to_t12m_ebitda',
    'current_px_to_free_cash_flow',
    'free_cash_flow_yield',
    'pe_ratio',
    'price_to_boe_reserves',
    'px_to_book_ratio',
    'px_to_cash_flow',
    'px_to_ebitda',
    'px_to_est_ebitda',
    'px_to_ffo_ratio',
    'px_to_free_cash_flow',
    'px_to_sales_ratio',
    'px_to_tang_bv_per_sh',
    'shareholder_yield',
    'short_int_ratio',
]

#@FUNDAMENTAL_DATA_CACHING.cache_multi_on_arguments(namespace='fundamental_data:model_manager:daily', asdict=True)
def apex__base_daily_fundamental_data(*identifiers):
    def get_data_fn(identifier):
        bbg = ApexBloomberg()
        return bbg.history(identifier, BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS)

    with parallel_backend('threading', n_jobs=40):
        result = Parallel()(delayed(get_data_fn)(i) for i in identifiers)

    result = funcy.zipdict(identifiers, result)
    return result

def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def crossectional_rank_decorator(fn):
    def new_fn(tickers):
        try:
            alpha = fn(tickers)
        except ValueError:
            return None
        market_data = apex__amd(*tickers, parse=True)['px_last']
        alpha_res = alpha.rank(axis=1)
        alpha_res = alpha_res.subtract(alpha_res.mean(axis=1), axis=0)
        alpha_res = alpha_res.divide(alpha_res.abs().sum(axis=1), axis=0)
        return alpha_res[~market_data.isnull()]
    return new_fn

def crossectional_rank(alpha):
    alpha_res = alpha.rank(axis=1)
    alpha_res = alpha_res.subtract(alpha_res.mean(axis=1), axis=0)
    alpha_res = alpha_res.divide(alpha_res.abs().sum(axis=1), axis=0)
    return alpha_res

def apex__universe_bloomberg_fundamental_field(tickers, field_name):
    from apex.toolz.caches import UNIVERSE_DATA_CACHING
    @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace=f'fundamental_data:model_manager:ffilled:{field_name}', asdict=True)
    def bbg_bloomberg_fundamental_field(*tickers):
        bbg = ApexBloomberg()
        market_data = apex__amd(*tickers, parse=True)['px_last']
        dates = sorted(market_data.index.tolist())
        result = bbg.fundamentals(tickers, field_name).reindex(dates).fillna(method='ffill')
        result.columns = result.columns.droplevel(1)
        result = result[~market_data.isnull()]
        return {x: result[x] for x in tickers}
    return pd.concat(bbg_bloomberg_fundamental_field(*tickers), axis=1)

def apex__universe_bloomberg_field(tickers, field_name):
    from apex.toolz.caches import UNIVERSE_DATA_CACHING
    @UNIVERSE_DATA_CACHING.cache_multi_on_arguments(namespace=f'field_data:model_manager:ffilled:{field_name}', asdict=True)
    def bbg_bloomberg_field(*tickers):
        bbg = ApexBloomberg()
        market_data = apex__amd(*tickers, parse=True)['px_last']
        dates = sorted(market_data.index.tolist())
        result = bbg.history(tickers, field_name).reindex(dates).fillna(method='ffill')
        result.columns = result.columns.droplevel(1)
        result = result[~market_data.isnull()]
        return {x: result[x] for x in tickers}
    return pd.concat(bbg_bloomberg_field(*tickers), axis=1)


@crossectional_rank_decorator
def universe_growth_factor(tickers):
    return apex__universe_bloomberg_fundamental_field(tickers, 'EBITDA_GROWTH')

@crossectional_rank_decorator
def universe_value_factor(tickers):
    return apex__universe_bloomberg_fundamental_field(tickers, 'EV_TO_T12M_EBITDA')

@crossectional_rank_decorator
def universe_size_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'CUR_MKT_CAP')

@crossectional_rank_decorator
def universe_credit_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'RSK_BB_IMPLIED_CDS_SPREAD')

@crossectional_rank_decorator
def universe_leverage_factor(tickers):
    data = apex__universe_bloomberg_fundamental_field(tickers, 'BS_LT_BORROW') / \
        apex__universe_bloomberg_fundamental_field(tickers, 'EBITDA')
    return data

@crossectional_rank_decorator
def universe_yield_factor(tickers):
    data = apex__universe_bloomberg_fundamental_field(tickers, 'CF_FREE_CASH_FLOW') / \
        apex__universe_bloomberg_fundamental_field(tickers, 'ENTERPRISE_VALUE')
    return data

@crossectional_rank_decorator
def universe_volatility_factor(tickers):
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    market_data['returns'] = market_data['returns'].fillna(0)
    market_data = market_data.fillna(method='ffill')
    volatility = default_volatility(market_data.iloc[-20:].dropna(how='all'))
    return volatility

@crossectional_rank_decorator
def universe_momentum_factor(tickers):
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    returns = market_data['returns'].fillna(0)
    momentum = returns.rolling(252).sum() - returns.rolling(21).sum()
    return momentum

@crossectional_rank_decorator
def universe_earnings_quality_factor(tickers):
    data = apex__universe_bloomberg_fundamental_field(tickers, 'IS_COMP_EPS_ADJUSTED')
    data = data.pct_change()
    data[data == 0.0] = np.nan
    results = {}
    for c in data.columns:
        results[c] = data[c].dropna().ewm(span=8).std()
    results = pd.concat(results, axis=1).fillna(method='ffill')
    return results

@crossectional_rank_decorator
def universe_earnings_yield_factor(tickers):
    data = apex__universe_bloomberg_fundamental_field(tickers, 'EARN_YLD_HIST')
    return data

@crossectional_rank_decorator
def universe_dividend_yield_factor(tickers):
    data = apex__universe_bloomberg_fundamental_field(tickers, 'AVERAGE_DIVIDEND_YIELD')
    return data

@crossectional_rank_decorator
def universe_profitability_factor(tickers):
    income_xo = apex__universe_bloomberg_fundamental_field(tickers, 'IS_INC_BEF_XO_ITEM')
    assets_to_normalize_by = apex__universe_bloomberg_fundamental_field(tickers, 'BS_TOT_ASSET')
    cfo = apex__universe_bloomberg_fundamental_field(tickers, 'CF_CASH_FROM_OPER')
    roa = income_xo/assets_to_normalize_by
    droa = roa.diff(252)
    cfo = cfo / assets_to_normalize_by
    f_accruals = cfo - roa
    result = crossectional_rank(f_accruals)
    result += crossectional_rank(cfo)
    result += crossectional_rank(roa)
    result += crossectional_rank(droa)
    return result

@crossectional_rank_decorator
def universe_liquidity_factor(tickers):
    current_ratio = apex__universe_bloomberg_fundamental_field(tickers, 'CUR_RATIO')
    return current_ratio

@crossectional_rank_decorator
def universe_operating_efficiency_factor(tickers):
    gross_margin = apex__universe_bloomberg_fundamental_field(tickers, 'GROSS_MARGIN').fillna(0)
    turnover_ratio = apex__universe_bloomberg_fundamental_field(tickers, 'ASSET_TURNOVER').fillna(0)
    result = crossectional_rank(turnover_ratio) + crossectional_rank(gross_margin)
    return result

@crossectional_rank_decorator
def universe_best_eps_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'best_eps')

@crossectional_rank_decorator
def universe_best_ev_to_best_ebitda_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'best_ev_to_best_ebitda')

@crossectional_rank_decorator
def universe_best_gross_margin_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'best_gross_margin')

@crossectional_rank_decorator
def universe_current_ev_to_t12m_ebitda_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'current_ev_to_t12m_ebitda')

@crossectional_rank_decorator
def universe_current_px_to_free_cash_flow_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'current_px_to_free_cash_flow')

@crossectional_rank_decorator
def universe_free_cash_flow_yield_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'free_cash_flow_yield')

@crossectional_rank_decorator
def universe_pe_ratio_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'pe_ratio')

@crossectional_rank_decorator
def universe_price_to_boe_reserves_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'price_to_boe_reserves')

@crossectional_rank_decorator
def universe_px_to_book_ratio_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_book_ratio')

@crossectional_rank_decorator
def universe_px_to_cash_flow_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_cash_flow')

@crossectional_rank_decorator
def universe_px_to_ebitda_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_ebitda')

@crossectional_rank_decorator
def universe_px_to_est_ebitda_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_est_ebitda')

@crossectional_rank_decorator
def universe_px_to_ffo_ratio_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_ffo_ratio')

@crossectional_rank_decorator
def universe_px_to_free_cash_flow_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_free_cash_flow')

@crossectional_rank_decorator
def universe_px_to_sales_ratio_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_sales_ratio')

@crossectional_rank_decorator
def universe_px_to_tang_bv_per_sh_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'px_to_tang_bv_per_sh')

@crossectional_rank_decorator
def universe_shareholder_yield_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'shareholder_yield')

@crossectional_rank_decorator
def universe_short_int_ratio_factor(tickers):
    return apex__universe_bloomberg_field(tickers, 'short_int_ratio')


FACTORS = {
    'growth': universe_growth_factor,
    'value': universe_value_factor,
    'size': universe_size_factor,
    'credit': universe_credit_factor,
    'leverage': universe_leverage_factor,
    'yield': universe_yield_factor,
    'volatility': universe_volatility_factor,
    'momentum': universe_momentum_factor,
    'earnings_quality': universe_earnings_quality_factor,
    'earnings_yield': universe_earnings_yield_factor,
    'dividend_yield': universe_dividend_yield_factor,
    'profitability': universe_profitability_factor,
    'liquidity': universe_liquidity_factor,
    'operating_efficiency': universe_operating_efficiency_factor,
    'current_ev_to_t12m_ebitda': universe_current_ev_to_t12m_ebitda_factor,
    'current_px_to_free_cash_flow': universe_current_px_to_free_cash_flow_factor,
    'free_cash_flow_yield': universe_free_cash_flow_yield_factor,
    'pe_ratio': universe_pe_ratio_factor,
    'price_to_boe_reserves': universe_price_to_boe_reserves_factor,
    'px_to_book_ratio': universe_px_to_book_ratio_factor,
    'px_to_cash_flow': universe_px_to_cash_flow_factor,
    'px_to_ebitda': universe_px_to_ebitda_factor,
    'px_to_est_ebitda': universe_px_to_est_ebitda_factor,
    'px_to_ffo_ratio': universe_px_to_ffo_ratio_factor,
    'px_to_free_cash_flow': universe_px_to_free_cash_flow_factor,
    'px_to_sales_ratio': universe_px_to_sales_ratio_factor,
    'px_to_tang_bv_per_sh': universe_px_to_tang_bv_per_sh_factor,
    'shareholder_yield': universe_shareholder_yield_factor,
    'short_int_ratio': universe_short_int_ratio_factor
}


def rank_max_abs_scaler(raw_data, axis=0):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        data = data.T
        maxval = data.expanding().max()
        minval = data.expanding().min()
        data = data + minval
        data = data/maxval
        result = data.T
    else:
        data = data.rank(axis=1) - 1
        maxval = data.max(axis=1)
        scale = 1/maxval
        result = data.multiply(scale, axis=0)

    return result

def alpha_normalization_lo(alpha_data, returns, availability, cutoff=0, pct_rank=True):
    alpha_data = rank_max_abs_scaler(alpha_data[availability])
    alpha_data[alpha_data < cutoff] = np.nan
    alpha_data = alpha_data.divide(alpha_data.sum(axis=1), axis=0)
    return alpha_data

def compute_returns_with_tcs(portfolio, returns, transaction_costs):
    portfolio = portfolio.shift(2)
    return (portfolio * returns - portfolio.diff().abs() * transaction_costs * 0.0001).sum(axis=1)

def diff_transform(series, smooth_window=20, diff_window=1):
    assert diff_window > 0
    return series.ewm(span=smooth_window).mean().diff(diff_window)

def short_term_vs_long_term_transform(series, short_term_window=20, long_term_window=252):
    return series.ewm(span=long_term_window).mean() - series.ewm(span=short_term_window).mean()


SMOOTHING_TRANSFORMS = {

    'identity': lambda x: x,
    'identity_lagged': lambda x: x.shift(1),

    'ewm2': lambda x: x.ewm(halflife=2).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm15': lambda x: x.ewm(halflife=15).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm40': lambda x: x.ewm(halflife=40).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
    'ewm200': lambda x: x.ewm(halflife=200).mean(),

}

TRANSFORMS = {
    'identity': lambda x: x,
    'identity_lagged': lambda x: x.shift(1),

    'ewm2': lambda x: x.ewm(halflife=2).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm15': lambda x: x.ewm(halflife=15).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm40': lambda x: x.ewm(halflife=40).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
    'ewm200': lambda x: x.ewm(halflife=200).mean(),

    # Now the diffs/short term momentums
    'diff_s20_d2': curry(diff_transform, smooth_window=20, diff_window=2),
    'diff_s20_d10': curry(diff_transform, smooth_window=20, diff_window=10),

    'diff_s100_d2': curry(diff_transform, smooth_window=100, diff_window=2),
    'diff_s100_d10': curry(diff_transform, smooth_window=100, diff_window=10),

    'diff_s50_d2': curry(diff_transform, smooth_window=50, diff_window=2),
    'diff_s50_d10': curry(diff_transform, smooth_window=50, diff_window=10),

    'diff_s200_d2': curry(diff_transform, smooth_window=200, diff_window=2),
    'diff_s200_d10': curry(diff_transform, smooth_window=200, diff_window=10),

    'short_vs_long_term': short_term_vs_long_term_transform
}

def factor_compute_and_expand(tickers, factor, library_id, availability):
    """
    Computes factors and expands them through transforms
    """
    raw_factor_data = FACTORS[factor](tickers)

    mkt_data = apex__amd(*tickers, parse=True)
    mkt_returns = mkt_data['returns']
    factors_returns = {}
    for transform_name, transform_fn in TRANSFORMS.items():
        factor_name = factor + '_' + transform_name
        factor_data = transform_fn(raw_factor_data)

        factor_lo_model = alpha_normalization_lo(factor_data, mkt_returns, availability, pct_rank=False)
        write_to_alpha_folder(library_id, factor_name + f'_pos', data=factor_lo_model)
        factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
        factors_returns[factor_name + f'_pos'] = factor_returns

        factor_lo_model = alpha_normalization_lo(-factor_data, mkt_returns, availability, pct_rank=False)
        factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
        write_to_alpha_folder(library_id, factor_name + f'_neg', data=factor_lo_model)
        factors_returns[factor_name + f'_neg'] = factor_returns

        for cutoff in [0, 0.15, 0.25, 0.5]:
            factor_lo_model = alpha_normalization_lo(factor_data, mkt_returns, availability, cutoff=cutoff)

            factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
            if factor_returns.mean() > 0:
                write_to_alpha_folder(library_id, factor_name + f'_pos_cutoff={cutoff}', data={'portfolio': factor_lo_model, 'returns': factor_returns})
                factors_returns[factor_name + f'_pos_cutoff={cutoff}'] = factor_returns

            factor_lo_model = alpha_normalization_lo(-factor_data, mkt_returns, availability, cutoff=cutoff)
            factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
            if factor_returns.mean() > 0:
                write_to_alpha_folder(library_id, factor_name + f'_neg_cutoff={cutoff}', data={'portfolio': factor_lo_model, 'returns': factor_returns})
                factors_returns[factor_name + f'_neg_cutoff={cutoff}'] = factor_returns

    # Now we transform after cut off
    for cutoff in [0.15, 0.25, 0.5]:
        for transform_name, transform_fn in TRANSFORMS.items():
            factor_name = factor + '_rev__' + transform_name
            factor_data = alpha_normalization_lo(raw_factor_data, mkt_returns, availability, cutoff=cutoff)
            factor_lo_model = transform_fn(factor_data)[availability]
            factor_lo_model = factor_data.divide(factor_lo_model.sum(axis=1), axis=0)

            factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
            if factor_returns.mean() > 0:
                write_to_alpha_folder(library_id, factor_name + f'_pos_cutoff={cutoff}', data={'portfolio': factor_lo_model, 'returns': factor_returns})
                factors_returns[factor_name + f'_pos_cutoff={cutoff}'] = factor_returns

            factor_data = alpha_normalization_lo(-raw_factor_data, mkt_returns, availability, cutoff=cutoff)
            factor_lo_model = transform_fn(factor_data)[availability]
            factor_lo_model = factor_data.divide(factor_lo_model.sum(axis=1), axis=0)

            factor_returns = compute_returns_with_tcs(factor_lo_model, mkt_returns, 10)
            if factor_returns.mean() > 0:
                write_to_alpha_folder(library_id, factor_name + f'_neg_cutoff={cutoff}', data={'portfolio': factor_lo_model, 'returns': factor_returns})
                factors_returns[factor_name + f'_neg_cutoff={cutoff}'] = factor_returns

    return pd.concat(factors_returns, axis=1)


def get_portfolio_returns(library_id, portfolio):
    from apex.toolz.arctic import ApexArctic
    arc = ApexArctic()
    store = arc.get_library(library_id)
    try:
        return store.read(portfolio).data['returns']
    except:
        return None

def create_low_turnover_port(portfolio, halflife=125):
    availability = ~portfolio.isnull()
    slow_port = portfolio.fillna(0).ewm(halflife=halflife).mean()[availability]
    slow_port = slow_port.divide(slow_port.sum(axis=1), axis=0)
    return slow_port

def compute_turnover_by_year(portfolio):
    turnover = portfolio.diff().abs().sum(axis=1)
    return turnover.groupby(turnover.index.year).sum()/2

@dataclass
class ApexLongOnlyModelManager:
    """
    The Model Manager will build all portfolios that historically maximizes returns against
    a universe.

    Right now

    """
    universe_name: str
    universe: list # list of tickers
    availability: typing.Any
    recalculate: bool = field(default=True)
    benchmark: str = field(default='AMZ Index')

    def get_factor_returns(self, library_id=None):
        if library_id is None:
            library_id = self.library_id
        store = self.get_store(library_id=library_id)
        pool = ApexDaskClient()

        availability_sc = pool.scatter(self.availability)
        factor_returns = [pool.submit(factor_compute_and_expand, self.universe, x, self.library_id, availability_sc) for x in FACTORS]
        factor_returns = pd.concat([x.result() for x in factor_returns], axis=1)

        return factor_returns


    def build(self, library_id=None):
        """
        Builds portfolios and caches them in Arctic for caching

        1. Compute factors and expand them in cluster
        2. Filter factors by IR (> 0)
        3.
        """
        if library_id is None:
            library_id = self.library_id
        store = self.get_store(library_id=library_id)
        factor_returns = self.get_factor_returns(library_id=library_id)

        mkt_data = apex__amd(*self.universe, parse=True)
        mkt_returns = mkt_data['returns']

        # Now let's compute best portfolio over last 15 years
        analysis_start_dt = str((pd.Timestamp.now() - pd.DateOffset(years=15)).year)
        factor_returns = factor_returns.loc[analysis_start_dt:]

        # IR filtering
        benchmark_returns = apex__amd(self.benchmark, parse=True)['returns'][self.benchmark].loc[analysis_start_dt:]

        factor_std = factor_returns.std()
        factors_to_use = factor_std[factor_std > 0].index.tolist()

        factor_alpha = (1+factor_returns[factors_to_use].fillna(0).subtract(benchmark_returns, axis=0)).prod().sort_values(ascending=False)
        factor_alpha = factor_alpha.replace([np.inf, -np.inf], np.nan).dropna()
        factor_alpha = factor_alpha[factor_alpha > 1]
        factor_alpha = factor_alpha[factor_alpha < 10000] # Bugs/data errors
        print(factor_alpha)

        # Candidate portfolio selection/initialization
        candidate_portfolios = factor_alpha.index.tolist()

        # Set up
        # Pick 5 portfolios equally spaced - skipping first.
        base_portfolios = [
            candidate_portfolios[0],
            candidate_portfolios[5],
            candidate_portfolios[10],
            candidate_portfolios[15],
            candidate_portfolios[20],
        ] + candidate_portfolios[:50]

        candidate_portfolio_name = base_portfolios[0]
        portfolio = store.read(candidate_portfolio_name).data['portfolio'].fillna(0).loc[analysis_start_dt:]
        portfolio_rets = compute_returns_with_tcs(portfolio, mkt_returns, 10).loc[analysis_start_dt:]
        portfolio_cumulative_returns = (1+portfolio_rets).prod()
        portfolios_used = [candidate_portfolio_name]
        for candidate_portfolio_name in base_portfolios[1:]:
            portfolio = (portfolio * len(portfolios_used) +\
                store.read(candidate_portfolio_name).data['portfolio']\
                     .fillna(0).loc[analysis_start_dt:])/(len(portfolios_used) + 1)
            portfolios_used.append(candidate_portfolio_name)


        portfolio = portfolio.divide(portfolio.sum(axis=1), axis=0)
        portfolio_rets = compute_returns_with_tcs(portfolio, mkt_returns, 10).loc[analysis_start_dt:]
        portfolio_cumulative_returns = (1 + portfolio_rets).prod()

        curr_port_sortino = portfolio_rets.mean()/portfolio_rets[portfolio_rets < 0].std()*np.sqrt(252)
        curr_port_sharpe = portfolio_rets.mean()/portfolio_rets.std()*np.sqrt(252)
        curr_port_max_dd = apex_metric__max_dd(portfolio_rets)
        curr_port_ir = (portfolio_rets - benchmark_returns)
        curr_port_ir = curr_port_ir.mean()/curr_port_ir.std()*np.sqrt(252)


        # Decay setup
        lmb = 0.005
        curr_decay_val = 1

        # Portfolio selection logic - needs only to improve cumulative returns @ 10bps/side.
        print(f'[BASE = {candidate_portfolio_name}] - Cumulative Returns:', portfolio_cumulative_returns, 'Sharpe', portfolio_rets.mean()/portfolio_rets.std()*np.sqrt(252), 'Sortino', curr_port_sortino, 'Max DD', curr_port_max_dd)
        for ix in range(1, 5):
            # Double pass
            for candidate_portfolio_name in candidate_portfolios:
                curr_decay_val *= (1-lmb)
                if curr_decay_val < 0.005: # about 1000 tests
                    break
                try:
                    test_portfolio = store.read(candidate_portfolio_name).data['portfolio'].fillna(0).loc[analysis_start_dt:]
                    new_port = (portfolio + test_portfolio * curr_decay_val)
                    new_port = new_port.divide(new_port.sum(axis=1), axis=0)

                    new_port_rets = compute_returns_with_tcs(new_port, mkt_returns, 10).loc[analysis_start_dt:]
                    new_port_cum_rets = (1+new_port_rets).prod()
                    new_port_sortino = new_port_rets.mean()/new_port_rets[new_port_rets < 0].std()*np.sqrt(252)
                    new_port_sharpe = new_port_rets.mean()/new_port_rets.std()*np.sqrt(252)
                    new_port_max_dd = apex_metric__max_dd(new_port_rets)
                    new_port_ir = (new_port_rets - benchmark_returns)
                    new_port_ir = new_port_ir.mean()/new_port_ir.std()*np.sqrt(252)

                    condition_one = new_port_cum_rets / portfolio_cumulative_returns > 1.01
                    condition_two = new_port_sortino / curr_port_sortino
                    condition_three = new_port_ir / curr_port_ir
                    condition_four = curr_port_max_dd / new_port_max_dd

                    if condition_one or (condition_two * condition_three * condition_four  > 1.01):
                        print(f'[With {candidate_portfolio_name}] - Cumulative Returns:', new_port_cum_rets, 'Sharpe', new_port_sharpe, 'Sortino', new_port_sortino, 'Max DD', new_port_max_dd)
                        portfolio = new_port
                        portfolio_cumulative_returns = new_port_cum_rets
                        curr_port_sortino = new_port_sortino
                        curr_port_max_dd = new_port_max_dd
                        curr_port_sharpe = new_port_sharpe
                        curr_port_ir = new_port_ir
                        portfolios_used.append(candidate_portfolio_name)

                except KeyboardInterrupt:
                    break
                except KeyError:
                    try:
                        test_portfolio = store.read(candidate_portfolio_name).data['portfolio'].fillna(0).loc[analysis_start_dt:]
                        new_port = (portfolio * len(portfolios_used) + test_portfolio)/(len(portfolios_used) + 1)
                        new_port_rets = compute_returns_with_tcs(new_port, mkt_returns, 10).loc[analysis_start_dt:]
                        new_port_cum_rets = (1+new_port_rets).prod()
                        new_port_sortino = new_port_rets.mean()/new_port_rets[new_port_rets < 0].std()*np.sqrt(252)
                        new_port_max_dd = apex_metric__max_dd(new_port_rets)
                        new_port_ir = (new_port_rets - benchmark_returns)
                        new_port_ir = new_port_ir.mean()/new_port_ir.std()*np.sqrt(252)


                        condition_one = new_port_cum_rets / portfolio_cumulative_returns > 1.01
                        condition_two = new_port_sortino / curr_port_sortino
                        condition_three = new_port_ir / curr_port_ir
                        condition_four = curr_port_max_dd / new_port_max_dd

                        if condition_one or (condition_two * condition_three * condition_four > 1.01):
                            print(f'[With {candidate_portfolio_name}] - Cumulative Returns:', new_port_cum_rets, 'Sharpe', new_port_rets.mean()/new_port_rets.std()*np.sqrt(252), 'Sortino', new_port_sortino, 'Max DD', new_port_max_dd)
                            portfolio = new_port
                            portfolio_cumulative_returns = new_port_cum_rets
                            curr_port_sortino = new_port_sortino
                            curr_port_max_dd = new_port_max_dd
                            curr_port_ir = new_port_ir
                            portfolios_used.append(candidate_portfolio_name)
                    except:
                        continue
            curr_decay_val = 0.5**ix

        base_portfolio = portfolio.copy()

        base_portfolio_returns = {}
        portfolios = {}
        for transform_name, transform_fn in SMOOTHING_TRANSFORMS.items():
            portfolio_transformed = transform_fn(base_portfolio)[self.availability]
            portfolio_transformed = portfolio_transformed.divide(portfolio_transformed.sum(axis=1), axis=0)
            portfolios[transform_name] = portfolio_transformed
            portfolio_transformed_returns = compute_returns_with_tcs(portfolio_transformed, mkt_returns, 10).loc[analysis_start_dt:]
            base_portfolio_returns[transform_name] = portfolio_transformed_returns

        base_portfolio_returns = (1+pd.concat(base_portfolio_returns, axis=1)).prod().sort_values(ascending=False)

        final_portfolio = portfolios[base_portfolio_returns.index[0]]

        return {
            'portfolios': {
                'final': final_portfolio,
                'base': base_portfolio,
                'low_turnover__hl=100': create_low_turnover_port(final_portfolio, halflife=100),
                'low_turnover__hl=125': create_low_turnover_port(final_portfolio, halflife=125),
                'low_turnover__hl=150': create_low_turnover_port(final_portfolio, halflife=150),
                'low_turnover__hl=200': create_low_turnover_port(final_portfolio, halflife=200),
            },
            'selected_portfolios': portfolios_used,
            'library_id': library_id,
        }


def compute_model_result(model_result, benchmark='AMZ Index', plot=True, portfolio_kind='final', transaction_costs=10, start_year=2005):
    portfolio = model_result['portfolios'][portfolio_kind]
    market_data = apex__amd(*portfolio.columns.tolist(), parse=True)
    portfolio_returns = compute_returns_with_tcs(portfolio, market_data['returns'], transaction_costs).loc[str(start_year):].fillna(0)
    if plot:
        try:
            import pyfolio as pf
            pf.create_returns_tear_sheet(portfolio_returns.fillna(0).loc[str(start_year):].fillna(0), benchmark_rets=apex__amd(benchmark, parse=True)['returns'][benchmark].loc[str(start_year):].fillna(0))
        except:
            pass
    return portfolio_returns