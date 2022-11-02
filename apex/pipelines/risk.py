import numpy as np
import pandas as pd

import pytest
from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.accounts.factors import ACCOUNT_NEUTRAL_FACTORS
from apex.pipelines.covariance import lw_cov, oas_cov
from apex.toolz.bloomberg import (apex__adjusted_market_data,
                                  fix_security_name_index,
                                  get_index_member_weights_on_day)
from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.universe import ApexCustomRoweIndicesUniverse
from sklearn.linear_model import HuberRegressor


def compute_risk_contribution(weights, covariance):
    diag_wts = pd.DataFrame(
        np.diag(weights.values),
        index=weights.index,
        columns=weights.index
    )
    port_var = weights @ covariance @ weights
    total_risk = diag_wts @ (covariance @ weights)
    result = (total_risk / port_var).sort_values(ascending=False)
    result = result[result.abs() > 1e-10]
    result = result/sum(result)
    return result


def compute_account_total_risk_contrib(account, ds):
    w = get_account_weights_by_date(account, ds)
    returns = apex__adjusted_market_data(*w.index.tolist(), parse=True)['returns']
    cov = oas_cov(returns.iloc[-500:].fillna(0))
    return compute_risk_contribution(w, cov)

def compute_account_active_risk_contrib(account, benchmark, ds):
    w = get_account_weights_by_date(account, ds)
    bw = get_index_member_weights_on_day(benchmark, pd.to_datetime(ds))

    bw = fix_security_name_index(bw)
    w = fix_security_name_index(w)

    securities = sorted(set(w.index.tolist() + bw.index.tolist()))
    returns = apex__adjusted_market_data(*securities, parse=True)['returns'][securities]
    returns = apex__adjusted_market_data(*securities, parse=True)['returns'][securities]
    cov = oas_cov(returns.iloc[-500:].fillna(0))
    w = w.reindex(securities).fillna(0)
    bw = bw.reindex(securities).fillna(0)
    return compute_risk_contribution(w - bw, cov)



def rowe_factor_risks_vs_amna_for_account(account_name, date):
    ds = pd.to_datetime(date).strftime('%Y-%m-%d')
    rowe_universe = ApexCustomRoweIndicesUniverse()
    indices = rowe_universe.to_dict()['custom_indices']
    rowe_universe_dict = rowe_universe.to_dict()['custom_indices']
    index_to_name = {x: x.replace('_', ' ').capitalize() for x in indices.keys()}
    index_tickers = rowe_universe.tickers

    sma_weights = get_account_weights_by_date(account_name, ds)
    benchmark_weights = fix_security_name_index(get_index_member_weights_on_day('AMNA Index', ds))

    securities = sorted(set(sma_weights.index.tolist() + benchmark_weights.index.tolist() + index_tickers))
    returns = pd.concat(apex__adjusted_market_data(*securities).values(), axis=1).swaplevel(axis=1)['day_to_day_tot_return_gross_dvds'] / 100.0

    sma_weights = sma_weights.reindex(securities).fillna(0)
    benchmark_weights = benchmark_weights.reindex(securities).fillna(0)

    current_position_returns = returns.multiply(sma_weights).sum(axis=1)
    index_returns = {}
    for ix, index_name in index_to_name.items():
        ix_securities = rowe_universe_dict[ix]
        index_returns[index_name] = returns[ix_securities].mean(axis=1)
    index_returns = pd.DataFrame(index_returns)

    results = {}
    for index in index_returns:
        mutual_info = ApexMutualInformationAnalyzer(X=index_returns[index], y=current_position_returns)
        results[index] = mutual_info.linear_beta(constant=True)['Beta']
    results = pd.Series(results)
    return results

def macro_risk_factors_for_account(account_name, date):
    ds = pd.to_datetime(date).strftime('%Y-%m-%d')

    indices = {
        'S&P 500': ['SPX Index'],
        'AMEI': ['AMEI Index'],
        'AMZ': ['AMZ Index'],
        'AMNA': ['AMNA Index'],
        'Natural Gas': ['UNG US Equity'],
        'Crude Oil': ['USO US Equity']
    }
    index_tickers = ['AMNA Index',
        'AMEI Index',
        'AMZ Index',
        'USO US Equity',
        'UNG US Equity',
        'SPX Index',
    ]
    sma_weights = get_account_weights_by_date(account_name, ds)
    benchmark_weights = fix_security_name_index(get_index_member_weights_on_day('AMNA Index', ds))

    securities = sorted(set(sma_weights.index.tolist() + benchmark_weights.index.tolist() + index_tickers))
    returns = pd.concat(apex__adjusted_market_data(*securities).values(), axis=1).swaplevel(axis=1)['day_to_day_tot_return_gross_dvds'] / 100.0

    sma_weights = sma_weights.reindex(securities).fillna(0)
    benchmark_weights = benchmark_weights.reindex(securities).fillna(0)

    current_position_returns = returns.multiply(sma_weights).sum(axis=1)
    index_returns = {}
    for index_name, ix_securities in indices.items():
        index_returns[index_name] = returns[ix_securities].mean(axis=1).fillna(0)
    index_returns = pd.DataFrame(index_returns)

    results = {}
    for index in index_returns:
        mutual_info = ApexMutualInformationAnalyzer(X=index_returns[index], y=current_position_returns, analysis_window=252)
        results[index] = mutual_info.linear_beta(constant=True)['Beta']

    results = pd.Series(results)
    return results


def compute_account_factor_loadings(account, ds):
    weights = get_account_weights_by_date(account, ds)
    tickers = sorted(weights.index.tolist())
    market_data = apex__adjusted_market_data(*tickers, parse=True)
    returns = market_data['returns'][tickers]

    FACTOR_RETURNS = {}
    FACTOR_SCORES = {}
    ACCOUNT_FACTOR_LOADINGS = {}
    for factor_name, factor_fn in ACCOUNT_NEUTRAL_FACTORS.items():
        FACTOR_SCORES[factor_name] = factor_fn(account, ds)[tickers]
        FACTOR_RETURNS[factor_name] = FACTOR_SCORES[factor_name].shift(1) * returns
    FACTOR_RETURNS = pd.concat(FACTOR_RETURNS, axis=1).sum(axis=1, level=0).loc['2005':]
    factor_cov = lw_cov(FACTOR_RETURNS)

    # Now let's estimate the factor loadings per asset
    security_factor_loadings = {}
    portfolio_loadings = {}
    for ticker in tickers:
        ticker_returns = returns[ticker].fillna(0).loc['2005':]
        regressor = HuberRegressor().fit(FACTOR_RETURNS, ticker_returns)
        fit = pd.Series(regressor.coef_, FACTOR_RETURNS.columns)
        security_factor_loadings[ticker] = fit
        portfolio_loadings[ticker] = fit * weights[ticker]

    portfolio_loadings = pd.DataFrame(portfolio_loadings).sum(axis=1)
    portfolio_loadings = portfolio_loadings/portfolio_loadings.abs().sum()
    portfolio_factor_er = FACTOR_RETURNS.mean() * portfolio_loadings
    return {
        'factor_loadings': portfolio_loadings,
        'factor_er': portfolio_factor_er,
        'er_breakdown': portfolio_factor_er/np.sum(np.abs(portfolio_factor_er)),
    }

def compute_account_active_factor_risks(account, benchmark, ds):
    pass
