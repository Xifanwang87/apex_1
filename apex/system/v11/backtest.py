import inflection
import numpy as np
import pyfolio as pf


def apex__compute_strategy_trades(market_data, portfolio):
    drifted = portfolio.shift(1) * (1 + market_data['returns'])
    return portfolio - drifted.divide(drifted.sum(axis=1), axis=0)

def apex__compute_turnover_with_trades(trades):
    buys = trades[trades > 0].fillna(0).abs().sum(axis=1)
    sales = trades[trades < 0].fillna(0).abs().sum(axis=1)
    return np.minimum(buys.abs(), sales.abs())

def apex__performance_stats(weights, portfolio_returns):
    weights = weights.copy()
    weights['cash'] = 0
    perf_stats = pf.timeseries.perf_stats(portfolio_returns.fillna(0).dropna(), positions=weights.fillna(0).abs().dropna())
    perf_stats.index = map(inflection.underscore, perf_stats.index.str.replace(' ', '_'))
    return perf_stats

def apex__compute_strategy_returns(market_data, portfolio, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns']
    portfolio = portfolio.shift(1) # Addl lag

    portfolio_drifted = (portfolio.shift(1) * (1 + returns))
    portfolio_drifted = portfolio_drifted.divide(portfolio_drifted.abs().sum(axis=1), axis=0)
    
    trades = portfolio - portfolio_drifted
    return (portfolio.shift(1) * returns - trades.abs() * transaction_costs * 0.0001).sum(axis=1)


def apex__backtest_portfolio_weights(market_data, portfolio, transaction_costs=15):
    """
    Simple backtest with transaction costs and slippage
    """
    strategy_returns = apex__compute_strategy_returns(market_data, portfolio, transaction_costs).reindex(portfolio.index)
    stats = apex__performance_stats(portfolio, strategy_returns)
    return {
        'stats': stats,
        'returns': strategy_returns,
    }
