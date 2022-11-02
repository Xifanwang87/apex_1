import pyfolio as pf
import numpy as np


def alpha_performance_stats(weights, portfolio_returns):
    return {
        'performance_stats': pf.timeseries.perf_stats(portfolio_returns.dropna(), positions=weights.abs().dropna()),
        'drawdown_stats': pf.timeseries.gen_drawdown_table(portfolio_returns),
        'distribution': pf.timeseries.calc_distribution_stats(portfolio_returns)
    }

def get_sign_adjusted_results(weights, security_returns):
    security_attr = (weights.shift(1) * security_returns)
    returns = security_attr.sum(axis=1).dropna()
    sign = np.sign(np.mean(returns))
    if sign < 0:
        weights = -weights
        security_attr = (weights.shift(1) * security_returns)
        returns = security_attr.sum(axis=1).dropna()
        assert returns.mean() > 0

    return {
        'factor_sign': sign,
        'weights': weights,
        'returns': returns,
        'attribution': security_attr,
        'current_weights': weights.iloc[-1].dropna(),
    }
