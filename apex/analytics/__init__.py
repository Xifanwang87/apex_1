import math
# cimport numpy
import numpy.random as nrand
import pandas as pd
import numpy as np
from pandas import DataFrame


def apex_metric__vol(returns):
    # Return the standard deviation of returns
    return np.std(returns)

def apex_metric__beta(returns, market):
    # Create a matrix of [returns, market]
    m = returns.copy()
    m['market'] = market
    # Return the covariance of m divided by the standard deviation of the market returns
    return m.corr()['market'] * m.std() / (market.std())

def apex_metric__corr(returns, market):
    # Create a matrix of [returns, market]
    m = returns.copy()
    m['market'] = market
    # Return the covariance of m divided by the standard deviation of the market returns
    return m.corr()['market']


def apex_metric__conditionalcorr(returns, market, sigma_level):
    # Create a matrix of [returns, market]
    m = returns.copy()
    m['market'] = market
    for col in m.columns:
        m[col] = (m[col] - m[col].mean()) / m[col].std(ddof=0)
    m = m[m['market'] < sigma_level]
    # Now let's filter the returns by -2sigm
    # Return the covariance of m divided by the standard deviation of the market returns
    return m.corr()['market']

def apex_metric__lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = returns.copy()
    threshold_array.ix[:] = threshold
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(lower=0)
    # Return the sum of the difference to the power of order
    return (diff ** order).sum() / len(returns)


def apex_metric__hpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = returns.copy()
    threshold_array.ix[:] = threshold
    # Calculate the difference between the threshold and the returns
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(lower=0)
    # Return the sum of the different to the power of order
    return (diff ** order).sum() / len(returns)


def apex_metric__var(returns, alpha):
    res = {}
    if isinstance(returns, pd.Series):
        sorted_returns = returns.sort_values()
        index = int(alpha * len(sorted_returns))
        return abs(sorted_returns.ix[index])

    for col in returns.columns:
        sorted_returns = sorted(returns[col])
        res[col] = sorted_returns
    res = pd.DataFrame(res)
    # Calculate the index associated with alpha
    index = int(alpha * len(res))
    # VaR should be positive
    return abs(res.ix[index])


def apex_metric__cvar(returns, alpha):
    # This method calculates the condition VaR of the returns

    if isinstance(returns, pd.Series):
        sorted_returns = returns.sort_values()
        index = int(alpha * len(sorted_returns))
        sum_var = sorted_returns.ix[:index].sum()
        return abs(sum_var) / index

    res = {}
    for col in returns.columns:
        sorted_returns = sorted(returns[col])
        res[col] = sorted_returns
    res = pd.DataFrame(res)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = res.ix[:index].sum()
    # Return the average VaR
    # CVaR should be positive
    return sum_var.abs() / index


def apex_metric__prices(returns, base):
    # Converts returns into prices
    return (returns + 1).cumprod()*base


def apex_metric__dd(returns, tau):
    # Returns the draw-down given time period tau
    values = apex_metric__prices(returns.ix[-tau:], 100)
    expanding_max = values.expanding(min_periods=1).max()
    drawdown_to_here = expanding_max/values - 1.0
    return drawdown_to_here.ix[-1]

def apex_metric__max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    values = (1.0 + returns).cumprod()
    expanding_max = values.expanding(min_periods=1).max()
    drawdown_to_here = values/expanding_max - 1.0
    if isinstance(returns, pd.Series):
        return abs(drawdown_to_here.min())
    return drawdown_to_here.min().abs()


def apex_metric__average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = apex_metric__dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = pd.DataFrame(drawdowns).reset_index(drop=True).ix[:periods].mean()
    return drawdowns

def apex_metric__average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = apex_metric__dd(returns, i) ** 2.0
        drawdowns.append(drawdown_i)
    drawdowns = pd.DataFrame(drawdowns).reset_index(drop=True).ix[:periods].mean()
    return drawdowns


def apex_metric__treynor_ratio(er, returns, market, rf):
    return (er - rf) / apex_metric__beta(returns, market)


def apex_metric__sharpe_ratio(er, returns, rf):
    return (er - rf) / apex_metric__vol(returns)


def apex_metric__information_ratio(returns, benchmark):
    return (returns.mean() - benchmark.mean())/ returns.std()


def apex_metric__modigliani_ratio(er, returns, benchmark, rf):
    np_rf = np.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (apex_metric__vol(rdiff) / apex_metric__vol(bdiff)) + rf


def apex_metric__excess_var(er, returns, rf, alpha):
    return (er - rf) / apex_metric__var(returns, alpha)


def apex_metric__conditional_sharpe_ratio(er, returns, rf, alpha):
    return (er - rf) / apex_metric__cvar(returns, alpha)


def apex_metric__omega_ratio(er, returns, rf, target=0):
    return (er - rf) / apex_metric__lpm(returns, target, 1)


def apex_metric__sortino_ratio(er, returns, rf, target=0):
    return (er - rf) / apex_metric__lpm(returns, target, 2)


def apex_metric__kappa_three_ratio(er, returns, rf, target=0):
    return (er - rf) / np.power(apex_metric__lpm(returns, target, 3), 1/3)


def apex_metric__gain_loss_ratio(returns, target=0):
    return apex_metric__hpm(returns, target, 1) / apex_metric__lpm(returns, target, 1)


def apex_metric__upside_potential_ratio(returns, target=0):
    return apex_metric__hpm(returns, target, 1) / np.sqrt(apex_metric__lpm(returns, target, 2))



def apex_metric__sterling_ration(er, returns, rf, periods):
    return (er - rf) / apex_metric__average_dd(returns, periods)


def apex_metric__calmar_ratio(er, returns, rf):
    return (er - rf) / apex_metric__max_dd(returns)

def apex_metric__burke_ratio(er, returns, rf, periods):
    return (er - rf) / np.sqrt(apex_metric__average_dd_squared(returns, periods))



class ApexPerformanceAnalyzer(object):
    """
    Calculates yearly performance based on columns of dataframe.

    Parameters
    ----------
    data    : pd.DataFrame with returns for whatever needs to be calculated. Indexed by date.
    """
    def __init__(self, portfolio_returns, indices_returns, benchmark_ticker, risk_free_rate, expected_returns=None):
        self.portfolio_returns = pd.DataFrame({'portfolio': portfolio_returns})
        self.indices_returns = indices_returns
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate
        self.expected_returns = portfolio_returns.mean()
        self.vol = apex_metric__vol(self.portfolio_returns)
        self.beta = apex_metric__beta(self.portfolio_returns, self.indices_returns[self.benchmark_ticker])
        self.var = apex_metric__var(self.portfolio_returns, 0.05)
        self.cvar = apex_metric__cvar(self.portfolio_returns, 0.05)
        self.max_dd = apex_metric__max_dd(self.portfolio_returns)
        self.treynor_ratio = apex_metric__treynor_ratio(self.expected_returns, self.portfolio_returns, self.indices_returns[self.benchmark_ticker], self.risk_free_rate)
        self.sharpe_ratio = apex_metric__sharpe_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate)
        self.information_ratio = apex_metric__information_ratio(self.portfolio_returns, self.indices_returns[self.benchmark_ticker])
        # Risk-adjusted return based on Value at Risk
        self.excess_var = apex_metric__excess_var(self.expected_returns, self.portfolio_returns, self.risk_free_rate, 0.05)
        self.conditional_sharpe_ratio = apex_metric__conditional_sharpe_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate, 0.05)
        # Risk-adjusted return based on Lower Partial Moments
        self.omega_ratio = apex_metric__omega_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate)
        self.sortino_ratio = apex_metric__sortino_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate)
        self.kappa_three_ratio = apex_metric__kappa_three_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate)
        self.gain_loss_ratio = apex_metric__gain_loss_ratio(self.portfolio_returns)
        self.upside_potential_ratio = apex_metric__upside_potential_ratio(self.portfolio_returns)

        # Risk-adjusted return based on Drawdown risk
        self.calmar_ratio = apex_metric__calmar_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate)
        self.sterling_ration = apex_metric__sterling_ration(self.expected_returns, self.portfolio_returns, self.risk_free_rate, len(self.portfolio_returns))
        self.burke_ratio = apex_metric__burke_ratio(self.expected_returns, self.portfolio_returns, self.risk_free_rate, len(self.portfolio_returns))