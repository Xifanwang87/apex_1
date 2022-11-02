from apex.toolz.bloomberg import apex__adjusted_market_data
import pandas as pd
import numpy as np

class DistancePairsTradingPortfolio:
    def __init__(self, base_security=None, universe=None):
        self.base_security = base_security
        self.market_data = apex__adjusted_market_data(self.base_security, *universe, parse=True)

        self.secondary_securities = self._compute_secondary_securities()
        self.signal = self._compute_signal()

    def _compute_secondary_securities(self):
        market_data = self.market_data
        base_security_returns = market_data['returns'][self.base_security].dropna()
        returns = market_data['returns'].reindex(base_security_returns.index)
        del returns[self.base_security]
        days_security = {}
        for day in base_security_returns.iloc[252:].index:
            available_securities = returns.loc[day].dropna().index.tolist()
            pairs = (returns[available_securities].fillna(0).cumsum().subtract(base_security_returns.cumsum(), axis=0)).loc[:day].iloc[-252:].pow(2).dropna(how='any')
            days_security[day] = pairs.sum().sort_values().index[0]
        return pd.Series(days_security)

    def _compute_signal(self):
        pairs = self.secondary_securities
        market_data = self.market_data
        base_security_returns = market_data['returns'][self.base_security].dropna()
        returns = market_data['returns'].reindex(base_security_returns.index).fillna(0)
        result = {}
        for day in pairs.index:
            current_pair = pairs.loc[day]
            pair_returns = (returns[self.base_security].loc[:day].iloc[-252:].cumsum() - returns[current_pair].loc[:day].iloc[-252:].cumsum())
            zscore = pair_returns.iloc[-1]/pair_returns.std()
            result[(day, current_pair)] = 1.0/zscore
        signal = pd.Series(result).reset_index().rename(columns={'level_0': 'date', 'level_1': 'ticker', 0: 'signal'})
        signal = signal.pivot_table(columns='ticker', values='signal', index='date')
        signal = signal.reindex(returns.index).fillna(0)
        signal = signal.divide(signal.abs().sum(axis=1), axis=0)
        signal[self.base_security] = -signal.sum(axis=1)
        return signal

    def signal_returns(self):
        signal = self.signal
        returns = self.market_data['returns']
        signal_rets = returns[signal.columns].subtract(returns[self.base_security], axis=0)
        return (signal.shift(1) * signal_rets).reindex(returns[self.base_security].dropna().index)

def compute_signal_for_security_in_universe(security, universe):
    result = DistancePairsTradingPortfolio(base_security=security, universe=universe)
    return result.signal