import contextlib
import inspect
import itertools
import logging
import multiprocessing
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy import diag, log, sqrt, trace
from numpy.linalg import inv
from statsmodels.api import OLS

import xarray as xr
from apex.toolz.dask import ApexDaskClient as Client
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def zion_xr__compute_strategy_returns__distributed(market_data: xr.Dataset, portfolios: xr.Dataset, transaction_costs, additional_lag=1, return_combined=True, margin_rate=0.03):
    """
    Last modifications:
    2019-11-30: Drifted portfolio having same leverage as original portfolio
    2019-12-02: Margin cost on levered portfolios
    """
    def target_fn(portfolio, returns):
        portfolio = portfolio.shift(time=additional_lag).fillna(0) # Addl lag
        portfolio_leverage = np.abs(portfolio).sum('ticker')
        portfolio_shifted = portfolio.shift(time=1)

        portfolio_drifted = (portfolio_shifted * (1 + returns))
        portfolio_drifted = portfolio_drifted / np.abs(portfolio_drifted).sum('ticker')
        portfolio_drifted = portfolio_drifted * portfolio_leverage.shift(time=1)

        trades = np.abs(portfolio_shifted - portfolio_drifted)

        universe = portfolio.where(np.abs(portfolio) > 0).isnull()
        universe = (~universe).astype(float)
        margin_cost_per_day = (portfolio_leverage - 1) * margin_rate / 252
        margin_costs_by_security = margin_cost_per_day / universe.sum('ticker')
        margin_costs_by_security = (universe * margin_costs_by_security).fillna(0)
        return (portfolio_shifted * returns - trades * transaction_costs * 0.0001 - margin_costs_by_security).sum('ticker')

    pool = Client()
    returns = market_data['returns'].fillna(0)
    returns_sc = pool.scatter(returns)

    strategy_returns = {}
    for portfolio_name in portfolios.data_vars:
        strategy_returns[portfolio_name] = pool.submit(target_fn, portfolios[portfolio_name], returns_sc)
    strategy_returns = pool.gather(strategy_returns)
    return xr.Dataset(strategy_returns)


def zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs, additional_lag=1, return_combined=True, margin_rate=0.03):
    """
    Last modifications:
    2019-11-30: Drifted portfolio having same leverage as original portfolio
    2019-12-02: Margin cost on levered portfolios
    """
    returns = market_data['returns'].fillna(0)
    strategy_returns = {}
    if isinstance(signal_dataset, xr.Dataset):
        if len(signal_dataset) < 20:
            for portfolio_name in signal_dataset:
                portfolio = signal_dataset[portfolio_name].shift(time=additional_lag).fillna(0) # Addl lag
                portfolio_leverage = np.abs(portfolio).sum('ticker')
                portfolio_shifted = portfolio.shift(time=1)

                portfolio_drifted = (portfolio_shifted * (1 + returns))
                portfolio_drifted = portfolio_drifted / np.abs(portfolio_drifted).sum('ticker')
                portfolio_drifted = portfolio_drifted * portfolio_leverage.shift(time=1)

                trades = np.abs(portfolio_shifted - portfolio_drifted)

                universe = portfolio.where(np.abs(portfolio) > 0).isnull()
                universe = (~universe).astype(float)
                margin_cost_per_day = (portfolio_leverage - 1) * margin_rate / 252
                margin_costs_by_security = margin_cost_per_day / universe.sum('ticker')
                margin_costs_by_security = (universe * margin_costs_by_security).fillna(0)

                if not return_combined:
                    strategy_returns[portfolio_name] = (portfolio_shifted * returns - trades * transaction_costs * 0.0001 - margin_costs_by_security)
                else:
                    strategy_returns[portfolio_name] = (portfolio_shifted * returns - trades * transaction_costs * 0.0001 - margin_costs_by_security).sum('ticker')
                del trades
                del portfolio_drifted
                del portfolio_shifted
                del portfolio
            strategy_returns = xr.Dataset(strategy_returns)
        else:
            strategy_returns = zion_xr__compute_strategy_returns__distributed(market_data, signal_dataset, transaction_costs,
                                                                              additional_lag=additional_lag, margin_rate=margin_rate)
    else:
        portfolio = signal_dataset.shift(time=additional_lag).fillna(0) # Addl lag
        portfolio_leverage = np.abs(portfolio).sum('ticker')
        portfolio_shifted = portfolio.shift(time=1)

        portfolio_drifted = (portfolio_shifted * (1 + returns))
        portfolio_drifted = portfolio_drifted / np.abs(portfolio_drifted).sum('ticker')
        portfolio_drifted = portfolio_drifted * portfolio_leverage.shift(time=1) # Because leverage doesn't change

        universe = portfolio.where(np.abs(portfolio) > 0).isnull()
        universe = (~universe).astype(float)
        margin_cost_per_day = (portfolio_leverage - 1) * margin_rate / 252
        margin_costs_by_security = margin_cost_per_day / universe.sum('ticker')
        margin_costs_by_security = (universe * margin_costs_by_security).fillna(0)
        trades = np.abs(portfolio_shifted - portfolio_drifted)
        if not return_combined:
            strategy_returns = (portfolio_shifted * returns - trades * transaction_costs * 0.0001 - margin_costs_by_security)
        else:
            strategy_returns = (portfolio_shifted * returns - trades * transaction_costs * 0.0001 - margin_costs_by_security).sum('ticker')
        del trades
        del portfolio_drifted
        del portfolio_shifted
        del portfolio
    #import gc
    #gc.collect()
    return strategy_returns


def zion__performance_stats(weights, portfolio_returns):
    weights = weights.copy()
    weights['cash'] = 0
    import pyfolio as pf
    import inflection
    perf_stats = pf.timeseries.perf_stats(portfolio_returns, positions=weights)
    perf_stats.index = map(inflection.underscore, perf_stats.index.str.replace(' ', '_'))
    del weights
    return perf_stats

def zion_xr__backtest_portfolio_weights(market_data, signal_dataset, transaction_costs=15, additional_lag=1):
    """
    Signal dataset is an xr Dataset with vars = portfolios
    """
    signal_returns = zion_xr__compute_strategy_returns(market_data, signal_dataset, transaction_costs, additional_lag=additional_lag)
    if isinstance(signal_dataset, xr.Dataset):
        signals = list(signal_dataset.data_vars.keys())
        stats = pd.DataFrame({x: zion__performance_stats(signal_dataset[x].to_pandas(), signal_returns[x].to_pandas()) for x in signals}).T
    elif isinstance(signal_dataset, xr.DataArray):
        stats = zion__performance_stats(signal_dataset.to_pandas(), signal_returns.to_pandas())
    # Now need to do this...
    return {
        'stats': stats,
        'returns': signal_returns,
    }

zion_xr__backtest = zion_xr__backtest_portfolio_weights

class PickleMixin(object):
    def save(self, filename):
        """ Save object as a pickle """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        """ Load pickled object. """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AlgoResult(PickleMixin):
    """ Results returned by algo's run method. The class containts useful
    metrics such as sharpe ratio, mean return, drawdowns, ... and also
    many visualizations.
    You can specify transactions by setting AlgoResult.fee. Fee is
    expressed in a percentages as a one-round fee.
    """

    def __init__(self, X, B):
        """
        :param X: Price relatives.
        :param B: Weights.
        """
        # set initial values
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X

        # update logarithms, fees, etc.
        self._recalculate()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        self._recalculate()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        self._recalculate()

    @property
    def fee(self):
        return self._fee

    @fee.setter
    def fee(self, value):
        """ Set transaction costs. Fees can be either float or Series
        of floats for individual assets with proper indices. """
        if isinstance(value, dict):
            value = pd.Series(value)
        if isinstance(value, pd.Series):
            missing = set(self.X.columns) - set(value.index)
            assert len(missing) == 0, 'Missing fees for {}'.format(missing)

        self._fee = value
        self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        r = (self.X - 1) * self.B
        self.asset_r = r + 1
        self.r = r.sum(axis=1) + 1

        # stock went bankrupt
        self.r[self.r < 0] = 0.

        # add fees
        if not isinstance(self._fee, float) or self._fee != 0:
            fees = (self.B.shift(-1).mul(self.r, axis=0) - self.B * self.X).abs()
            fees.iloc[0] = self.B.ix[0]
            fees.iloc[-1] = 0.
            fees *= self._fee

            self.asset_r -= fees
            self.r -= fees.sum(axis=1)

        self.r_log = np.log(self.r)

    @property
    def weights(self):
        return self.B

    @property
    def equity(self):
        return self.r.cumprod()

    @property
    def equity_decomposed(self):
        """ Return equity decomposed to individual assets. """
        return self.asset_r.cumprod()

    @property
    def asset_equity(self):
        return self.X.cumprod()

    @property
    def total_wealth(self):
        return self.r.prod()

    @property
    def profit_factor(self):
        x = self.r_log
        up = x[x > 0].sum()
        down = -x[x < 0].sum()
        return up / down if down != 0 else np.inf

    @property
    def sharpe(self):
        """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year.
        """
        return sharpe(self.r_log, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def information(self):
        """ Information ratio benchmarked against uniform CRP portfolio. """
        s = self.X.mean(axis=1)
        x = self.r_log - np.log(s)

        mu, sd = x.mean(), x.std()

        freq = self.freq()
        if sd > 1e-8:
            return mu / sd * np.sqrt(freq)
        elif mu > 1e-8:
            return np.inf * np.sign(mu)
        else:
            return 0.

    @property
    def ucrp_sharpe(self):
        result = CRP().run(self.X.cumprod())
        return result.sharpe

    @property
    def growth_rate(self):
        return self.r_log.mean() * self.freq()

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()

    @property
    def annualized_return(self):
        return np.exp(self.r_log.mean() * self.freq()) - 1

    @property
    def annualized_volatility(self):
        return np.exp(self.r_log).std() * np.sqrt(self.freq())

    @property
    def drawdown_period(self):
        ''' Returns longest drawdown perid. Stagnation is a drawdown too. '''
        x = self.equity
        period = [0.] * len(x)
        peak = 0
        for i in range(len(x)):
            # new peak
            if x[i] > peak:
                peak = x[i]
                period[i] = 0
            else:
                period[i] = period[i-1] + 1
        return max(period) * 252. / self.freq()

    @property
    def max_drawdown(self):
        ''' Returns highest drawdown in percentage. '''
        x = self.equity
        return max(1. - x / x.cummax())

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    @property
    def turnover(self):
        return self.B.diff().abs().sum().sum()

    def freq(self, x=None):
        """ Number of data items per year. If data does not contain
        datetime index, assume daily frequency with 252 trading days a year."""
        x = x or self.r
        return freq(x.index)

    def alpha_beta(self):
        rr = (self.X - 1).mean(1)

        m = OLS(self.r - 1, np.vstack([np.ones(len(self.r)), rr]).T)
        reg = m.fit()
        alpha, beta = reg.params.const * 252, reg.params.x1
        return alpha, beta

    def summary(self, name=None):
        alpha, beta = self.alpha_beta()
        return """Summary{}:
    Profit factor: {:.2f}
    Sharpe ratio: {:.2f}
    Information ratio (wrt UCRP): {:.2f}
    UCRP sharpe: {:.2f}
    Beta / Alpha: {:.2f} / {:.3%}
    Annualized return: {:.2%}
    Annualized volatility: {:.2%}
    Longest drawdown: {:.0f} days
    Max drawdown: {:.2%}
    Winning days: {:.1%}
    Turnover: {:.1f}
        """.format(
            '' if name is None else ' for ' + name,
            self.profit_factor,
            self.sharpe,
            self.information,
            self.ucrp_sharpe,
            beta,
            alpha,
            self.annualized_return,
            self.annualized_volatility,
            self.drawdown_period,
            self.max_drawdown,
            self.winning_pct,
            self.turnover,
            )

    def plot(self, weights=True, assets=True, portfolio_label='PORTFOLIO', show_only_important=True, **kwargs):
        """ Plot equity of all assets plus our strategy.
        :param weights: Plot weights as a subplot.
        :param assets: Plot asset prices.
        :return: List of axes.
        """
        res = ListResult([self], [portfolio_label])
        if not weights:
            ax1 = res.plot(assets=assets, **kwargs)
            return [ax1]
        else:
            if show_only_important:
                ix = self.B.abs().sum().nlargest(n=20).index
                B = self.B.loc[:, ix].copy()
                assets = B.columns if assets else False
                B['_others'] = self.B.drop(ix, 1).sum(1)
            else:
                B = self.B.copy()

            plt.figure(1)
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            res.plot(assets=assets, ax=ax1, **kwargs)
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

            # plot weights as lines
            if B.drop(['CASH'], 1, errors='ignore').values.min() < -0.01:
                B.sort_index(axis=1).plot(ax=ax2, ylim=(min(0., B.values.min()), max(1., B.values.max())),
                                          legend=False, color=_colors(len(assets) + 1))
            else:
                # fix rounding errors near zero
                if B.values.min() < 0:
                    pB = B - B.values.min()
                else:
                    pB = B
                pB.sort_index(axis=1).plot(ax=ax2, ylim=(0., max(1., pB.sum(1).max())),
                                           legend=False, color=_colors(len(assets) + 1), kind='area', stacked=True)
            plt.ylabel('weights')
            return [ax1, ax2]

    def hedge(self, result=None):
        """ Hedge results with results of other strategy (subtract weights).
        :param result: Other result object. Default is UCRP.
        :return: New AlgoResult object.
        """
        if result is None:
            result = CRP().run(self.X.cumprod())

        return AlgoResult(self.X, self.B - result.B)

    def plot_decomposition(self, **kwargs):
        """ Decompose equity into components of individual assets and plot
        them. Does not take fees into account. """
        ax = self.equity_decomposed.plot(**kwargs)
        return ax

    @property
    def importance(self):
        ws = self.weights.sum()
        return (ws / sum(ws)).order(ascending=False)

    def plot_total_weights(self):
        _, axes = plt.subplots(ncols=2)
        self.B.iloc[-1].sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Latest weights', ax=axes[1])
        self.B.sum().sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Total weights', ax=axes[0])


class ListResult(list, PickleMixin):
    """ List of AlgoResults. """

    def __init__(self, results=None, names=None):
        results = results if results is not None else []
        names = names if names is not None else []
        super(ListResult, self).__init__(results)
        self.names = names

    def append(self, result, name):
        super(ListResult, self).append(result)
        self.names.append(name)

    def to_dataframe(self):
        """ Calculate equities for all results and return one dataframe. """
        eq = {}
        for result, name in zip(self, self.names):
            eq[name] = result.equity
        return pd.DataFrame(eq)

    def save(self, filename, **kwargs):
        # do not save it with fees
        #self.fee = 0.
        #self.to_dataframe().to_pickle(*args, **kwargs)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        # df = pd.read_pickle(*args, **kwargs)
        # return cls([df[c] for c in df], df.columns)

        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def fee(self):
        return {name: result.fee for result, name in zip(self, self.names)}

    @fee.setter
    def fee(self, value):
        for result in self:
            result.fee = value

    def summary(self):
        return '\n'.join([result.summary(name) for result, name in zip(self, self.names)])


@contextlib.contextmanager
def mp_pool(n_jobs):
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    pool = multiprocessing.Pool(n_jobs)
    try:
        yield pool
    finally:
        pool.close()

def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1);
        if tmax >= s[ii+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] -1)/m

    return np.maximum(y-tmax,0.)


def log_progress_(i, total, by=1):
    """ Log progress by pcts. """
    progress = ((100 * i / total) // by) * by
    last_progress = ((100 * (i-1) / total) // by) * by

    if progress != last_progress:
        logging.debug('Progress: {}%...'.format(progress))


def combinations(S, r):
    """ Generator of all r-element combinations of stocks from portfolio S. """
    for ncols in itertools.combinations(S.columns, r):
        #yield S.iloc[:,ncols]
        yield S[list(ncols)]



class Algo(object):
    """ Base class for algorithm calculating weights for online portfolio.
    You have to subclass either step method to calculate weights sequentially
    or weights method, which does it at once. weights method might be useful
    for better performance when using maatrix calculation, but be careful about
    look-ahead bias.

    Upper case letters stand for matrix and lower case for vectors (such as
    B and b for weights).
    """

    # if true, replace missing values by last values
    REPLACE_MISSING = False

    # type of prices going into weights or step function
    #    ratio:  pt / pt-1
    #    log:    log(pt / pt-1)
    #    raw:    pt
    PRICE_TYPE = 'ratio'

    def __init__(self, min_history=None, frequency=1):
        """ Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
        :param frequency: algorithm should trade every `frequency` periods
        """
        self.min_history = min_history or 0
        self.frequency = frequency

    def init_weights(self, m):
        """ Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(m)

    def init_step(self, X):
        """ Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """
        pass

    def step(self, x, last_b, history):
        """ Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError('Subclass must implement this!')

    def _use_history_step(self):
        """ Use history parameter in step method? """
        step_args = inspect.getargspec(self.step)[0]
        return len(step_args) >= 4

    def weights(self, X, min_history=None, log_progress=True):
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        min_history = self.min_history if min_history is None else min_history

        # init
        B = X.copy() * 0.
        last_b = self.init_weights(X.shape[1])
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        # use history in step method?
        use_history = self._use_history_step()

        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.ix[t] = last_b

            # keep initial weights for min_history
            if t < min_history:
                continue

            # trade each `frequency` periods
            if (t + 1) % self.frequency != 0:
                continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t+1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            if log_progress:
                log_progress_(t, len(X), by=10)

        return B

    def _split_index(self, ix, nr_chunks, freq):
        """ Split index into chunks so that each chunk except of the last has length
        divisible by freq. """
        chunksize = int(len(ix) / freq / nr_chunks + 1) * freq
        return [ix[i*chunksize:(i+1)*chunksize] for i in range(len(ix) / chunksize + 1)]

    def run(self, S, n_jobs=1, log_progress=True):
        """ Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
        :param show_progress: Log computation progress. Works only for algos with
            defined step method.
        :param n_jobs: run step method in parallel (step method can't depend on last weights)
        """
        if log_progress:
            logging.debug('Running {}...'.format(self.__class__.__name__))

        if isinstance(S, ListResult):
            P = S.to_dataframe()
        else:
            P = S

        # convert prices to proper format
        X = self._convert_prices(P, self.PRICE_TYPE, self.REPLACE_MISSING)

        # get weights
        if n_jobs == 1:
            try:
                B = self.weights(X, log_progress=log_progress)
            except TypeError:   # weights are missing log_progress parameter
                B = self.weights(X)
        else:
            with mp_pool(n_jobs) as pool:
                ix_blocks = self._split_index(X.index, pool._processes * 2, self.frequency)
                min_histories = np.maximum(np.cumsum([0] + map(len, ix_blocks[:-1])) - 1, self.min_history)

                B_blocks = pool.map(_parallel_weights, [(self, X.ix[:ix_block[-1]], min_history, log_progress)
                                    for ix_block, min_history in zip(ix_blocks, min_histories)])

            # join weights to one dataframe
            B = pd.concat([B_blocks[i].ix[ix] for i, ix in enumerate(ix_blocks)])

        # cast to dataframe if weights return numpy array
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)

        if log_progress:
            logging.debug('{} finished successfully.'.format(self.__class__.__name__))

        # if we are aggregating strategies, combine weights from strategies
        # and use original assets
        if isinstance(S, ListResult):
            B = sum(result.B.mul(B[col], axis=0) for result, col in zip(S, B.columns))
            return AlgoResult(S[0].X, B)
        else:
            return AlgoResult(self._convert_prices(S, 'ratio'), B)

    def next_weights(self, S, last_b, **kwargs):
        """ Calculate weights for next day. """
        # use history in step method?
        use_history = self._use_history_step()
        history = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        x = history.iloc[-1]

        if use_history:
            b = self.step(x, last_b, history, **kwargs)
        else:
            b = self.step(x, last_b, **kwargs)
        return pd.Series(b, index=S.columns)

    def run_subsets(self, S, r, generator=False):
        """ Run algorithm on all stock subsets of length r. Note that number of such tests can be
        very large.
        :param S: stock prices
        :param r: number of stocks in a subset
        :param generator: yield results
        """
        def subset_generator():
            total_subsets = comb(S.shape[1], r)

            for i, S_sub in enumerate(combinations(S, r)):
                # run algorithm on given subset
                result = self.run(S_sub, log_progress=False)
                name = ', '.join(S_sub.columns.astype(str))

                # log progress by 1 pcts
                log_progress_(i, total_subsets, by=1)

                yield result, name
            raise StopIteration

        if generator:
            return subset_generator()
        else:
            results = []
            names = []
            for result, name in subset_generator():
                results.append(result)
                names.append(name)
            return ListResult(results, names)

    @classmethod
    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == 'raw':
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.iteritems():
                init_val = s.ix[s.first_valid_index()]
                r[name] = s / init_val
            X = pd.DataFrame(r)

            if replace_missing:
                X.ix[0] = 1.
                X = X.fillna(method='ffill')

            return X

        elif method == 'absolute':
            return S

        elif method in ('ratio', 'log'):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method='ffill')
            for name, s in X.iteritems():
                X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.

            if replace_missing:
                X = X.fillna(1.)

            return np.log(X) if method == 'log' else X

        else:
            raise ValueError('invalid price conversion method')

    @classmethod
    def run_combination(cls, S, **kwargs):
        """ Get equity of algo using all combinations of parameters. All
        values in lists specified in kwargs will be optimized. Other types
        will be passed as they are to algo __init__ (like numbers, strings,
        tuples).
        Return ListResult object, which is basically a wrapper of list of AlgoResult objects.
        It is possible to pass ListResult to Algo or run_combination again
        to get AlgoResult. This is useful for chaining of Algos.

        Example:
            S = ...load data...
            list_results = Anticor.run_combination(S, alpha=[0.01, 0.1, 1.])
            result = CRP().run(list_results)

        :param S: Stock prices.
        :param kwargs: Additional arguments to algo.
        :param n_jobs: Use multiprocessing (-1 = use all cores). Use all cores by default.
        """
        if isinstance(S, ListResult):
            S = S.to_dataframe()

        n_jobs = kwargs.pop('n_jobs', -1)

        # extract simple parameters
        simple_params = {k: kwargs.pop(k) for k, v in kwargs.items()
                         if not isinstance(v, list)}

        # iterate over all combinations
        names = []
        params_to_try = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(params.items() + simple_params.items())
            params_to_try.append(all_params)

            # create name with format param:value
            name = ','.join([str(k) + '=' + str(v) for k, v in params.items()])
            names.append(name)

        # try all combinations in parallel
        with mp_pool(n_jobs) as pool:
            results = pool.map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])
        results = map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])

        return ListResult(results, names)

    def copy(self):
        return copy.deepcopy(self)


class OLMAR(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()


    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        projc = simplex_proj(b)
        return projc


def opt_weights(X, metric='return', max_leverage=1, rf_rate=0., alpha=0., freq=252, no_cash=False, sd_factor=1., **kwargs):
    """ Find best constant rebalanced portfolio with regards to some metric.
    :param X: Prices in ratios.
    :param metric: what performance metric to optimize, can be either `return` or `sharpe`
    :max_leverage: maximum leverage
    :rf_rate: risk-free rate for `sharpe`, can be used to make it more aggressive
    :alpha: regularization parameter for volatility in sharpe
    :freq: frequency for sharpe (default 252 for daily data)
    :no_cash: if True, we can't keep cash (that is sum of weights == max_leverage)
    """
    assert metric in ('return', 'sharpe', 'drawdown')

    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    if metric == 'return':
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    elif metric == 'sharpe':
        objective = lambda b: -sharpe(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)),
                                      rf_rate=rf_rate, alpha=alpha, freq=freq, sd_factor=sd_factor)
    elif metric == 'drawdown':
        def objective(b):
            R = np.dot(X - 1, b) + 1
            L = np.cumprod(R)
            dd = max(1 - L / np.maximum.accumulate(L))
            annual_ret = np.mean(R) ** freq - 1
            return -annual_ret / (dd + alpha)

    if no_cash:
        cons = ({'type': 'eq', 'fun': lambda b: max_leverage - sum(b)},)
    else:
        cons = ({'type': 'ineq', 'fun': lambda b: max_leverage - sum(b)},)

    from scipy.optimize import minimize
    while True:
        # problem optimization
        res = minimize(objective, x_0, bounds=[(0., max_leverage)]*len(x_0), constraints=cons, method='slsqp', **kwargs)

        # result can be out-of-bounds -> try it again
        EPS = 1E-7
        if (res.x < 0. - EPS).any() or (res.x > max_leverage + EPS).any():
            X = X + np.random.randn(1)[0] * 1E-5
            logging.debug('Optimal weights not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution does not exist, use zero weights.')
                res.x = np.zeros(X.shape[1])
            else:
                logging.warning('Converged, but not successfully.')
            break

    return res.x


def freq(ix):
    """ Number of data items per year. If data does not contain
    datetime index, assume daily frequency with 252 trading days a year."""
    assert isinstance(ix, pd.Index), 'freq method only accepts pd.Index object'

    # sort if data is not monotonic
    if not ix.is_monotonic:
        ix = ix.sort_values()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 365.
    else:
        return 252.

# add alias to allow use of freq keyword in functions
_freq = freq

class CRP(Algo):
    """ Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super(CRP, self).__init__()
        self.b = b


    def step(self, x, last_b):
        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b


    def weights(self, X):
        if self.b is None:
            return np.ones(X.shape) / X.shape[1]
        else:
            return np.repeat([self.b], X.shape[0], axis=0)


class BCRP(CRP):
    """ Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed
    with hindsight. It is often used as benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, **kwargs):
        self.opt_weights_kwargs = kwargs

    def weights(self, X):
        """ Find weights which maximize return on X in hindsight! """
        # update frequency
        self.opt_weights_kwargs['freq'] = freq(X.index)

        self.b = opt_weights(X, **self.opt_weights_kwargs)

        return super(BCRP, self).weights(X)

def freq(ix):
    """ Number of data items per year. If data does not contain
    datetime index, assume daily frequency with 252 trading days a year."""
    assert isinstance(ix, pd.Index), 'freq method only accepts pd.Index object'

    # sort if data is not monotonic
    if not ix.is_monotonic:
        ix = ix.sort_values()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 365.
    else:
        return 252.

# add alias to allow use of freq keyword in functions
_freq = freq

class CRP(Algo):
    """ Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super(CRP, self).__init__()
        self.b = b


    def step(self, x, last_b):
        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b


    def weights(self, X):
        if self.b is None:
            return np.ones(X.shape) / X.shape[1]
        else:
            return np.repeat([self.b], X.shape[0], axis=0)


class BCRP(CRP):
    """ Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed
    with hindsight. It is often used as benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, **kwargs):
        self.opt_weights_kwargs = kwargs

    def weights(self, X):
        """ Find weights which maximize return on X in hindsight! """
        # update frequency
        self.opt_weights_kwargs['freq'] = freq(X.index)

        self.b = opt_weights(X, **self.opt_weights_kwargs)

        return super(BCRP, self).weights(X)

class EG(Algo):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
    """

    def __init__(self, eta=0.05):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(EG, self).__init__()
        self.eta = eta


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b):
        b = last_b * np.exp(self.eta * x / sum(x * last_b))
        return b / sum(b)


def bcrp_weights(X):
    """ Find best constant rebalanced portfolio.
    :param X: Prices in ratios.
    """
    return opt_weights(X)

class BNN(Algo):
    """ Nearest neighbor based strategy. It tries to find similar sequences of price in history and
    then maximize objective function (that is profit) on the days following them.

    Reference:
        L. Gyorfi, G. Lugosi, and F. Udina. Nonparametric kernel based sequential
        investment strategies. Mathematical Finance 16 (2006) 337–357.
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, k=5, l=10):
        """
        :param k: Sequence length.
        :param l: Number of nearest neighbors.
        """

        super(BNN, self).__init__(min_history=k+l-1)

        self.k = k
        self.l = l


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # find indices of nearest neighbors throughout history
        ixs = self.find_nn(history, self.k, self.l)

        # get returns from the days following NNs
        J = history.iloc[[history.index.get_loc(i) + 1 for i in ixs]]

        # get best weights
        return bcrp_weights(J)


    def find_nn(self, H, k, l):
        """ Note that nearest neighbors are calculated in a different (more efficient) way than shown
        in the article.

        param H: history
        """
        # calculate distance from current sequence to every other point
        D = H * 0
        for i in range(1, k+1):
            D += (H.shift(i-1) - H.iloc[-i])**2
        D = D.sum(1).iloc[:-1]

        # sort and find nearest neighbors
        D.sort_values()
        return D.index[:l]


def rolling_cov_pairwise(df, *args, **kwargs):
    d = {}
    for c in df.columns:
        d[c] = pd.rolling_cov(df[c], df, *args, **kwargs)
    p = pd.Panel(d)
    return p.transpose(1, 0, 2)


def rolling_corr(x, y, **kwargs):
    """ Rolling correlation between columns from x and y. """
    def rolling(dataframe, *args, **kwargs):
        ret = dataframe.copy()
        for col in ret:
            ret[col] = ret[col].rolling(*args, **kwargs).mean()
        return ret

    n, k = x.shape

    EX = rolling(x, **kwargs)
    EY = rolling(y, **kwargs)
    EX2 = rolling(x ** 2, **kwargs)
    EY2 = rolling(y ** 2, **kwargs)

    RXY = np.zeros((n, k, k))

    for i, col_x in enumerate(x):
        for j, col_y in enumerate(y):
            DX = EX2[col_x] - EX[col_x] ** 2
            DY = EY2[col_y] - EY[col_y] ** 2
            product = x[col_x] * y[col_y]
            RXY[:, i, j] = product.rolling(**kwargs).mean() - EX[col_x] * EY[col_y]
            RXY[:, i, j] = RXY[:, i, j] / np.sqrt(DX * DY)

    return RXY, EX.values


class ONS(Algo):
    """
    Online newton step algorithm.

    Reference:
        A.Agarwal, E.Hazan, S.Kale, R.E.Schapire.
        Algorithms for Portfolio Management based on the Newton Method, 2006.
        http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_AgarwalHKS06.pdf
    """

    REPLACE_MISSING = True

    def __init__(self, delta=0.125, beta=1., eta=0.):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super(ONS, self).__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta


    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        m = X.shape[1]
        self.A = np.mat(np.eye(m))
        self.b = np.mat(np.zeros(m)).T


    def step(self, r, p):
        # calculate gradient
        grad = np.mat(r / np.dot(p, r)).T
        # update A
        self.A += grad * grad.T
        # update b
        self.b += (1 + 1./self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M):
        """ Projection of x to simplex indiced by matrix M. Uses quadratic programming.
        """
        m = M.shape[0]

        P = matrix(2*M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m,1)))
        A = matrix(np.ones((1,m)))
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])



class Anticor(Algo):
    """ Anticor (anti-correlation) is a heuristic portfolio selection algorithm.
    It adopts the consistency of positive lagged cross-correlation and negative
    autocorrelation to adjust the portfolio. Eventhough it has no known bounds and
    hence is not considered to be universal, it has very strong empirical results.

    It has implemented C version in scipy.weave to improve performance (around 10x speed up).
    Another option is to use Numba.

    Reference:
        A. Borodin, R. El-Yaniv, and V. Gogan.  Can we learn to beat the best stock, 2005.
        http://www.cs.technion.ac.il/~rani/el-yaniv-papers/BorodinEG03.pdf
    """

    def __init__(self, window=30, c_version=True):
        """
        :param window: Window parameter.
        :param c_version: Use c_version, up to 10x speed-up.
        """
        super(Anticor, self).__init__()
        self.window = window
        self.c_version = c_version


    def weights(self, X):
        window = self.window
        port = X
        n, m = port.shape
        weights = 1. / m * np.ones(port.shape)

        CORR, EX = rolling_corr(port, port.shift(window), window=window)

        for t in range(n - 1):
            M = CORR[t, :, :]
            mu = EX[t, :]

            # claim[i,j] is claim from stock i to j
            claim = np.zeros((m, m))

            for i in range(m):
                for j in range(m):
                    if i == j: continue

                    if mu[i] > mu[j] and M[i, j] > 0:
                        claim[i, j] += M[i, j]
                        # autocorrelation
                        if M[i, i] < 0:
                            claim[i, j] += abs(M[i, i])
                        if M[j, j] < 0:
                            claim[i, j] += abs(M[j, j])

            # calculate transfer
            transfer = claim * 0.
            for i in range(m):
                total_claim = sum(claim[i, :])
                if total_claim != 0:
                    transfer[i, :] = weights[t, i] * claim[i, :] / total_claim

            # update weights
            weights[t + 1, :] = weights[t, :] + np.sum(transfer, axis=0) - np.sum(transfer, axis=1)
        return weights


class CWMR(Algo):
    """ Confidence weighted mean reversion.

    Reference:
        B. Li, S. C. H. Hoi, P.L. Zhao, and V. Gopalkrishnan.
        Confidence weighted mean reversion strategy for online portfolio selection, 2013.
        http://jmlr.org/proceedings/papers/v15/li11b/li11b.pdf
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, eps=-0.5, confidence=0.95):
        """
        :param eps: Mean reversion threshold (expected return on current day must be lower
                    than this threshold). Recommended value is -0.5.
        :param confidence: Confidence parameter for profitable mean reversion portfolio.
                    Recommended value is 0.95.
        """
        super(CWMR, self).__init__()

        # input check
        if not (0 <= confidence <= 1):
            raise ValueError('confidence must be from interval [0,1]')

        self.eps = eps
        self.theta = sc.stats.norm.ppf(confidence)


    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        m = X.shape[1]
        self.sigma = np.matrix(np.eye(m) / m**2)


    def step(self, x, last_b):
        # initialize
        m = len(x)
        mu = np.matrix(last_b).T
        sigma = self.sigma
        theta = self.theta
        eps = self.eps
        x = np.matrix(x).T    # matrices are easier to manipulate

        # 4. Calculate the following variables
        M = mu.T * x
        V = x.T * sigma * x
        x_upper = sum(diag(sigma) * x) / trace(sigma)

        # 5. Update the portfolio distribution
        mu, sigma = self.update(x, x_upper, mu, sigma, M, V, theta, eps)

        # 6. Normalize mu and sigma
        mu = simplex_proj(mu)
        sigma = sigma / (m**2 * trace(sigma))
        """
        sigma(sigma < 1e-4*eye(m)) = 1e-4;
        """
        self.sigma = sigma
        return mu

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        # lambda from equation 7
        foo = (V - x_upper * x.T * np.sum(sigma, axis=1)) / M**2 + V * theta**2 / 2.
        a = foo**2 - V**2 * theta**4 / 4
        b = 2 * (eps - log(M)) * foo
        c = (eps - log(M))**2 - V * theta**2

        a,b,c = a[0,0], b[0,0], c[0,0]

        lam = max(0,
                  (-b + sqrt(b**2 - 4 * a * c)) / (2. * a),
                  (-b - sqrt(b**2 - 4 * a * c)) / (2. * a))
        # bound it due to numerical problems
        lam = min(lam, 1E+7)

        # update mu and sigma
        U_sqroot = 0.5 * (-lam * theta * V + sqrt(lam**2 * theta**2 * V**2 + 4*V))
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + theta * lam / U_sqroot * diag(x)**2)
        """
        tmp_sigma = inv(inv(sigma) + theta*lam/U_sqroot*diag(xt)^2);
        % Don't update sigma if results are badly scaled.
        if all(~isnan(tmp_sigma(:)) & ~isinf(tmp_sigma(:)))
            sigma = tmp_sigma;
        end
        """
        return mu, sigma


class PAMR(Algo):
    """ Passive aggressive mean reversion strategy for portfolio selection.
    There are three variants with different parameters, see original article
    for details.

    Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/PAMR_ML_final.pdf
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, eps=0.5, C=500, variant=0):
        """
        :param eps: Control parameter for variant 0. Must be >=0, recommended value is
                    between 0.5 and 1.
        :param C: Control parameter for variant 1 and 2. Recommended value is 500.
        :param variant: Variants 0, 1, 2 are available.
        """
        super(PAMR, self).__init__()

        # input check
        if not(eps >= 0):
            raise ValueError('epsilon parameter must be >=0')

        if variant == 0:
            if eps is None:
                raise ValueError('eps parameter is required for variant 0')
        elif variant == 1 or variant == 2:
            if C is None:
                raise ValueError('C parameter is required for variant 1,2')
        else:
            raise ValueError('variant is a number from 0,1,2')

        self.eps = eps
        self.C = C
        self.variant = variant


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b):
        # calculate return prediction
        b = self.update(last_b, x, self.eps, self.C)
        return b


    def update(self, b, x, eps, C):
        """ Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        le = max(0., np.dot(b, x) - eps)

        if self.variant == 0:
            lam = le / np.linalg.norm(x - x_mean)**2
        elif self.variant == 1:
            lam = min(C, le / np.linalg.norm(x - x_mean)**2)
        elif self.variant == 2:
            lam = le / (np.linalg.norm(x - x_mean)**2 + 0.5 / C)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b - lam * (x - x_mean)

        # project it onto simplex
        return simplex_proj(b)




def norm(x):
    if isinstance(x, pd.Series):
        axis = 0
    else:
        axis = 1
    return np.sqrt((x**2).sum(axis=axis))


class RMR(OLMAR):
    """ Robust Median Reversion. Strategy exploiting mean-reversion by robust
    L1-median estimator. Practically the same as OLMAR.

    Reference:
        Dingjiang Huang, Junlong Zhou, Bin Li, Steven C.H. Hoi, Shuigeng Zhou
        Robust Median Reversion Strategy for On-Line Portfolio Selection, 2013.
        http://ijcai.org/papers13/Papers/IJCAI13-296.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10., tau=0.001):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        :param tau: Precision for finding median. Recommended value is around 0.001. Strongly
                    affects algo speed.
        """
        super(RMR, self).__init__(window, eps)
        self.tau = tau

    def predict(self, x, history):
        """ find L1 median to historical prices """
        y = history.mean()
        y_last = None
        while y_last is None or norm(y - y_last) / norm(y_last) > self.tau:
            y_last = y
            d = norm(history - y)
            y = history.div(d, axis=0).sum() / (1. / d).sum()
        return y / x


class WMAMR(PAMR):
    """ Weighted Moving Average Passive Aggressive Algorithm for Online Portfolio Selection.
    It is just a combination of OLMAR and PAMR, where we use mean of past returns to predict
    next day's return.

    Reference:
        Li Gao, Weiguo Zhang
        Weighted Moving Averag Passive Aggressive Algorithm for Online Portfolio Selection, 2013.
        http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6643896
    """

    PRICE_TYPE = 'ratio'

    def __init__(self, window=5, **kwargs):
        """
        :param w: Windows length for moving average.
        :param kwargs: Additional arguments for PAMR.
        """
        super(WMAMR, self).__init__(**kwargs)

        if window < 1:
            raise ValueError('window parameter must be >=1')
        self.window = window


    def step(self, x, last_b, history):
        xx = history[-self.window:].mean()
        # calculate return prediction
        b = self.update(last_b, xx, self.eps, self.C)
        return b



class CORN(Algo):
    """
    Correlation-driven nonparametric learning approach. Similar to anticor but instead
    of distance of return vectors they use correlation.
    In appendix of the article, universal property is proven.

    Two versions are available. Fast which provides around 2x speedup, but uses more memory
    (linear in window) and slow version which is memory efficient. Most efficient would
    be to rewrite it in sweave or numba.

    Reference:
        B. Li, S. C. H. Hoi, and V. Gopalkrishnan.
        Corn: correlation-driven nonparametric learning approach for portfolio selection, 2011.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/TIST-CORN.pdf
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, window=5, rho=0.1, fast_version=True):
        """
        :param window: Window parameter.
        :param rho: Correlation coefficient threshold. Recommended is 0.
        :param fast_version: If true, use fast version which provides around 2x speedup, but uses
                             more memory.
        """
        # input check
        if not(-1 <= rho <= 1):
            raise ValueError('rho must be between -1 and 1')
        if not(window >= 2):
            raise ValueError('window must be greater than 2')

        super(CORN, self).__init__()
        self.window = window
        self.rho = rho
        self.fast_version = fast_version

        # assign step method dynamically
        self.step = self.step_fast if self.fast_version else self.step_slow


    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        if self.fast_version:
            # redefine index to enumerate
            X.index = range(len(X))

            foo = [X.shift(i) for i in range(self.window)]
            self.X_flat = pd.concat(foo, axis=1)
            self.X = X
            self.t = self.min_history - 1


    def step_slow(self, x, last_b, history):
        if len(history) <= self.window:
            return last_b
        else:
            # init
            window = self.window
            indices = []
            m = len(x)

            # calculate correlation with predecesors
            X_t = history.iloc[-window:].values.flatten()
            for i in range(window, len(history)):
                X_i = history.ix[i-window:i-1].values.flatten()
                if np.corrcoef(X_t, X_i)[0,1] >= self.rho:
                    indices.append(i)

            # calculate optimal portfolio
            C = history.ix[indices, :]

            if C.shape[0] == 0:
                b = np.ones(m) / float(m)
            else:
                b = self.optimal_weights(C)

            return b


    def step_fast(self, x, last_b):
        # iterate time
        self.t += 1

        if self.t < self.window:
            return last_b
        else:
            # init
            window = self.window
            m = len(x)

            X_t = self.X_flat.ix[self.t]
            X_i = self.X_flat.iloc[window-1 : self.t]
            c = X_i.apply(lambda r: np.corrcoef(r.values, X_t.values)[0,1], axis=1)

            C = self.X.ix[c.index[c >= self.rho] + 1]

            if C.shape[0] == 0:
                b = np.ones(m) / float(m)
            else:
                b = self.optimal_weights(C)

            return b

    def optimal_weights(self, X):
        freq = _freq(X.index)
        return opt_weights(X, freq=freq)



def zion__batch_online_portfolios(dataset, portfolios, algo_name='bcrp', batch_size=15, batched=True):
    """
    We compute these lol
    """

    ALGOS_AVAILABLE = {
        'olmar': lambda: OLMAR(),
        'bnn': lambda: BNN(),
        'eg': lambda: EG(),
        'bcrp': lambda: BCRP(),
        'ons': lambda: ONS(),
        'anticor': lambda: Anticor(),
        'cwmr': lambda: CWMR(),
        'pamr': lambda: PAMR(),
        'rmr': lambda: RMR(),
        'corn': lambda: CORN(),
        'wmamr': lambda: WMAMR(),
    }
    ALGO = ALGOS_AVAILABLE[algo_name]
    def algo_target_fn(rets, window=5, eps=10):
        algo = ALGO()
        if algo.PRICE_TYPE == 'raw':
            prices = (1 + rets.fillna(0)).cumprod()
        elif algo.PRICE_TYPE == 'ratio':
            prices = 1 + rets.fillna(0)
        elif algo.PRICE_TYPE == 'log':
            prices = np.log(1 + rets.fillna(0))

        result = algo.run(prices)
        weights = xr.DataArray(result.weights, dims=['time', 'ticker'])
        weights = weights / weights.sum('ticker')
        return weights

    # For future tests
    target_fn = algo_target_fn

    # Set-up
    # Not sure if best is additional_lag=1 or additional_lag=0
    from toolz import partition_all
    portfolios_returns = 1 + zion_xr__backtest(dataset, portfolios, transaction_costs=0, additional_lag=0)['returns'].to_dataframe().loc['2000':]
    if batched:
        batches = list(partition_all(batch_size, list(portfolios.data_vars.keys())))
    else:
        batches = [list(portfolios.data_vars.keys())]
    from apex.toolz.dask import ApexDaskClient as Client
    pool = Client()
    olmar_futures = [pool.submit(target_fn, portfolios_returns[list(x)]) for x in batches]
    olmar_results = []
    for future in olmar_futures:
        try:
            olmar_results.append(future.result())
        except:
            continue

    olmar_results_ds = {f'algo_ix={ix}': olmar_results[ix] for ix in range(len(olmar_results))}
    olmar_portfolios = {}
    for batch_name, batch_portfolio in olmar_results_ds.items():
        batch_strategies =  batch_portfolio.ticker.values.tolist()
        finalized_portfolio = sum(portfolios[x].fillna(0) * batch_portfolio.sel(ticker=x).fillna(0) for x in batch_strategies)
        finalized_portfolio = finalized_portfolio / np.abs(finalized_portfolio).sum('ticker')
        olmar_portfolios[batch_name] = finalized_portfolio
    olmar_portfolios = xr.Dataset(olmar_portfolios)
    if len(olmar_portfolios) == 1:
        return olmar_portfolios['algo_ix=0']
    return olmar_portfolios


def zion__online_portfolio(returns, algo_name='bcrp', olmar_params={'window': 5, 'eps': 10}):
    """
    We compute these lol
    """

    ALGOS_AVAILABLE = {
        'olmar': lambda: OLMAR(window=olmar_params['window'], eps=olmar_params['eps']),
        'bnn': lambda: BNN(),
        'eg': lambda: EG(),
        'bcrp': lambda: BCRP(),
        'ons': lambda: ONS(),
        'anticor': lambda: Anticor(),
        'cwmr': lambda: CWMR(),
        'pamr': lambda: PAMR(),
        'rmr': lambda: RMR(),
        'corn': lambda: CORN(),
        'wmamr': lambda: WMAMR(),
    }
    ALGO = ALGOS_AVAILABLE[algo_name]
    def algo_target_fn(rets):
        algo = ALGO()
        if algo.PRICE_TYPE == 'raw':
            prices = (1 + rets.fillna(0)).cumprod()
        elif algo.PRICE_TYPE == 'ratio':
            prices = 1 + rets.fillna(0)
        elif algo.PRICE_TYPE == 'log':
            prices = np.log(1 + rets.fillna(0))
        result = algo.run(prices)
        weights = xr.DataArray(result.weights, dims=['time', 'ticker'])
        weights = weights / weights.sum('ticker')
        return weights

    online_portfolio = algo_target_fn(returns)
    return online_portfolio
