import pickle
from pathlib import Path

import funcy as fy
import numba as nb
import numpy as np
import pandas as pd
import pygmo as pg
from apex.alpha.analysis import (alpha_performance_stats,
                                 get_sign_adjusted_results)
from apex.alpha.market_alphas import MARKET_ALPHAS
from apex.pipelines.factor_portfolio import (construct_portfolio,
                                             default_covariance)
from apex.security import ApexSecurity
from apex.toolz.bloomberg import ApexBloomberg, apex__adjusted_market_data
from apex.toolz.caches import UNIVERSE_DATA_CACHING
from apex.toolz.dask import ApexDaskClient
from apex.toolz.reporting import (SalientDataframeExcelReport,
                                  SalientExcelLineChartSheet,
                                  SalientReportWorkbook)
from apex.universe import APEX_UNIVERSES
from dask import delayed
from distributed import fire_and_forget
import matplotlib.pyplot as plt

@nb.njit
def fitness_fn(covariance, target_risk_contribution, volatility_target, scores, weights):
    ctw = covariance @ weights
    port_var = weights @ ctw

    mcr_abs = np.abs(np.diag(weights) @ ctw)
    mcr_abs = mcr_abs / port_var
    objective_fn = np.sum(np.power(mcr_abs - target_risk_contribution, 2))

    they_are_wts = np.power(np.sum(np.abs(weights)) - 1., 2)
    return objective_fn, they_are_wts

class risk_budget_weights:
    def __init__(self, dim=None, covariance=None, volatility_target=None, scores=None):
        self.dim = dim
        self.covariance = covariance
        self.volatility_target = volatility_target
        self.scores = scores
        self.target_risk_contribution = np.abs(self.scores)/np.sum(np.abs(self.scores))

        upper_bounds, lower_bounds = [], []
        for x in self.scores:
            if np.sign(x) > 0:
                upper_bounds.append(1)
                lower_bounds.append(0.)
            else:
                upper_bounds.append(0.)
                lower_bounds.append(-1)

        self.ub = upper_bounds
        self.lb = lower_bounds

    def get_bounds(self):
        return (self.lb, self.ub)

    def fitness(self, w):
        return fitness_fn(self.covariance, self.target_risk_contribution, self.volatility_target, self.scores, w)

    def get_name(self):
        return "Risk Budgeted Weights"

    def get_nic(self):
        return 1

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


def opt_risk_budget_problem(problem):
    """
    Problem: maximize sum(abs(signal_i)*log(abs(signal_i)))
    s.t. w'Cw < sigma
    """
    ipt = pg.ipopt()
    ipt.set_string_option("hessian_approximation", "limited-memory")
    ipt.set_numeric_option("tol", 5E-4) # Don't care about anything smaller than 5 basis pts
    ipt_algo = pg.algorithm(ipt)
    prb = pg.problem(problem)
    prb.c_tol = [0.15]

    pop = pg.population(prb, size=1)
    pop = ipt_algo.evolve(pop)
    return pop


def construct_risk_budget_weights(scores, covariance_mtx, volatility_target=0.2):
    problem = risk_budget_weights(dim=scores.values.size, covariance=covariance_mtx.values,
                                  volatility_target=volatility_target,
                                  scores=scores.values)
    algo_res = opt_risk_budget_problem(problem).champion_x
    algo_res = pd.Series(algo_res, scores.index)
    return algo_res

def construct_portfolio_for_date(date, signal, returns, volatility_target=0.2):
    signal = signal.loc[:date].iloc[-1].dropna()
    returns = returns.loc[:date].iloc[:-1][signal.index.tolist()]
    covariance = default_covariance(returns)
    weights = construct_risk_budget_weights(signal, covariance, volatility_target=0.2)
    return weights


def construct_alpha_portfolio_from_signal(signal, market_data, volatility_target=0.2):
    weights = None
    returns = market_data['returns']
    signal.index = pd.to_datetime(signal.index)
    returns.index = pd.to_datetime(returns.index)
    signal = signal.loc['1996' :].dropna(how='all', axis=1).dropna(how='all', axis=0)
    returns = returns[[x for x in signal.columns if x in returns.columns]]
    signal = signal[[x for x in signal.columns if x in returns.columns]]
    clt = ApexDaskClient()
    signal = signal.loc['1996':]
    s_signal = clt.scatter(signal)
    s_returns = clt.scatter(market_data['returns'])

    construct_portfolio_fn = fy.partial(construct_portfolio_for_date,
                                        volatility_target=volatility_target)

    result = {}
    for date in signal.index:
        result[date] = clt.submit(construct_portfolio_fn, date, s_signal, s_returns)
    result = fy.zipdict(result.keys(), clt.gather(list(result.values())))
    result = pd.DataFrame(result).T
    result = result.fillna(0)
    return result


def construct_alpha_portfolio_from_signal__local(signal, market_data, volatility_target=0.2):
    weights = None
    returns = market_data['returns']
    signal.index = pd.to_datetime(signal.index)
    returns.index = pd.to_datetime(returns.index)
    signal = signal.loc['1996' :].dropna(how='all', axis=1).dropna(how='all', axis=0)
    returns = returns[[x for x in signal.columns if x in returns.columns]]
    signal = signal[[x for x in signal.columns if x in returns.columns]]
    construct_portfolio_fn = fy.partial(construct_portfolio_for_date,
                                        volatility_target=volatility_target)

    result = {}
    for date in signal.index:
        result[date] = construct_portfolio_fn(date, signal, returns)
    #result = fy.zipdict(result.keys(), clt.gather(list(result.values())))
    result = pd.DataFrame(result).T
    result = result.fillna(0)
    return result


def create_report_workbook(name, results, file_loc, tags=[]):
    report_wb = SalientReportWorkbook()

    ### WEIGHTS
    weights_data = results['weights'].copy()
    weights_data[weights_data == 0.0] = np.nan
    weights_data = weights_data.dropna(how='all')
    weights_sheet = SalientDataframeExcelReport(
        weights_data,
        'Weights',
    )

    ### Performance
    performance_data = pd.DataFrame({'Performance Metrics': results['performance']['performance_stats']})
    stats_sheet = SalientDataframeExcelReport(
        performance_data,
        'Statistics',
    )

    ### Drawdown
    dd_data = results['performance']['drawdown_stats']
    dd_sheet = SalientDataframeExcelReport(
        dd_data,
        'Drawdown',
    )
    ### Chart
    cumulative_returns = (results['returns'].fillna(0) + 1).cumprod()
    cum_rets_sheet = SalientDataframeExcelReport(
        pd.DataFrame({'Cumulative Returns': cumulative_returns}),
        'Cumulative Returns',
    )

    # Add all
    for sheet in [cum_rets_sheet, stats_sheet, dd_sheet, weights_sheet]:
        report_wb.add_sheet(sheet)
    report_wb.save(file_loc)

    return file_loc



def analyze_portfolio(weights, security_returns):
    sign_adjusted_results = get_sign_adjusted_results(weights.copy(), security_returns)
    sign_adjusted_results['weights']['cash'] = 0.0
    weights, returns = sign_adjusted_results['weights'], sign_adjusted_results['returns']
    stats = alpha_performance_stats(weights, returns)
    result = {
        'weights': weights,
        'attribution': sign_adjusted_results['attribution'],
        'factor_sign': sign_adjusted_results['factor_sign'],
        'returns': returns,
        'performance': stats
    }
    return result

def get_experiment_run_loc(name, tags):
    file_loc = Path(f'/mnt/data/experiments/alpha/reports/{name}/')
    file_loc.mkdir(exist_ok=True, parents=True)
    if tags:
        tag = '.'.join(tags)
    else:
        tag = ''
    file_loc = file_loc / f'{tag}.pkl'
    return file_loc

def get_experiment_run_dir(name):
    file_loc = Path(f'/mnt/data/experiments/alpha/reports/{name}/')
    file_loc.mkdir(exist_ok=True, parents=True)
    return file_loc

DEFAULT_SMOOTH_VALUES = list(np.arange(1, 26)) + list(np.arange(30, 105, 5)) + list(np.arange(110, 260, 10))

def generate_experiment_runs(universe,
    alpha_name, weights,
    returns, smooth=DEFAULT_SMOOTH_VALUES,
    volatility_target=0.2, experiment_id=2, tags=[]):
    client = MLFlowTrackingClient(tracking_uri="http://10.15.201.160:20000")
    experiment_name = f'{alpha_name}'
    for exp in client.list_experiments():
        if exp.name == experiment_name:
            experiment_id = exp.experiment_id
            break
    else:
        experiment_id = client.create_experiment(experiment_name)

    ds = pd.Timestamp.now().strftime('%Y-%m-%d')
    for smooth in DEFAULT_SMOOTH_VALUES:
        run_name = ':'.join([
            ds,
            str(smooth)
        ])
        run = client.create_run(experiment_id,
            run_name=run_name,
        )
        run_id = run.info.run_uuid
        smooth_w = weights.ewm(span=smooth).mean()
        curr_vol = (smooth_w.shift(2) * returns).sum(axis=1).std() * np.sqrt(252)
        smooth_w = smooth_w * volatility_target / curr_vol
        portfolio_analytics = analyze_portfolio(smooth_w, returns)
        metrics_performance = portfolio_analytics['performance']['performance_stats'].to_dict()
        client.log_param(run_id, 'weight_smoothing', smooth)
        metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'cumulative_returns', 'annual_returns', 'annual_volatility']
        for param_name, param_value in metrics_performance.items():
            client.log_metric(run_id, param_name, param_value)
        client.log_param(run_id, 'universe', universe)
        client.log_param(run_id, 'factor_sign', portfolio_analytics['factor_sign'])
        report_loc = get_experiment_run_loc(alpha_name, tags=[f'smooth_{smooth}'])
        report_dir = get_experiment_run_dir(alpha_name)
        with open(report_loc, 'wb+') as f:
            pickle.dump(portfolio_analytics, f)
        client.log_artifact(run_id, report_loc)
    return True


def validity_table(market_data, min_securities=12):
    last_prices = market_data['px_last'].fillna(method='ffill', limit=2)
    initial_result = (~(last_prices.isnull())).shift(250).fillna(False)
    price_cutoff = last_prices > 10
    result = price_cutoff & initial_result
    total_secs = (result).sum(axis=1) > min_securities
    total_secs = total_secs[total_secs].dropna()
    return result.reindex(total_secs.index)

def min_max_scaler(raw_data, axis=0, feature_range=(-3, 3)):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        return min_max_scaler(raw_data.T, axis=0).T

    data_min = data.min(axis=1)
    data_max = data.max(axis=1)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    scale = (feature_range[1] - feature_range[0])/data_range
    min_scale = feature_range[0] - data_min.multiply(scale)
    result = data.multiply(scale, axis=0).add(min_scale, axis=0)
    return result

def max_abs_scaler(raw_data, axis=0, subtract_mean=True):
    """
    Axis=1 means computing it in time-series way.
    """
    data = raw_data.replace([np.inf, -np.inf], np.nan)
    if axis == 1:
        return min_max_scaler(raw_data.T, axis=0).T

    maxabs = data.abs().max(axis=1)
    scale = 1/maxabs
    result = data.multiply(scale, axis=0)
    if subtract_mean:
        result = result.subtract(result.mean(axis=1), axis=0)
    return result


def call_market_data_alpha_fn(fn, market_data):
    return fn(opens=market_data['px_open'].fillna(method='ffill', limit=2),
              highs=market_data['px_high'].fillna(method='ffill', limit=2),
              lows=market_data['px_low'].fillna(method='ffill', limit=2),
              closes=market_data['px_last'].fillna(method='ffill', limit=2),
              returns=market_data['returns'].fillna(0),
              volumes=market_data['px_volume'].fillna(method='ffill', limit=2).fillna(0))


def create_market_alpha_scores(alpha_fn, market_data):
    validity = validity_table(market_data)
    results = call_market_data_alpha_fn(alpha_fn, market_data)
    results = results[validity].fillna(limit=2, method='ffill')
    return results

def long_short_deciles(scores, long_deciles={9, 8, 7, 6, 5}, short_deciles={0, 1, 2, 3, 4}):
    deciles = (scores.rank(axis=1, pct=True) * 9)
    return max_abs_scaler(deciles.fillna(method='ffill', limit=3)).fillna(method='ffill', limit=2)


def compute_market_alpha_scores(alpha_fn, market_data,
                              long_deciles={9, 8, 7, 6, 5},
                              short_deciles={0, 1, 2, 3, 4},
                              minimum_securities=12):
    scores = create_market_alpha_scores(alpha_fn, market_data)
    return compute_long_short_score_portfolio(scores, market_data)

def compute_long_short_score_portfolio(scores, market_data):
    score_portfolio = long_short_deciles(scores)
    score_portfolio = score_portfolio[validity_table(market_data)]
    return score_portfolio.dropna(how='all')


@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='v1')
def get_market_data(universe):
    universe_tickers = APEX_UNIVERSES[universe]
    data = apex__adjusted_market_data(*universe_tickers)
    data = pd.concat(data.values(), axis=1)
    data = data.swaplevel(axis=1)
    data = data.sort_index(axis=1)
    data = data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'})
    data['returns'] = data['returns'] / 100
    return data

@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='security.v1')
def get_market_data_security(security):
    data = apex__adjusted_market_data(security)[security][security]
    data = data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'})
    data['returns'] = data['returns'] / 100
    return data

def compute_alpha_results(alpha_name, alpha_score, data=None, universe='AMNA'):
    print('[INFO] Getting universe data')
    market_data = get_market_data(universe)
    print('[INFO] Building portfolio')
    weights = construct_alpha_portfolio_from_signal(alpha_score, market_data)
    print('[INFO] Generating runs')
    return generate_experiment_runs(universe, alpha_name, weights, market_data['returns'])

@UNIVERSE_DATA_CACHING.cache_on_arguments()
def bloomberg_field_factor_data(field_name, universe):
    universe = APEX_UNIVERSES[universe]
    bbg = ApexBloomberg()
    result = bbg.history(universe, field_name)
    result.columns = result.columns.droplevel(1)
    return result

def compute_bloomberg_field_score(field_name, universe, cutoff=0.0):
    data = bloomberg_field_factor_data(field_name, universe)
    return rank_signal_from_bbg(data, cutoff=cutoff)


def apex__market_alpha_experiment_pipeline(alpha_fn, universe):
    market_data = get_market_data(universe)
    score = compute_market_alpha_scores(alpha_fn, market_data)
    alpha_name = ' '.join(alpha_fn.__name__.split('_'))
    return compute_alpha_results(alpha_name.capitalize(), score, universe=universe)

def rank_signal_from_bbg(result, cutoff=0.75):
    result = result.rank(axis=1)
    max_rank = result.max(axis=1)
    result = result.divide(max_rank, axis=0)
    result = result * 2 - 1
    result[result.abs() < cutoff] = 0
    return result.dropna(how='all')

def apex__bbg_experiment_pipeline(field_name, universe):
    score = compute_bloomberg_field_score(field_name, universe)
    alpha_name = ' '.join(field_name.split('_'))
    return compute_alpha_results(alpha_name.capitalize(), score, universe=universe)
