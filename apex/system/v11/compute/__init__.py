import threading
import typing
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import bottleneck as bn
import empyrical as ep
import inflection
import numba as nb
import numpy as np
import pandas as pd
import pyfolio as pf
from dask.distributed import as_completed
from dataclasses import dataclass, field
from toolz import curry, dissoc, partition_all

import xarray as xr
from apex.system.v11.backtest import (apex__backtest_portfolio_weights,
                                      apex__compute_strategy_returns,
                                      apex__compute_strategy_trades,
                                      apex__compute_turnover_with_trades,
                                      apex__performance_stats)
from apex.system.v11.data import (ApexBlackboard, ApexTemporaryBlackboard,
                                  ApexUniverseBlackboard, ds_to_df,
                                  apex__build_universe_dataset, apex__dataset)
from apex.system.v11.universes.midstream import \
    create_strategy_subuniverses as apex_default__create_strategy_subuniverses
from apex.toolz.dask import ApexDaskClient
from apex.toolz.deco import lazyproperty, retry
from apex.toolz.arctic import ArcticApex
import time
from functools import wraps

from .market_alpha import MARKET_ALPHAS, MARKET_ALPHAS_BY_FAMILY

DASK_KWARGS = {
    'key',
    'workers',
    'retries',
    'resources',
    'priority',
    'allow_other_workers',
    'fifo_timeout',
    'actor',
    'actors',
    'pure',
}

class ApexQueuedDaskPool:
    """
    Description:

    I want to submit tasks to this pool and when
    max_running == unfinished tasks, we want to put the tasks
    in queue and submit them to the cluster as tasks are
    finished.
    """
    def __init__(self, max_running=5):
        self.max_running = max_running
        self.remaining_tasks = 0
        self.running_tasks = 0

        self.cluster = ApexDaskClient()
        self.id = uuid.uuid4().hex

        self._queue = deque()
        self.futures = {}
        self._done = []
        self.blackboard = ApexTemporaryBlackboard(name=self.id, output_type='pd')

        self.mutex = threading.Lock()

        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)

        self.terminated = False

        self._watcher = Thread(target=apex__watch_futures_queue, args=(self,))
        self._watcher.start()


    def __getattr__(self, name):
        return getattr(self.cluster, name)

    def submit(self, *args, **kwargs):
        with self.not_empty:
            task = {'id': uuid.uuid4().hex,
                    'fn': self.cluster.submit,
                    'args': args,
                    'kwargs': kwargs}
            self.remaining_tasks += 1
            self._queue.append(task)
            self.not_empty.notify_all()
        return task['id']

    def map(self, fn, *iterables, **kwargs):
        tasks_args = zip(*iterables)
        result = []
        for ix, task_args in enumerate(tasks_args):
            result.append(self.submit(fn, *task_args, **kwargs))
        return result

    def gather(self, task_ids=None):
        """
        Gather on this has a different purpose - it gets the finished results.
        """
        with self.mutex:
            ids = self._done
            if not task_ids:
                task_ids = ids
            return {x: self.blackboard[x] for x in ids if x in task_ids}

    def scatter(self, *args, **kwargs):
        return self.cluster.scatter(*args, **kwargs)

    def __del__(self):
        while self.remaining_tasks > 0 or self.running_tasks > 0:
            time.sleep(0.25)

        self.terminated = True
        self._watcher.join()
        lib_name = self.blackboard.library_name
        self.blackboard.close()
        arc = ArcticApex()
        if lib_name in arc.libraries:
            arc.session.delete_library(lib_name)


def apex__task_finished_callback(self, future):
    with self.mutex:
        task_info = self.futures[future]
        result = future.result()
        self.blackboard.write(task_info['id'], result)
        self._done.append(task_info['id'])
        self.running_tasks -= 1
        self.not_full.notify_all()

def apex__watch_futures_queue(pool, timer=0.25):
    """
    Watches the queue.
    """
    q = pool._queue
    task_callback = curry(apex__task_finished_callback, pool)
    while not pool.terminated:
        with pool.not_empty:
            while not q:
                pool.not_empty.wait()

        with pool.not_full:
            while pool.running_tasks >= pool.max_running:
                pool.not_full.wait()

        with pool.mutex:
            task = q.pop()
            fn = task['fn']
            args = task['args']
            kwargs = task['kwargs']
            fut = fn(*args, **kwargs)
            pool.futures[fut] = task

            fut.add_done_callback(task_callback)
            pool.running_tasks += 1
            pool.remaining_tasks -= 1


TEMPORARY_SAVE_FOLDER = Path('/apex.data/apex.portfolios/tmp/')
SAVE_FOLDER = Path('/apex.data/apex.portfolios/apex/')


TRANSFORMS = {
    'identity': lambda x: x,
    'delayed': lambda x: x.shift(1),

    'ewm1': lambda x: x.ewm(halflife=1).mean(),
    'ewm3': lambda x: x.ewm(halflife=3).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm50': lambda x: x.ewm(halflife=50).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
}

DIFFED_TRANSFORMS = {
    'ewm1_diffed': lambda x: x.ewm(halflife=1).mean().diff(),
    'ewm3_diffed': lambda x: x.ewm(halflife=3).mean().diff(),
    'ewm5_diffed': lambda x: x.ewm(halflife=5).mean().diff(),
    'ewm10_diffed': lambda x: x.ewm(halflife=10).mean().diff(),
    'ewm20_diffed': lambda x: x.ewm(halflife=20).mean().diff(),
    'ewm50_diffed': lambda x: x.ewm(halflife=50).mean().diff(),
    'ewm100_diffed': lambda x: x.ewm(halflife=100).mean().diff(),
}

def apex__save_portfolio_data(portfolio, folder, prefix=None):
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    filename = uuid.uuid4().hex + '.pq.gz'
    if prefix:
        filename = prefix + '__' + filename
    filename = folder / filename
    portfolio.to_parquet(filename, compression='gzip')
    return filename

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


def base_keygen(prefix=None):
    if prefix:
        return ':'.join([prefix, uuid.uuid4().hex])
    else:
        return uuid.uuid4().hex

def apex__transform_pipeline(raw_alpha, availability, save_prefix=None, save_loc=None, blackboard=None):
    """
    1. Transform
    2. Compute portfolio construction (inv vol weighted, vol weighetd, etc)
    """
    base_alpha = raw_alpha.copy()
    results = []
    keygen = curry(base_keygen, prefix=save_prefix)
    save_portfolio_data = curry(apex__save_portfolio_data, prefix=save_prefix)

    for transform_name, transform_fn in TRANSFORMS.items():
        transform_name += '_pos'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    for transform_name, transform_fn in DIFFED_TRANSFORMS.items():
        transform_name += '_pos'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    base_alpha = -raw_alpha.copy()
    for transform_name, transform_fn in TRANSFORMS.items():
        transform_name += '_neg'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    for transform_name, transform_fn in DIFFED_TRANSFORMS.items():
        transform_name += '_neg'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    if save_loc is not None:
        return True
    else:
        return results


def apex__apply_transforms_to_alpha(raw_alpha, availability, save_prefix=None, save_loc=None, blackboard=None):
    """
    1. Transform
    2. Compute portfolio construction (inv vol weighted, vol weighetd, etc)
    """
    base_alpha = raw_alpha.copy()
    results = []
    keygen = curry(base_keygen, prefix=save_prefix)
    save_portfolio_data = curry(apex__save_portfolio_data, prefix=save_prefix)

    for transform_name, transform_fn in TRANSFORMS.items():
        transform_name += '_pos'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    for transform_name, transform_fn in DIFFED_TRANSFORMS.items():
        transform_name += '_pos'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    base_alpha = -raw_alpha.copy()
    for transform_name, transform_fn in TRANSFORMS.items():
        transform_name += '_neg'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    for transform_name, transform_fn in DIFFED_TRANSFORMS.items():
        transform_name += '_neg'
        tdata = transform_fn(base_alpha)[availability]
        normalized_tdata = tdata.rank(axis=1)

        if save_loc is not None:
            save_portfolio_data(normalized_tdata, save_loc)
        elif blackboard is not None:
            key = keygen() + ':' + transform_name
            blackboard.write(key, normalized_tdata)
            results.append(key)
        else:
            results.append(normalized_tdata)

    if save_loc is not None:
        return True
    else:
        return results

@dataclass
class EqualWeightedPortfolioSelector:
    universe_bb: ApexUniverseBlackboard
    n_portfolios: int = field(default=10)
    start_date: str = field(default='2000-01-01')
    transaction_costs: int = field(default=15)
    portfolios: dict = field(default_factory=dict)
    backtests: dict = field(default_factory=dict)
    selection_metrics: list = field(default_factory=lambda: ['cumulative_returns', 'sortino_ratio'])

    @lazyproperty
    def market_data(self):
        return self.universe_bb.market_data.loc[self.start_date:]

    def try_adding(self, portfolio):
        portfolio = portfolio.loc[self.start_date:]
        port_id = uuid.uuid4().hex
        self.portfolios[port_id] = portfolio
        self.backtests[port_id] = apex__backtest_portfolio_weights(self.market_data, portfolio, transaction_costs=self.transaction_costs)
        if len(self.portfolios) < self.n_portfolios:
            return True

        portfolio_metrics = {x: self.backtests[x]['stats'][self.selection_metrics] for x in self.portfolios}
        selection = pd.DataFrame(portfolio_metrics).T.rank(axis=0, ascending=True).sum(axis=1).sort_values(ascending=False)
        selection = selection.index[:self.n_portfolios]
        self.portfolios = {x: self.portfolios[x] for x in selection}
        self.backtests = {x: self.backtests[x] for x in selection}
        return True

    @property
    def portfolio(self):
        portfolio = pd.concat(self.portfolios, axis=1).fillna(0)
        portfolio = portfolio.groupby(axis=1, level=1).sum()
        portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)
        return portfolio


def apex__alpha_selection_pipeline(
    universe_bb: ApexUniverseBlackboard,
    strategy_bb: ApexBlackboard,
    alpha_keys: list,
    strategy_subuniverse: str,
    long_short=False,
    start_date='2000-01-01',
    selection_metrics=['cumulative_returns', 'sortino_ratio'],
    cutoffs=[0, 0.25, 0.5],
    vol_metric='ewm_vol_20d_hl',
    n_selection=10,
    transaction_costs=15):
    """
    1. Here we need to create cutoff portfolios and
    """
    availability = strategy_bb[strategy_subuniverse]
    availability = availability[availability].astype(float)

    vol = universe_bb.get(vol_metric)

    portfolio_selector = EqualWeightedPortfolioSelector(universe_bb=universe_bb,
                                                        n_portfolios=n_selection,
                                                        transaction_costs=transaction_costs,
                                                        start_date=start_date,
                                                        selection_metrics=selection_metrics)
    for alpha_name in alpha_keys:
        raw_alpha = strategy_bb.get(alpha_name, output_type='pd')
        raw_alpha = (raw_alpha * availability).rank(axis=1)
        if long_short:
            raw_alpha = raw_alpha * 2 - 1
        raw_alpha = raw_alpha.divide(raw_alpha.max(axis=1), axis=0)

        for cutoff in cutoffs:
            cutoff_alpha = raw_alpha[np.abs(raw_alpha) >= cutoff].fillna(0)
            cutoff_alpha = cutoff_alpha.divide(cutoff_alpha.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_alpha)

            ca_inv_vol = cutoff_alpha/vol
            cutoff_port = ca_inv_vol.divide(ca_inv_vol.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_port)

            ca_vol = cutoff_alpha * vol
            cutoff_port = ca_vol.divide(ca_vol.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_port)

    return portfolio_selector.portfolio



def apex__select_best_alpha_transforms(
    universe_bb: ApexUniverseBlackboard,
    strategy_bb: ApexBlackboard,
    alpha_keys: list,
    strategy_subuniverse: str,
    long_short=False,
    start_date='2000-01-01',
    selection_metrics=['cumulative_returns', 'sortino_ratio'],
    cutoffs=[0, 0.25, 0.5],
    vol_metric='ewm_vol_20d_hl',
    n_selection=10,
    transaction_costs=15):
    """
    1. Here we need to create cutoff portfolios and
    """
    availability = strategy_bb[strategy_subuniverse]
    availability = availability[availability].astype(float)

    vol = universe_bb.get(vol_metric)

    portfolio_selector = EqualWeightedPortfolioSelector(universe_bb=universe_bb,
                                                        n_portfolios=n_selection,
                                                        transaction_costs=transaction_costs,
                                                        start_date=start_date,
                                                        selection_metrics=selection_metrics)
    for alpha_name in alpha_keys:
        raw_alpha = strategy_bb.get(alpha_name, output_type='pd')
        raw_alpha = (raw_alpha * availability).rank(axis=1)
        if long_short:
            raw_alpha = raw_alpha * 2 - 1
        raw_alpha = raw_alpha.divide(raw_alpha.max(axis=1), axis=0)

        for cutoff in cutoffs:
            cutoff_alpha = raw_alpha[np.abs(raw_alpha) >= cutoff].fillna(0)
            cutoff_alpha = cutoff_alpha.divide(cutoff_alpha.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_alpha)

            ca_inv_vol = cutoff_alpha/vol
            cutoff_port = ca_inv_vol.divide(ca_inv_vol.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_port)

            ca_vol = cutoff_alpha * vol
            cutoff_port = ca_vol.divide(ca_vol.abs().sum(axis=1), axis=0)
            portfolio_selector.try_adding(cutoff_port)

    return portfolio_selector.portfolio

@retry(NotImplementedError, tries=2)
def apex__simple_market_alpha_wrapper(fn):
    @wraps(fn)
    def wrapped(market_data):
        try:
            if isinstance(market_data, xr.Dataset):
                market_data = ds_to_df(market_data[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns']])
            else:
                market_data = market_data[['px_last', 'px_open', 'px_high', 'px_low', 'px_volume', 'returns']]

            return fn(opens=market_data['px_open'],
                    highs=market_data['px_high'],
                    lows=market_data['px_low'],
                    closes=market_data['px_last'],
                    returns=market_data['returns'],
                    volumes=market_data['px_volume'])
        except:
            import traceback
            traceback.print_exc()
            raise NotImplementedError( f"When the alpha function fails... name: {fn.__name__}" )

    return wrapped


@retry(Exception, tries=2)
def apex__alpha_selection_portfolio(universe_bb, strategy_bb, subuniverse_name, alpha_name, raw_alpha, base_universe='universe:base'):
    """
    This creates the transformed pipeline and selects top portfolios
    """
    try:
        availability = strategy_bb[subuniverse_name]
        print( f'Computing transform pipeline for {subuniverse_name}:{alpha_name}' )

        computed_transforms_re = f'transforms:{subuniverse_name}:{alpha_name}.*'
        computed = strategy_bb.cache.list_symbols(regex=computed_transforms_re)
        expected_len = (len(TRANSFORMS) + len(DIFFED_TRANSFORMS)) * 2
        if len(computed) < expected_len:
            for var in computed:
                strategy_bb.cache.delete(var)
            transform_pipe_results = apex__transform_pipeline(raw_alpha,
                                                            availability,
                                                            save_prefix=f'transforms:{subuniverse_name}:{alpha_name}',
                                                            blackboard=strategy_bb)
        else:
            transform_pipe_results = computed
        print(f'Computing portfolio selection for {subuniverse_name}:{alpha_name}')
        portfolio = apex__alpha_selection_pipeline(universe_bb,
                                                strategy_bb,
                                                transform_pipe_results,
                                                subuniverse_name)

        alpha_save_name = f'apex:selected_portfolio:{subuniverse_name}:{alpha_name}'
        print( f'Saving portfolio selection for {subuniverse_name}:{alpha_name} @ {alpha_save_name}' )
        backtest = apex__backtest_portfolio_weights(universe_bb.market_data, portfolio, transaction_costs=15)
        strategy_bb.write(alpha_save_name, {'portfolio': portfolio, 'backtest': backtest}, metadata={
            'subuniverse_name': subuniverse_name,
            'alpha_name': alpha_name
        })

        # Cleanup
        computed = strategy_bb.cache.list_symbols(regex=computed_transforms_re)
        for var in computed:
            strategy_bb.cache.delete(var)

        return alpha_save_name
    except:
        import traceback
        traceback.print_exc()
        raise

def apex__alpha_pipeline_generator(strategy, subuniverse_name, market_alpha=None, market_alpha_family=None, smart_beta=None):
    """
    Computes alpha pipeline for a strategy
    """
    universe_bb = strategy.universe_bb
    strategy_bb = strategy.blackboard

    availability = strategy_bb[subuniverse_name]
    market_data = universe_bb.market_data

    if market_alpha:
        market_alphas = market_alpha
        if isinstance(market_alpha, str):
            market_alphas = [market_alpha]
        for market_alpha_name in market_alphas:
            alpha_fn = apex__simple_market_alpha_wrapper(MARKET_ALPHAS[market_alpha_name])
            raw_alpha = alpha_fn(market_data)[availability]
            yield (market_alpha, raw_alpha)

    if market_alpha_family:
        for alpha_name, alpha_fn in MARKET_ALPHAS_BY_FAMILY[market_alpha_family].items():
            alpha_fn = apex__simple_market_alpha_wrapper(alpha_fn)
            raw_alpha = alpha_fn(market_data)[availability]
            yield (alpha_name, raw_alpha)

    if smart_beta:
        smart_betas = smart_beta
        if isinstance(smart_beta, str):
            smart_betas = [smart_beta]
        for smart_beta_name in smart_betas:
            yield (smart_beta_name, universe_bb[smart_beta_name][availability])


def apex__compute_alpha_pipeline(strategy, subuniverse_name, pipeline, gather=True, distributed=False):
    if distributed:
        pool = ApexDaskClient()
        futures = {}
        for alpha_name, alpha_data in pipeline:
            fut = pool.submit(apex__alpha_selection_portfolio, strategy.universe_bb, strategy.blackboard, subuniverse_name, alpha_name, alpha_data)
            futures[alpha_name] = fut
        result = pool.gather(futures)
        pool.close()
    else:
        result = {}
        for alpha_name, alpha_data in pipeline:
            result[alpha_name] = apex__alpha_selection_portfolio(strategy.universe_bb, strategy.blackboard, subuniverse_name, alpha_name, alpha_data)
    return result



def apex__compute_alpha_selection_all_subuniverses(strategy, pipeline, base_universe='universe:base'):
    """
    Computes alpha selection for a particular pipeline
    """
    subuniverses = strategy.blackboard['subuniverses']
    subuniverse_result = defaultdict(dict)
    current_variables = set(strategy.blackboard.variables)

    pool = ApexDaskClient()
    availability = strategy.blackboard[base_universe]
    alpha_results = {}
    for alpha_name, alpha_data in pipeline:
        # Can be parallel
        future = pool.submit(apex__apply_transforms_to_alpha, alpha_data, availability,
                                 save_prefix=f'transforms:{base_universe}:{alpha_name}',
                                 blackboard=strategy_bb)
        alpha_results[future] = (alpha_name, alpha_data)

    for alpha_fut in as_completed(alpha_results):
        alpha_transforms = alpha_fut.result()

        for subuniverse_name in subuniverses:
            # Can be parallel
            res = pool.submit(apex__select_best_alpha_transforms, strategy.universe_bb, strategy.blackboard, subuniverse_name, alpha_name, future)
            subuniverse_result[f_sub_name][f_a_name] = res

    subuniverse_result = pool.gather(subuniverse_result)
    return subuniverse_result

def apex__compute_alpha_selection_all_subuniverses_distributed(strategy, pipeline, base_universe='universe:base', distributed=True, max_tasks=2):
    """
    Computes alpha selection for a particular pipeline
    """
    if not distributed:
        raise NotImplementedError("Still going to implement")
    subuniverses = strategy.blackboard['subuniverses']
    subuniverse_result = defaultdict(dict)
    current_variables = set(strategy.blackboard.variables)

    queued_pool = ApexQueuedDaskPool(max_running=max_tasks)
    task_id_to_input = {}
    for alpha_name, alpha_data in pipeline:
        for subuniverse_name in subuniverses:
            selected_name = f'apex:selected_portfolio:{subuniverse_name}:{alpha_name}'
            if selected_name in current_variables:
                continue

            task_id = queued_pool.submit(apex__alpha_selection_portfolio, strategy.universe_bb, strategy.blackboard, subuniverse_name, alpha_name, alpha_data)
            task_id_to_input[task_id] = (subuniverse_name, alpha_name)


    remaining = set(list(task_id_to_input.keys()))
    #print('Remaining tasks selection:', '\n', sorted(remaining))
    while remaining:
        gather_results = queued_pool.gather(task_ids=remaining)
        for task_id, task_result in gather_results.items():
            remaining.remove(task_id)
            f_sub_name, f_a_name = task_id_to_input[task_id]
            #print('[INFO] Got new subuniverse task completed:', task_id, f_sub_name, f_a_name)
            subuniverse_result[f_sub_name][f_a_name] = task_result
        time.sleep(5)
    queued_pool.blackboard.clear_cache()
    queued_pool.blackboard.close()
    return subuniverse_result


def apex__strategy_blackboard_initialization(
        universe_name,
        strategy_name,
        tickers,
        create_universe=False,
        subuniverse_fn=None,
        temporary=True,
        cleanup_blackboard=True,
        subuniverse_fn_vars=('default_availability', 'cur_mkt_cap')):

    if create_universe:
        universe_created = apex__build_universe_dataset(universe_name, tickers)
        if not universe_created:
            raise NotImplementedError("Universe creation failed")

    universe_bb = ApexUniverseBlackboard(name=universe_name, update='if_empty', output_type='xr')
    if not temporary:
        strategy_bb = ApexBlackboard(name=strategy_name, output_type='xr')
    else:
        strategy_bb = ApexTemporaryBlackboard(name=strategy_name, output_type='xr')

    if cleanup_blackboard:
        strategy_bb.clear_cache()

    if subuniverse_fn:
        subuniverses = subuniverse_fn(universe_bb.get(subuniverse_fn_vars), strategy_bb, tickers)

    strategy_bb['tickers'] = tickers
    # Because it will be needed
    universe_bb.output_type = 'pd'
    strategy_bb.output_type = 'pd'
    return strategy_bb, universe_bb, subuniverses


@dataclass
class ApexStrategy:
    """
    ApexStrategy is a helper class to compute everything in a strategy.
    """
    name: str
    universe_bb: ApexUniverseBlackboard
    blackboard: ApexBlackboard
    temporary: bool

    @classmethod
    def create(cls, strategy_name,
               universe_name, tickers,
               create_universe=False,
               temporary=False,
               cleanup_blackboard=True,
               subuniverse_fn=None,
               subuniverse_fn_vars=('default_availability', 'cur_mkt_cap')):

        strategy_bb, universe_bb, subuniverses = apex__strategy_blackboard_initialization(
            universe_name,
            strategy_name,
            tickers,
            temporary=temporary,
            cleanup_blackboard=cleanup_blackboard,
            create_universe=create_universe,
            subuniverse_fn=subuniverse_fn,
            subuniverse_fn_vars=subuniverse_fn_vars
        )

        strategy_bb.write('subuniverses', subuniverses)
        strategy = ApexStrategy(name=strategy_name,
            temporary=temporary,
            universe_bb=universe_bb,
            blackboard=strategy_bb
        )
        return strategy


def apex__compute_subuniverse_selection(strategy, subuniverse_name,
                                        selection_pctile_cutoff=0.2,
                                        # selection_start='2010-01-01', for the futures
                                        # selection_end='2018-01-01',
                                        selection_fields=['cumulative_returns', 'omega_ratio']):
    strategy_bb = strategy.blackboard
    universe_bb = strategy.universe_bb

    results = strategy_bb.get_matching(f'apex:selected_portfolio:{subuniverse_name}')
    num_strategies = len(results)
    strat_stats = {x: results[x]['backtest']['stats'] for x in results}
    strat_selection = pd.concat(strat_stats, axis=1).T
    strat_selection = strat_selection[selection_fields]
    strat_selection = strat_selection.rank(pct=True, axis=0).sum(axis=1).rank(pct=True)
    strat_selection = strat_selection[strat_selection > 1 - selection_pctile_cutoff]
    # Selecting best 10 for the universe
    selected = strat_selection.index.tolist()

    selected_portfolios = {x: results[x]['portfolio'] for x in selected}
    selected_returns = {x: results[x]['backtest']['returns'] for x in selected}
    selected_stats = {x: results[x]['backtest']['stats'] for x in selected}
    return {
        'portfolios': selected_portfolios,
        'returns': selected_returns,
        'stats': selected_stats
    }

def apex__combine_subuniverse_selection(strategy, subuniverse_selection):
    portfolio = pd.concat(subuniverse_selection['portfolios'], axis=1)
    portfolio = portfolio.groupby(axis=1, level=1).sum()
    portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)
    return portfolio

def apex__compute_subuniverse_result(strategy, subuniverse_name):
    subuniverse_selection = apex__compute_subuniverse_selection(strategy, subuniverse_name)
    portfolio = apex__combine_subuniverse_selection(strategy, subuniverse_selection)
    backtest = apex__backtest_portfolio_weights(strategy.universe_bb.market_data, portfolio, transaction_costs=15)
    return {
        'portfolio': portfolio,
        'subportfolios': subuniverse_selection,
        'backtest': backtest
    }



@nb.njit
def apex__nb_compute_low_turnover_weights(returns, availability, portfolio, blend_multiplier):
    """
    Blend every day at rate = blend multiplier / 2
    """
    new_port = np.zeros_like(portfolio)
    rets = returns + 1
    initialized = False
    for day in range(1, len(portfolio)):
        if not initialized:
            if np.nansum(np.abs(portfolio[day - 1])) < 1e-5:
                continue
            else:
                new_port[day - 1] = portfolio[day - 1]/np.nansum(portfolio[day - 1])
                initialized = True

        curr_pos = new_port[day - 1] * rets[day]
        curr_val = np.nansum(curr_pos)
        curr_wt = curr_pos/curr_val
        new_pos = curr_wt + (portfolio[day] - curr_wt) * blend_multiplier[day]
        new_pos[~availability[day]] = 0
        port_val = np.nansum(np.abs(new_pos))
        if port_val <= 1e-10:
            new_pos = new_port[day-1]
            port_val = np.nansum(np.abs(new_pos))
        new_port[day] = new_pos/port_val
    return new_port

def apex__compute_drifting_turnover_constrained_portfolio(strategy, tracking_portfolio, turnover_period,
                                                          maximum_gap_close=1,
                                                          reference_quantile=0.5,
                                                          availability='basic_availability',
                                                          variance_halflife=10):

    if 'cash' in tracking_portfolio.columns:
        tracking_portfolio = tracking_portfolio.drop(columns=['cash'])

    availability = strategy.universe_bb[availability]
    start_dt = tracking_portfolio.index[0] - pd.DateOffset(years=1)
    returns = strategy.universe_bb['returns'].loc[start_dt:].fillna(0)

    var = returns.ewm(halflife=variance_halflife).var()[availability]
    var = var.median(axis=1).dropna()
    var_exp_median = var.expanding().quantile(reference_quantile)

    # compute number of median days equivalent to rebalancing days
    turns_per_variance_day = 1/turnover_period
    blend_multiplier = var/var_exp_median * turns_per_variance_day

    # Now let's set up everything for numpy


    np_returns = strategy.universe_bb['returns'].reindex(tracking_portfolio.index)[tracking_portfolio.columns].fillna(0).values
    np_availability = availability.reindex(tracking_portfolio.index)[tracking_portfolio.columns].values
    np_blend_multiplier = np.minimum(blend_multiplier.reindex(tracking_portfolio.index), maximum_gap_close).fillna(0).values
    np_port = tracking_portfolio[availability].fillna(0).values
    result = apex__nb_compute_low_turnover_weights(np_returns, np_availability, np_port, np_blend_multiplier)
    result = pd.DataFrame(result, index=tracking_portfolio.index, columns=tracking_portfolio.columns)[availability]
    return result


def apex__market_alpha_family_pipeline_generator(strategy, subuniverse_name, market_alpha_family, max_tasks=2):
    """
    Computes alpha pipeline for a strategy
    """
    universe_bb = strategy.universe_bb
    strategy_bb = strategy.blackboard

    blackboard = strategy.blackboard
    current_variables = set(blackboard.variables)

    availability = strategy_bb[subuniverse_name]
    market_data = universe_bb.market_data

    queued_pool = ApexQueuedDaskPool(max_running=max_tasks)
    market_data_sc = queued_pool.scatter(market_data)
    alpha_name_by_task_id = {}
    for alpha_name, alpha_fn in MARKET_ALPHAS_BY_FAMILY[market_alpha_family].items():
        selected_name = f'apex:selected_portfolio:{subuniverse_name}:{alpha_name}'
        if selected_name in current_variables:
            continue
        alpha_fn = apex__simple_market_alpha_wrapper(alpha_fn)
        alpha_name_by_task_id[queued_pool.submit(alpha_fn, market_data_sc)] = alpha_name

    remaining = set(list(alpha_name_by_task_id.keys()))
    #print('Remaining raw alpha tasks:', '\n', sorted(remaining))
    while remaining:
        gather_results = queued_pool.gather(task_ids=remaining)
        for task_id, raw_alpha in gather_results.items():
            remaining.remove(task_id)
            alpha_data = raw_alpha[availability]
            alpha_name = alpha_name_by_task_id[task_id]
            #print('[INFO] Got new raw alpha task completed:', task_id, alpha_name)
            yield (alpha_name, alpha_data)

    #print('Done with all raw alpha tasks')
    queued_pool.blackboard.clear_cache()
    queued_pool.blackboard.close()






















def apex__select_best_alpha_transforms__all_subuniverses(
        strategy: ApexStrategy,
        alpha_keys: list,
        base_alpha_name: str,
        long_short=False,

        ## Strategy build params
        # Cuttofs for alpha & vol metric for volatility weighted portfolio
        cutoffs=[0, 0.25, 0.5],
        vol_metric='ewm_vol_20d_hl',

        # Blending time periods for default blend
        blend=True,
        blend_turnover_period=1,
        blend_reference_quantile=0.99,
        blend_variance_halflife=1,

        # Selection metrics
        selection_metrics=['calmar_ratio', 'omega_ratio'],
        n_selection=15,

        # Weight cap
        ## Unnecessary because the portfolio should be fairly diversified.
        cap_weights_lo=False,
        upper_weight_cap=0.1,
        lower_weight_cap=0.0025,

        # Backtest params
        start_date='2000-01-01',
        transaction_costs=15,

        # Save info
        save_suffix=None):
    long_only = not long_short
    universe_bb = strategy.universe_bb
    strategy_bb = strategy.blackboard

    universe_bb.output_type = 'xr'
    strategy_bb.output_type = 'xr'
    # datasets
    base_market_data = universe_bb.market_data

    #ds = ds / np.abs(ds).sum('ticker')
    var = base_market_data.returns.to_pandas().ewm(halflife=blend_variance_halflife).var()
    alpha_data = strategy_bb[alpha_keys]

    base_volatility = universe_bb[vol_metric]
    # Cleanup
    subuniverses = strategy_bb['subuniverses']
    results = {}
    for subuniverse in subuniverses:
        availability = strategy_bb[subuniverse]
        ds = alpha_data.where(availability).dropna('ticker', how='all').rank('ticker').dropna('time', how='all')

        availability = availability.loc[ds.time]
        availability = availability.where(availability).dropna('ticker', how='all').fillna(False).astype(bool)
        availability_pd = availability.to_pandas()

        market_data = base_market_data.loc[{'time': ds.time}].where(availability).dropna('ticker', how='all')

        volatility = base_volatility.loc[ds.time]
        volatility = volatility.where(availability).dropna('ticker', how='all').ffill('time')

        # Blending
        var = var.reindex(availability_pd.index)[availability_pd].dropna(how='all', axis=1)
        var = var.median(axis=1).fillna(method='ffill')
        var_exp_median = var.expanding().quantile(blend_reference_quantile)

        turns_per_variance_day = 1/blend_turnover_period
        blend_multiplier = var/var_exp_median * turns_per_variance_day

        # Normalizing alpha data
        ds = ds/ds.max('ticker')
        if long_short:
            ds = ds * 2 - 1

        portfolio_selector = EqualWeightedPortfolioSelector(market_data=ds_to_df(market_data),
                                                            n_portfolios=n_selection,
                                                            transaction_costs=transaction_costs,
                                                            start_date=start_date,
                                                            selection_metrics=selection_metrics)
        for cutoff in cutoffs:
            cutoff_ds = ds.where(np.abs(ds) > cutoff)
            cutoff_port = cutoff_ds / np.abs(cutoff_ds).sum('ticker')
            for key in alpha_keys:
                alpha_cutoff_port = cutoff_port[key].to_pandas()
                if cap_weights_lo and long_only:
                    alpha_cutoff_port = apex__cap_weights(alpha_cutoff_port, upper_weight_cap, lower_weight_cap)

                if blend:
                    np__cutoff_port = apex__nb_compute_low_turnover_weights(market_data.returns.values, availability.values, alpha_cutoff_port.values, blend_multiplier.values)
                    alpha_cutoff_port = pd.DataFrame(np__cutoff_port, index=alpha_cutoff_port.index, columns=alpha_cutoff_port.columns)

                alpha_cutoff_port = alpha_cutoff_port[availability_pd]
                alpha_cutoff_port = alpha_cutoff_port.divide(alpha_cutoff_port.abs().sum(axis=1), axis=0)
                portfolio_selector.try_adding(alpha_cutoff_port, name=f'{key}.base.{cutoff}')

            cutoff_ds_inv_vol = cutoff_ds / volatility
            cutoff_port = cutoff_ds_inv_vol / np.abs(cutoff_ds_inv_vol).sum('ticker')
            for key in alpha_keys:
                alpha_cutoff_port = cutoff_port[key].to_pandas()
                if cap_weights_lo and long_only:
                    alpha_cutoff_port = apex__cap_weights(alpha_cutoff_port, upper_weight_cap, lower_weight_cap)

                if blend:
                    np__cutoff_port = apex__nb_compute_low_turnover_weights(market_data.returns.values, availability.values, alpha_cutoff_port.values, blend_multiplier.values)
                    alpha_cutoff_port = pd.DataFrame(np__cutoff_port, index=alpha_cutoff_port.index, columns=alpha_cutoff_port.columns)

                alpha_cutoff_port = alpha_cutoff_port[availability_pd]
                alpha_cutoff_port = alpha_cutoff_port.divide(alpha_cutoff_port.abs().sum(axis=1), axis=0)
                portfolio_selector.try_adding(alpha_cutoff_port, name=f'{key}.inv_vol.{cutoff}')

            cutoff_ds_vol = cutoff_ds * volatility
            cutoff_port = cutoff_ds_vol / np.abs(cutoff_ds_vol).sum('ticker')
            for key in alpha_keys:
                alpha_cutoff_port = cutoff_port[key].to_pandas()
                if cap_weights_lo and long_only:
                    alpha_cutoff_port = apex__cap_weights(alpha_cutoff_port, upper_weight_cap, lower_weight_cap)

                if blend:
                    np__cutoff_port = apex__nb_compute_low_turnover_weights(market_data.returns.values, availability.values, alpha_cutoff_port.values, blend_multiplier.values)
                    alpha_cutoff_port = pd.DataFrame(np__cutoff_port, index=alpha_cutoff_port.index, columns=alpha_cutoff_port.columns)

                alpha_cutoff_port = alpha_cutoff_port[availability_pd]
                alpha_cutoff_port = alpha_cutoff_port.divide(alpha_cutoff_port.abs().sum(axis=1), axis=0)
                portfolio_selector.try_adding(alpha_cutoff_port, name=f'{key}.vol.{cutoff}')

        ## Lets create selection results
        alpha_save_name = f'apex:selected_portfolio:{subuniverse}:{base_alpha_name}'
        if save_suffix:
            alpha_save_name += f':{save_suffix}'
        results[subuniverse] = alpha_save_name
        portfolio = portfolio_selector.portfolio
        backtest = apex__backtest_portfolio_weights(ds_to_df(market_data), portfolio, transaction_costs=transaction_costs)
        strategy_bb.write(alpha_save_name, {
            'portfolio': portfolio,
            'subportfolios': portfolio_selector.portfolios,
            'backtest': backtest,
            'availability': availability,
            'transaction_costs': transaction_costs,
            'blended': blend,
            'market_data': market_data,
            'capped_weights': cap_weights_lo,
            'long_only': long_only,
            'start_date': start_date,
            },
            metadata={
                'subuniverse_name': subuniverse,
                'alpha_name': base_alpha_name
            }
        )
    universe_bb.output_type = 'pd'
    strategy_bb.output_type = 'pd'
    return results


# New idea for portfolio selector
# Start from worst to best performing
# Each portfolio is tested against adding itself from 5% to -5% to the current 'selection portfolio'
# We always pick the best option out of the 11 tests with the metrics we decide a priori

# Terminology: expander, aggregator, transformer
# expander: 1 portfolio -> 100 portfolios
# aggregator: 100 portfolios -> 1 portfolio
# transformer: 100 portfolios -> 100 portfolios

# (1) New family of transforms: take sign of long-short portfolio and decay it at certain rate (different half-lifes)
# New family of alphas: corrcoef at different time-periods for every pair of market data, but also look into modeling the
# correlation matrix itself as well
# New family of transforms: blending at different rates
# New portfolio selector: for blending rates in [1, ..., 10], add 5/x * sharpe/total_sharpes% of that portfolio to the master portfolio
# New alpha idea: Use blending mechanism and corr-coef alpha/short-term momentum data (apply new fam of transforms in (1) to the short-term exponential decaying average return at different rates) - this would basically allow the model to fit to the market's cross-sectional dynamics, time-series momentum, and cross-sectional intertemporal dynamics
# Take the data, put it through a small neural network, with goal of forecasting the return at different range of days
# Thats your alpha signal

# To code this, use exponentially decaying correlation on pandas to compute the rolling correlation of market data series for every stock (volumes vs lows for instance, but every pair of series of market data and for days in [3, 5, 10, 15, 20, 30, 40, 50, 100])
#  create a function that creates all short-term momentum signals (np.sign(x.ewm(halflife=num_days).mean()) for num_days in [1, 3, 5, 10, 20, 40, 60, 80, 100, 150, 200]
# Now pass these signals through the transformer portfolio expander for identity, filtering, and filtered pct change (stop using diffs here because diffs will distort for non-stationary series such as prices)
# Now pass these portfolios through the blending portfolio expander at rates [1, 3, 5, 10, 20, 50, 100]
# Combine portfolios that are profitable by sorting them in ascending order of selection variables and for each i, from N to 1 (1 being best portfolio), p_hat(i+1) = p(i) * (1 - alpha) + p(i+1) * alpha -> try different alphas between 0.25 and 0.05 (exponential decay based on rank)

### The surface algo
# Let N be the number of securities
# Compute N matrices where the column for security i is its returns but elsewhere it is the returns of the vol-normalized spread between i and j
# Apply the signed momentum algorithm, followed by expansion and blending
# I think that if i build the timeseries, cross-sectional, and intertemporal cross-sectional expander it will form the basis of this algo
# To build the aggregator we can try the following:
# 1. Create dynamic control for each strategy - each strategy is on when it is beating the benchmark of 6% annual returns on a rolling 1y window after transaction costs, allow cash ticker
# 2. At the sub-universe level: select strategies based on the fixed point idea - each strategy can add up to 5% of itself to the model; then we keep looping until no strategy but #1 can improve the portfolio, and once it is done adding itself to the model the loop breaks
# 3. Combine all subuniverses portfolios equally weighing them

# Maybe it is better to have a single portfolio per timeseries, cross-sectional, and intertemporal cross-sectional sleeves per feature. Have an expander/collapser within each sleeve. That way we build on a sub-universe basis only 3 x M portfolios (M=features).

# Idea Number 2 to implement: after completing a subuniverse's portfolio, compute its returns on a ticker-by-ticker level,
# and run the same models and their diffs through the portfolio builder against the errors of the portfolio
# - the ideal portfolio can make basically the ideal choices every day and get 100% of the loot
# So what you do is set the market_data['returns'] = returns - strategy_ticker_returns and build
# a strategy on top of this new return stream. This would be a 2nd order strategy
# You should keep doing it until the point where transaction costs doesn't let you do it anymore
# Try to have by launch a model of order 3, and per order you add a level of ts/cs/ics/raw signal portfolios to inputs
# Maybe 1st order you have the model 1 extra day delayed (3 days)
# 2nd order you have no extra delay
# 3rd order you add the 3 ts/cs/ics portfolios to the inputs to help the system, and also add an error forecaster?


TS_TRANSFORMS = {
    'identity': lambda x: x,
    'delayed': lambda x: x.shift(1),

    'ewm1': lambda x: x.ewm(halflife=1).mean(),
    'ewm3': lambda x: x.ewm(halflife=3).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm50': lambda x: x.ewm(halflife=50).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
}

TS_DIFFED_TRANSFORMS = {
    'ewm1_pct_chg': lambda x: x.ewm(halflife=1).mean().pct_change(),
    'ewm3_pct_chg': lambda x: x.ewm(halflife=3).mean().pct_change(),
    'ewm5_pct_chg': lambda x: x.ewm(halflife=5).mean().pct_change(),
    'ewm10_pct_chg': lambda x: x.ewm(halflife=10).mean().pct_change(),
    'ewm20_pct_chg': lambda x: x.ewm(halflife=20).mean().pct_change(),
    'ewm50_pct_chg': lambda x: x.ewm(halflife=50).mean().pct_change(),
    'ewm100_pct_chg': lambda x: x.ewm(halflife=100).mean().pct_change(),
}

def apex__timeseries_expander(dataset, base_signal):
    """
    The timeseries expander takes a signal, expands it across time, and yields back all the expansions
    This signal must be normalized to center at zero per stock
    """
    signal = np.sign(base_signal)

    vol = dataset['returns'].ewm(halflife=20).mean()
    # Now that we've cleaned up the signal let's create the portfolios
    # Because of the way we want to do this we need to split the portfolio into longs and shorts
    longs = signal[signal > 0]
    shorts = signal[signal < 0]
    from toolz import merge
    transforms = merge(TS_TRANSFORMS, TS_DIFFED_TRANSFORMS)
    for transform_name, transform in transforms.items():
        longs_t = transform(longs)
        shorts_t = transform(shorts)
        port_t = longs_t.fillna(0) + shorts_t.fillna(0)
        port_t = port_t.divide(port_t.abs().sum(axis=1))
        yield (transform_name, port_t)

        port_t = longs_t.fillna(0) + shorts_t.fillna(0)
        port_t = port_t / vol
        port_t = port_t.divide(port_t.abs().sum(axis=1))
        yield (transform_name + '_inv_vol_adj', port_t)

        port_t = longs_t.fillna(0) + shorts_t.fillna(0)
        port_t = port_t * vol
        port_t = port_t.divide(port_t.abs().sum(axis=1))
        yield (transform_name + '_vol_adj', port_t)

def apex__crosssectional_expander(dataset, base_signal):
    """
    The cross-sectional expander takes a signal, expands it across time, and then renormalizes it
    cross-sectionally to build a long-short portfolio
    """

    vol = dataset['returns'].ewm(halflife=20).mean()
    # Now that we've cleaned up the signal let's create the portfolios
    # Because of the way we want to do this we need to split the portfolio into longs and shorts
    from toolz import merge
    transforms = merge(TS_TRANSFORMS, TS_DIFFED_TRANSFORMS)
    for transform_name, transform in transforms.items():
        port_t = transform(base_signal)
        port_t = port_t.rank(axis=1, pct=True)
        port_t = port_t.subtract(port_t.median(axis=1), axis=0)
        port_t = port_t.divide(port_t.abs().sum(axis=1), axis=0)
        yield (transform_name, port_t)

        port_t = transform(base_signal)
        port_t = port_t.rank(axis=1, pct=True) / vol
        port_t = port_t.subtract(port_t.median(axis=1), axis=0)
        port_t = port_t.divide(port_t.abs().sum(axis=1), axis=0)
        yield (transform_name + '_inv_vol_adj', port_t)

        port_t = transform(base_signal)
        port_t = port_t.rank(axis=1, pct=True) * vol
        port_t = port_t.subtract(port_t.median(axis=1), axis=0)
        port_t = port_t.divide(port_t.abs().sum(axis=1), axis=0)
        yield (transform_name + '_vol_adj', port_t)

def apex__crosssectional_intertemporal_expander(dataset, signal):
    """
    The cross-sectional intertemporal expander will compute second order interactions

    To do so it computes a portfolio for spreads for each ticker

    # Expansion factor:
    # 1 -> len(tickers) * 2 * 16 * 3 = 1 -> 96 * len(tickers)

    Meaning for each signal it might be worth it to pass these portfolios through dynamic control
    systems and then having a simple average aggregator per signal as explained below
    @ apex__ticker_level_dynamic_control_aggregator or
    @ apex__portfolio_level_dynamic_control_aggregator
    """
    for ticker in dataset.tickers:
        spread_signal = (signal - signal[ticker])[~signal[ticker].isnull()]
        spread_signal = spread_signal.ewm(halflife=1).mean()[~signal[ticker].isnull()]
        spread_signal[ticker] = -spread_signal.sum(axis=1)
        yield from apex__timeseries_expander(dataset, spread_signal)
        yield from apex__crosssectional_expander(dataset, spread_signal)


def apex__compute_strategy_returns_disaggregated(market_data, portfolio, transaction_costs):
    """
    To be parsimonious I'm drifting everything by an extra day
    """
    returns = market_data['returns']
    portfolio = portfolio.shift(1) # Addl lag

    portfolio_drifted = (portfolio.shift(1) * (1 + returns))

    trades = portfolio - portfolio_drifted
    return (portfolio.shift(1) * returns - trades.abs() * transaction_costs * 0.0001)

def apex__ticker_level_dynamic_control_aggregator(dataset, portfolio_pipeline, financing_costs_bps=600):
    portfolio = None
    daily_financing_decay = financing_costs_bps / 100 / 100 / 252
    for _, p in portfolio_pipeline:
        p_rets = apex__compute_strategy_returns_disaggregated(dataset, p, 15) - daily_financing_decay * np.abs(p)
        p_cum_rets = (1 + p_rets.fillna(0)).rolling(22).prod()
        p_dc = (p_cum_rets > 1).ewm(halflife=1).mean()
        p = p * p_dc
        p = p[availability].fillna(0)
        if portfolio is None:
            portfolio = p
        else:
            portfolio = portfolio + p

    portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)
    return portfolio


def apex__portfolio_level_dynamic_control_aggregator(dataset, portfolio_pipeline, financing_costs_bps=600):
    portfolio = None
    daily_financing_decay = financing_costs_bps / 100 / 100 / 252
    for _, p in portfolio_pipeline:
        p_rets = apex__compute_strategy_returns(dataset, p, 15)
        # 5 years is what matters
        p_cum_rets = (1 + (p_rets.dropna() - daily_financing_decay)).rolling(252*5).prod()
        p_dd = 1 - p_cum_rets/p_cum_rets.rolling(252*5).max()
        alpha = financing_costs_bps / 100 / 100
        p_dc = 1 - np.minimum(p_dd / alpha, 1)
        p_dc = p_dc.ewm(halflife=1).mean()
        p = p.multiply(p_dc, axis=0)
        p = p[availability].fillna(0)
        if portfolio is None:
            portfolio = p
        else:
            portfolio = portfolio + p

    portfolio = portfolio.divide(portfolio.abs().sum(axis=1), axis=0)
    return portfolio

