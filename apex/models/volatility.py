# Variables for setting import
import concurrent.futures as cf
import datetime
import logging
import math
import pickle
import re
import time
import typing
import uuid
import warnings
from collections import ChainMap, OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial, reduce
from pathlib import Path
from pprint import pprint
from typing import Mapping, Sequence, TypeVar

# Other
import toml

# Default imports
import numpy as np
import numba as nb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import numba as nb
import scipy as sc
import sklearn as sk
import statsmodels.api as sm
import toolz as tz
import funcy as fy
import boltons as bs
import dogpile.cache as dc
import matplotlib.pyplot as plt
import torch.optim as to
import torch
import torch.tensor as tt
import torch.nn as tnn
import torch.nn.functional as F


# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from IPython.core.debugger import set_trace as bp
from scipy.optimize import Bounds, minimize
from toolz import partition_all

import dask_ml
import inflect
import joblib
##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.store import ApexDataStore
from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import (ApexBloomberg, get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.toolz.sampling import sample_indices, sample_values, ssample
from apex.universe import APEX_UNIVERSES
from apex.alpha.market_alphas import MARKET_ALPHAS
from joblib import Parallel, delayed, parallel_backend
from apex.pipelines.factor_model import bloomberg_field_factor_data, UNIVERSE_DATA_CACHING, rank_signal_from_bbg, get_market_data, compute_alpha_results, construct_alpha_portfolio_from_signal, max_abs_scaler, min_max_scaler


##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.store import ApexDataStore
from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import (ApexBloomberg, get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.toolz.sampling import sample_indices, sample_values, ssample
from apex.universe import APEX_UNIVERSES
from apex.alpha.market_alphas import MARKET_ALPHAS
from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.pipelines.covariance import lw_cov, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__adjusted_market_data,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_bloomberg_field_score,
                                         get_market_data, rank_signal_from_bbg)
from apex.pipelines.risk import (compute_account_active_risk_contrib,
                                 compute_account_total_risk_contrib)
from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg, apex__unadjusted_market_data,
                                  fix_security_name_index, get_security_metadata,
                                  get_index_member_weights_on_day)
from sklearn.linear_model import HuberRegressor

from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.universe import ApexCustomRoweIndicesUniverse, ApexUniverseAMNA
from apex.toolz.bloomberg import get_index_member_weights_multiday, get_security_historical_data
from apex.factors.by_universe import universe_size_factor
from apex.toolz.volatility import normalize_vol

from apex.toolz.universe import ApexUniverseAMNA, AMNA_AVAILABILITY_REMOVALS
from apex.toolz.bloomberg import BLOOMBERG_MARKET_DATA_FIELDS

Blackboard = lambda: defaultdict(Blackboard)

def fill_nas_series(x):
    last = x.last_valid_index()
    x.loc[:last] = x.loc[:last].ffill()
    return x

def instantiate_model(model_name):
    bb = Blackboard()
    universe = ['VXXB US Equity', 'SPY US Equity', 'SH US Equity', 'SPX Index', 'VIX Index']
    bb['name'] = model_name
    bb['universe'] = universe
    bb['security_metadata'] = get_security_metadata(*universe)
    adjusted_data = apex__adjusted_market_data(*universe, parse=True).loc['1990':]
    bbg = ApexBloomberg()
    ux_data = bbg.history('UX1 Index', BLOOMBERG_MARKET_DATA_FIELDS).swaplevel(axis=1)
    ux_data = ux_data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'})
    ux_data['returns'] = ux_data['returns']/100
    data = pd.concat([adjusted_data, ux_data], axis=1).sort_index(axis=1)
    bb['market_data'] = data
    bb['universe'].append('UX1 Index')
    bb['securities'] = [ApexSecurity.from_id(x) for x in universe]
    return bb

def compute_availability(model):
    data = model['market_data']['px_last'].apply(fill_nas_series).dropna(how='all')
    for bbid, date in AMNA_AVAILABILITY_REMOVALS.items():
        if bbid in data.columns:
            data.loc[date:, bbid] = np.nan
    availability = ~data.isnull()
    availability = availability & (data > 10)
    model['availability'] = availability
    return model

def create_settings(model):
    model['settings']['base_signal'] = {
        'window': 200,
    }
    return model


def compute_signal(data, bl=-0.08, bu=0.08, sl=-0.02, su=0.02, alpha=0.1):
    """
    For the VIX strategy we need the SPX and the
    """
    ux_data = data['px_last'][['VIX Index', 'UX1 Index']].dropna()
    relative_basis = ux_data['UX1 Index']/ux_data['VIX Index'] - 1
    dates = relative_basis.index.tolist()
    result = pd.DataFrame(np.nan, columns=['VXXB US Equity', 'SH US Equity', 'SPY US Equity'], index=dates)
    long_vxx_signal = relative_basis <= bl
    short_vxx_signal = relative_basis >= bu
    flat = (sl <= relative_basis) & (relative_basis <= su)

    result.loc[long_vxx_signal, 'VXXB US Equity'] = 1
    result.loc[long_vxx_signal, 'SPY US Equity'] = 1

    result.loc[short_vxx_signal, 'VXXB US Equity'] = -1
    result.loc[short_vxx_signal, 'SH US Equity'] = 1
    result.loc[flat, :] = 0
    result = result.fillna(method='ffill')
    result = result.fillna(0)#.ewm(alpha=alpha).mean()
    result = result.divide(result.abs().sum(axis=1), axis=0)
    result = result.fillna(0)
    return result
    return pd.DataFrame(result).T/2

    for day in dates:
        curr_rb = relative_basis.loc[day]
        if curr_rb <= bl:
            if in_vxx_spy:
                pass
            else:
                curr_signal = {'SH US Equity': 0, 'VXXB US Equity': 1, 'SPY US Equity': 1}
                in_vxx_spy = True
        if sl <= curr_rb <= su:
            curr_signal = {'SH US Equity': 0, 'VXXB US Equity': 0, 'SPY US Equity': 0}
            in_svxy_sh = False
            in_vxx_spy = False
        if curr_rb >= bu:
            if in_svxy_sh:
                pass
            else:
                curr_signal = {'VXXB US Equity': -1, 'SH US Equity': 1}
                in_svxy_sh = True
        result[day] = curr_signal
    return pd.DataFrame(result).T/2


# In[286]:


def compute_signal_fitness(data, bl=-0.08, bu=0.08, sl=-0.02, su=0.02, alpha=0):
    signal = compute_signal(data, bl=bl, bu=bu, sl=sl, su=su, alpha=alpha).shift(1)
    returns = data['returns'][signal.columns]
    result = signal * returns
    result = result.dropna()
    result = result.sum(axis=1).loc['2010':]
    try:
        print((1+result).prod(), result.mean()/result.std()*np.sqrt(252))
        return (1+result).prod() + result.mean()/result.std()*np.sqrt(252) * (1+result.prod())/10
    except:
        return -np.inf

class vix_strategy_optimization:
    def __init__(self, data):
        self.dim = 4
        self.data = data

    def fitness(self, x):
        bl, bu, sl, su = x[0], x[1], x[2], x[3]
        result = [-compute_signal_fitness(self.data, bl=bl, bu=bu, sl=sl, su=su)]
        return result
    def get_bounds(self):
        # Bounds are based on the scores.
        upper_bounds, lower_bounds = [0, 0.1, 0, 0.1], [-0.1, 0, -0.1, 0]
        return (lower_bounds, upper_bounds)

    def get_name(self):
        return "VIX Strategy Optimization"

    def get_nic(self):
        return 0

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


def get_model():
    model = instantiate_model('apex__volatility_risk_premia')
    model = compute_availability(model)
    model = create_settings(model)


    OPTIMAL_PARAMS = {'bl': -0.08796015761414766,
        'bu': 0.09383745663228775,
        'sl': -0.05754355829809474,
        'su': 0.03885727727474252
    }
    signal = compute_signal(model['market_data'], **OPTIMAL_PARAMS).fillna(0)
    signal = signal.divide(signal.abs().sum(axis=1), axis=0).fillna(0)
    model['weights'] = signal
    return model


def optimize_vix_strategy(data):
    """
    Problem: maximize sum(abs(signal_i)*log(abs(signal_i)))
    s.t. w'Cw < sigma
    """
    vix_strategy = vix_strategy_optimization(data)
    nl = pg.nlopt('cobyla')
    #nl.set_string_option("hessian_approximation", "limited-memory")
    algo = pg.algorithm(nl)
    pop = pg.population(vix_strategy, 5)
    pop = algo.evolve(pop)
    result = pd.Series(pop.champion_x, ['bl', 'bu', 'sl', 'su'])
    return result

def optimize_vix_strategy(data):
    """
    Problem: maximize sum(abs(signal_i)*log(abs(signal_i)))
    s.t. w'Cw < sigma
    """
    vix_strategy = vix_strategy_optimization(data)
    #nl.set_string_option("hessian_approximation", "limited-memory")
    #subalgo = pg.algorithm(pg.nlopt('cobyla'))
    #algo = pg.algorithm(pg.compass_search(max_fevals = 500))
    subalgo = pg.algorithm(pg.pso_gen(gen=10))
    algo = pg.algorithm(pg.mbh(algo=subalgo, stop=20))
    algo.set_verbosity(1)
    pop = pg.population(vix_strategy, 25)
    pop = algo.evolve(pop)
    result = pd.Series(pop.champion_x, ['bl', 'bu', 'sl', 'su'])
    return result

