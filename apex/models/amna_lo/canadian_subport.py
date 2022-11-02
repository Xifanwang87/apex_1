%matplotlib inline
%load_ext autoreload
%autoreload 2

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
import pyomo as po
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
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.security import ApexSecurity
from apex.toolz.bloomberg import ApexBloomberg, get_security_metadata, apex__adjusted_market_data as apex__amd
from apex.toolz.dask import ApexDaskClient, compute_delayed
from joblib import Parallel, delayed, parallel_backend


Blackboard = lambda: defaultdict(Blackboard)
from apex.toolz.universe import ApexUniverseAMNA

BROAD_UNIVERSE = ApexUniverseAMNA().tickers

CANADIANS = [x for x in BROAD_UNIVERSE if x.split(' ')[1] == 'CN']
NON_CANADIANS = [x for x in BROAD_UNIVERSE if x not in CANADIANS]


from toolz import valfilter, reduce
import dogpile.cache as dc

MODEL_CACHE = dc.make_region(key_mangler=lambda key: "apex:midstream:canadians_vs_non_canadians:cache" + key).configure(
    'dogpile.cache.redis',
    arguments = {
        'host': '10.15.201.154',
        'port': 6379,
        'db': 6,
        'redis_expiration_time': 60*60*4,   # 4 hours
    },
)

from dataclasses import dataclass, field
import redis
import pickle
from apex.toolz.bloomberg import get_security_fundamental_data
from joblib import Parallel, delayed, parallel_backend
import funcy

BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS = [
    'current_ev_to_t12m_ebitda',
    'current_px_to_free_cash_flow',
    'earn_yld_hist',
    'free_cash_flow_yield',
    'pe_ratio',
    'px_to_book_ratio',
    'px_to_cash_flow',
    'px_to_ebitda',
    'px_to_free_cash_flow',
    'px_to_sales_ratio',
    'shareholder_yield',
    'short_int_ratio',
]


@MODEL_CACHE.cache_multi_on_arguments(namespace='fundamental_data:v1', asdict=True)
def apex__fundamental_data(*identifiers):
    result = {}
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

@dataclass
class ApexModel:
    name: str
    _db_client: typing.Any = field(init=False)

    def __post_init__(self):
        self._db_client = redis.Redis(host='10.15.201.160', port='16383', db=0)

    def mangle_key(self, key):
        return self.name + ':' + key

    def __getitem__(self, key):
        return pickle.loads(self._db_client.get(self.mangle_key(key)))

    def __setitem__(self, key, value):
        return self._db_client.set(self.mangle_key(key), pickle.dumps(value))

    def set_universe(self, universe: list):
        self['universe'] = universe

    @property
    def universe(self):
        return self['universe']

    def market_data(self, update=False):
        if not update:
            try:
                result = self['market_data']
                if result is None:
                    return self.market_data(update=True)
            except:
                pass
        self['market_data'] = apex__amd(*self.universe, parse=True)
        return self['market_data']

    def fundamental_data(self, update=False):
        bbg = ApexBloomberg()
        if not update:
            try:
                return self['fundamental_data']
            except:
                pass
        fundamental_data = apex__fundamental_data(*self.universe)
        self['fundamental_data'] = pd.concat([x.apply(fill_nas_series) for x in fundamental_data.values()], axis=1).apply(fill_nas_series)
        return self['fundamental_data']

    @property
    def availability_high_liquidity(self):
        market_data = self.market_data()
        availability = market_data['px_last'].apply(fill_nas_series).rolling(250).median() > 10
        availability = availability & ((market_data['px_volume'] * market_data['px_last']).apply(fill_nas_series).rolling(250).mean() > 1000000)
        return availability


def crossectional_rank(alpha):
    alpha_res = alpha.rank(axis=1)
    alpha_res = alpha_res.subtract(alpha_res.mean(axis=1), axis=0)
    alpha_res = alpha_res.divide(alpha_res.abs().sum(axis=1), axis=0)
    return alpha_res

@dataclass
class ApexMidstreamCanadians(ApexModel):
    name: str = field(default='apex:midstream:canadians_vs_non_canadians:canadians')

    def base_lo_model(self, update=False):
        """
        Base model on high liquidity
        """
        availability = self.availability_high_liquidity
        fundamental_data = self.fundamental_data(update=update)
        market_data = self.market_data(update=update)
        fundamental_data_by_field = fundamental_data.swaplevel(axis=1).sort_index(axis=1)
        base_factor_model = fundamental_data_by_field['free_cash_flow_yield'][availability].rank(axis=1).fillna(0)
        base_factor_model += fundamental_data_by_field['earn_yld_hist'][availability].rank(axis=1).fillna(0)
        base_factor_model += fundamental_data_by_field['current_ev_to_t12m_ebitda'][availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model += fundamental_data_by_field['pe_ratio'][availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model += market_data['returns'].fillna(0).ewm(span=200).mean()[availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model = crossectional_rank(base_factor_model[availability])
        long_only_model = base_factor_model.rank(axis=1)[availability].fillna(0)
        long_only_model = long_only_model.divide(long_only_model.sum(axis=1), axis=0)
        return long_only_model.fillna(0)


@dataclass
class ApexMidstreamNonCanadians(ApexModel):
    name: str = field(default='apex:midstream:canadians_vs_non_canadians:non_canadians')

    def base_lo_model(self, update=False):
        """
        Base model on high liquidity
        """
        availability = self.availability_high_liquidity
        fundamental_data = self.fundamental_data(update=update)
        market_data = self.market_data(update=update)
        fundamental_data_by_field = fundamental_data.swaplevel(axis=1).sort_index(axis=1)
        base_factor_model = fundamental_data_by_field['free_cash_flow_yield'][availability].rank(axis=1).fillna(0)
        base_factor_model += fundamental_data_by_field['earn_yld_hist'][availability].rank(axis=1).fillna(0)
        base_factor_model += fundamental_data_by_field['current_ev_to_t12m_ebitda'][availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model += fundamental_data_by_field['pe_ratio'][availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model += market_data['returns'].fillna(0).ewm(span=200).mean()[availability].rank(axis=1, ascending=False).fillna(0)
        base_factor_model = crossectional_rank(base_factor_model[availability])
        long_only_model = base_factor_model.rank(axis=1)[availability].fillna(0)
        long_only_model = long_only_model.divide(long_only_model.sum(axis=1), axis=0)
        return long_only_model.fillna(0)


def compute_model_returns(wts, market_data):
    returns = market_data['returns']
    return (wts.shift(2) * returns - wts.diff().shift(2) * 0.0005).sum(axis=1)

def compute_model_weights(canadian_model, noncanadian_model):
    canadian_wts = canadian_model.base_lo_model(update=True)
    noncanadian_wts = noncanadian_model.base_lo_model(update=True)
    canadian_rets = compute_model_returns(canadian_wts, canadian_model.market_data(update=False))
    noncanadian_rets = compute_model_returns(noncanadian_wts, noncanadian_model.market_data(update=False))


    result_models = pd.DataFrame({'Canadians': canadian_rets, 'Non-Canadians': noncanadian_rets})
    result_weights = 1.0/result_models.ewm(span=20).std()
    result_weights = result_weights.multiply(result_models.ewm(span=20).std().sum(axis=1), axis=0)
    result_weights = result_weights.divide(result_weights.sum(axis=1), axis=0).shift(1)

    canadian_wts = canadian_wts.multiply(result_weights['Canadians'], axis=0)
    noncanadian_wts = noncanadian_wts.multiply(result_weights['Non-Canadians'], axis=0)

    return pd.concat([canadian_wts, noncanadian_wts], axis=1)


def get_model_weights():
    canadian_model = ApexMidstreamCanadians()
    canadian_model.set_universe(CANADIANS)

    noncanadian_model = ApexMidstreamNonCanadians()
    noncanadian_model.set_universe(NON_CANADIANS)
    return compute_model_weights(canadian_model, noncanadian_model)