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
from functools import partial, reduce, wraps
from pathlib import Path
from pprint import pprint
from typing import Mapping, Sequence, TypeVar

import boltons as bs
import dask_ml
import dogpile.cache as dc
import funcy as fy
import inflect
import joblib
import matplotlib.pyplot as plt
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import pyomo as po
import scipy as sc
import sklearn as sk
import statsmodels.api as sm
# Other
import toml
import toolz as tz
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as to
import torch.tensor as tt
# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from IPython.core.debugger import set_trace as bp
from joblib import Parallel, delayed, parallel_backend
from scipy.optimize import Bounds, minimize
from toolz import partition_all

from apex.alpha.market_alphas import MARKET_ALPHAS
##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__bbg_experiment_pipeline,
                                         apex__market_alpha_experiment_pipeline,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_market_alpha_scores,
                                         construct_alpha_portfolio_from_signal,
                                         get_market_data, max_abs_scaler,
                                         min_max_scaler, rank_signal_from_bbg)
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

from apex.pipelines.factor_model import (bloomberg_field_factor_data, UNIVERSE_DATA_CACHING, rank_signal_from_bbg, get_market_data, compute_alpha_results)



@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='v0.2')
def apex__bloomberg_fundamental_field_data(field_name, universe):
    market_data = get_market_data(universe)
    universe = APEX_UNIVERSES[universe]
    bbg = ApexBloomberg()
    result = bbg.fundamentals(universe, field_name)
    result.columns = result.columns.droplevel(1)
    result = result.reindex(market_data.index).fillna(method='ffill')
    result = result[market_data['px_last'].columns]
    result = result[~market_data['px_last'].isnull()]
    return result

def compute_bloomberg_field_score(field_name, universe, cutoff=0):
    data = bloomberg_field_factor_data(field_name, universe)
    return rank_signal_from_bbg(data, cutoff=cutoff)

def apex__bbg_experiment_pipeline(field_name, universe):
    score = compute_bloomberg_field_score(field_name, universe)
    alpha_name = ' '.join(field_name.split('_'))
    return compute_alpha_results(alpha_name.capitalize(), score, universe=universe)

def apex__data_experiment_pipeline(alpha_name, data_fn, universe, cutoff=0):
    data = data_fn(universe)
    score = rank_signal_from_bbg(data, cutoff=cutoff)
    return compute_alpha_results(alpha_name.capitalize(), score, universe=universe)


def profitability_signals(universe):
    market_data = get_market_data(universe)
    income_xo = apex__bloomberg_fundamental_field_data('IS_INC_BEF_XO_ITEM', universe)
    assets_to_normalize_by = apex__bloomberg_fundamental_field_data('BS_TOT_ASSET', universe)
    cfo = apex__bloomberg_fundamental_field_data('CF_CASH_FROM_OPER', universe)

    roa = income_xo/assets_to_normalize_by
    iroa = (roa > 0).astype(int)

    droa = roa.diff(252)
    idroa = (droa > 0).astype(int)

    cfo = cfo / assets_to_normalize_by
    icfo = (cfo > 0).astype(int)

    f_accruals = (cfo > roa).astype(int)
    return f_accruals + icfo + idroa + iroa


def leverage_and_liquidity_signals(universe):
    market_data = get_market_data(universe)
    long_term_borrow = apex__bloomberg_fundamental_field_data('BS_LT_BORROW', universe)
    current_ratio = apex__bloomberg_fundamental_field_data('CUR_RATIO', universe)
    assets_to_normalize_by = apex__bloomberg_fundamental_field_data('BS_TOT_ASSET', universe)

    leverage = long_term_borrow/assets_to_normalize_by
    idleverage = leverage.diff(252) < 0

    idliquid = current_ratio.diff(252) > 0
    return idliquid + idleverage


def operating_efficiency(universe):
    market_data = get_market_data(universe)
    gross_margin = apex__bloomberg_fundamental_field_data('GROSS_MARGIN', universe)
    turnover_ratio = apex__bloomberg_fundamental_field_data('ASSET_TURNOVER', universe)
    assets_to_normalize_by = apex__bloomberg_fundamental_field_data('BS_TOT_ASSET', universe)

    idmargin = gross_margin.diff(252) > 0
    idturnover = turnover_ratio.diff(252) > 0

    return idturnover + idmargin

def get_earnings_revision_signal(universe):
    market_data = get_market_data(universe)
    eeps_next_year = bloomberg_field_factor_data('EEPS_NXT_YR', universe).dropna(how='all').resample('B').last().reindex(market_data.index).fillna(method='ffill')
    eeps_change_one = ((eeps_next_year - eeps_next_year.shift(70))/eeps_next_year.abs())
    eeps_ratio_one = eeps_change_one / eeps_next_year.abs()
    eeps_ratio_one = eeps_ratio_one / eeps_ratio_one.rolling(252).std()
    eeps_signal_one = min_max_scaler(eeps_ratio_one)

    eeps_change_two = ((eeps_next_year - eeps_next_year.shift(5))/eeps_next_year.abs())
    eeps_ratio_two = eeps_change_two / eeps_next_year.abs()
    eeps_ratio_two = eeps_ratio_two / eeps_ratio_two.rolling(50).std()
    eeps_signal_two = max_abs_scaler(eeps_ratio_two)

    eeps_change_three = ((eeps_next_year - eeps_next_year.shift(40))/eeps_next_year.abs())
    eeps_ratio_three = eeps_change_three / eeps_next_year.abs()
    eeps_ratio_three = eeps_ratio_three / eeps_ratio_three.rolling(50).std()
    eeps_signal_three = rank_signal_from_bbg(eeps_ratio_three)

    data = max_abs_scaler((eeps_signal_one + eeps_signal_two + eeps_signal_three)/3.0)
    market_data = get_market_data(universe)
    data = data[~market_data['px_last'].isnull()]
    return data > 0

def berry_ratio_factor(universe):
    gross_profits = apex__bloomberg_fundamental_field_data('gross_profit', universe)
    operating_expenses = apex__bloomberg_fundamental_field_data('is_tot_oper_exp', universe)
    return ((gross_profits/operating_expenses).fillna(method='ffill', limit=20))

def cashflow_roe(universe):
    fcf_roe = apex__bloomberg_fundamental_field_data('FREE_CASH_FLOW_EQUITY', universe)
    return fcf_roe

def return_on_invested_capital(universe):
    roic = apex__bloomberg_fundamental_field_data('RETURN_ON_INV_CAPITAL', universe)
    adj_roic = apex__bloomberg_fundamental_field_data('ADJUSTED_ROIC_AS_REPORTED', universe)
    return rank_signal_from_bbg(roic) + rank_signal_from_bbg(adj_roic)

def price_to_book(universe):
    pxtobook = bloomberg_field_factor_data('PX_TO_BOOK_RATIO', universe)
    return pxtobook


def ebit_yield(universe):
    ebit_y = bloomberg_field_factor_data('EBIT_YIELD', universe)
    return ebit_y

def earnings_yield(universe):
    return bloomberg_field_factor_data('EARN_YLD_HIST', universe)

def shareholder_yield(universe):
    return bloomberg_field_factor_data('SHAREHOLDER_YIELD', universe)

def get_quantamental_score(universe):
    QUANTAMENTAL_FACTORS = {
        'Berry Factor': berry_ratio_factor,
        'Cashflow ROE': cashflow_roe,
        'ROIC': return_on_invested_capital,
        'Price to Book': price_to_book,
        'EBIT Yield': ebit_yield,
        'Earnings Yield': earnings_yield,
        'Shareholder Yield': shareholder_yield
    }
    QUANTAMENTAL_FACTORS_SIGNS = {
        'Berry Factor': 1,
        'Cashflow ROE': -1,
        'ROIC': 1,
        'Price to Book': -1,
        'EBIT Yield': 1,
        'Earnings Yield': 1,
        'Shareholder Yield': 1
    }
    scores = {}
    for f, fn in QUANTAMENTAL_FACTORS.items():
        scores[f] = QUANTAMENTAL_FACTORS_SIGNS[f] * fn(universe).ewm(15).mean()

    return (rank_signal_from_bbg(pd.concat(scores, axis=1).sum(axis=1, level=1), 0.25) > 0).astype(int)

def get_announcement_drift_score(universe):
    market_data = get_market_data(universe)
    data = apex__bloomberg_fundamental_field_data('IS_DIL_EPS_CONT_OPS', 'AMNA')
    signal = (data - data.fillna(method='ffill').rolling(250).mean())/data.fillna(method='ffill').expanding().std()
    signal = signal.dropna(how='all')
    signal = rank_signal_from_bbg(signal, cutoff=0.)
    signal = signal[~market_data['px_last'].isnull()]
    return (signal > 0).astype(int)

def get_asset_growth_fade_score(universe):
    market_data = get_market_data(universe)
    total_assets = apex__bloomberg_fundamental_field_data('BS_TOT_ASSET', 'AMNA')
    signal = (total_assets - total_assets.fillna(method='ffill').rolling(22*9).mean())/total_assets
    signal = signal.dropna(how='all')
    signal = rank_signal_from_bbg(signal, cutoff=0.8)
    signal = signal[~market_data['px_last'].isnull()]
    signal = signal.dropna(how='all')
    return -(signal > 0).astype(int)

def get_value_score(universe):
    """
    Value strategy. 1/n active risk allocation. 50% of active risk.
    """
    market_data = get_market_data(universe)
    quarterly_strats = profitability_signals(universe) + leverage_and_liquidity_signals(universe) + operating_efficiency(universe)
    earnings_revision = get_earnings_revision_signal(universe)

    quantamental_score = get_quantamental_score(universe)
    announcement_drift = get_announcement_drift_score(universe)
    asset_fade_score = get_asset_growth_fade_score(universe)
    result = (asset_fade_score + quarterly_strats + earnings_revision + quantamental_score + announcement_drift)
    signals = rank_signal_from_bbg(result, cutoff=0.)
    # Now portfolio
    return signals

def get_value_scores_by_universe(universe):
    """
    Value strategy. 1/n active risk allocation. 50% of active risk.
    """
    return {
        'profitability': profitability_signals(universe),
        'leverage_and_liquidity': leverage_and_liquidity_signals(universe),
        'operating_efficiency': operating_efficiency(universe),
        'earnings_revision': get_earnings_revision_signal(universe),
        'quantamental_score': get_quantamental_score(universe),
        'announcement_drift': get_announcement_drift_score(universe),
        'asset_fade_score': get_asset_growth_fade_score(universe),
    }