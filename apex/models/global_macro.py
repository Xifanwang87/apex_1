
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

import matplotlib.pyplot as plt
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
import toolz as tz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tt
from IPython.core.debugger import set_trace as bp
from scipy.optimize import Bounds, minimize
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   RobustScaler, StandardScaler)
from toolz import partition_all
from torch.utils.data import *
from torch.utils.data.sampler import *

import boltons as bs
import dask_ml
import dogpile.cache as dc
import funcy as fy
import inflect
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import pyomo as po
import statsmodels.api as sm
# Other
import toml
from apex.accounts import (get_account_holdings_by_date,
                           get_account_weights_by_date)
from apex.alpha.market_alphas import MARKET_ALPHAS
##########
## APEX ##
##########
from apex.data.access import get_security_market_data
from apex.factors.by_universe import (UNIVERSE_NEUTRAL_FACTORS,
                                      apex__universe_bloomberg_fundamental_field)
# Local
from apex.nn.layers import NoisyLinear, Swish, Swish1
from apex.pipelines.covariance import lw_cov, min_cov_det, oas_cov
from apex.pipelines.factor_model import (UNIVERSE_DATA_CACHING,
                                         apex__adjusted_market_data,
                                         bloomberg_field_factor_data,
                                         compute_alpha_results,
                                         compute_bloomberg_field_score,
                                         get_market_data, rank_signal_from_bbg)
from apex.pipelines.risk import (compute_account_active_risk_contrib,
                                 compute_account_total_risk_contrib)
from apex.security import ApexSecurity
from apex.store import ApexDataStore
from apex.toolz.arctic import ArcticApex
from apex.toolz.bloomberg import (ApexBloomberg, apex__adjusted_market_data,
                                  fix_security_name_index,
                                  get_index_member_weights_on_day,
                                  get_security_fundamental_data,
                                  get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.toolz.mutual_information import ApexMutualInformationAnalyzer
from apex.toolz.sampling import sample_indices, sample_values, ssample
from apex.toolz.universe import ApexCustomRoweIndicesUniverse, ApexUniverseAMNA
from apex.universe import APEX_UNIVERSES
# Others
from dataclasses import dataclass, field
from distributed import fire_and_forget
from joblib import Parallel, delayed, parallel_backend
from torchvision.transforms import Compose

from apex.optimization.eve import Eve


def compute_universe_factor_scores(universe, factor_dictionary=UNIVERSE_NEUTRAL_FACTORS.copy()):
    FACTOR_SCORES = {}
    for factor_name, factor_fn in factor_dictionary.items():
        try:
            FACTOR_SCORES[factor_name] = factor_fn(universe)
        except:
            pass
    return pd.concat(FACTOR_SCORES, axis=1)


SCALERS = {
    'robust': RobustScaler,
    'standard': StandardScaler,
    'normalizer': Normalizer,
    'maxabs': MaxAbsScaler,
    'minmax': MinMaxScaler
}

def apply_scaler(scaler):
    def scaling_function(data):
        for ix in data.index[35:]:
            data.loc[ix] = scaler(data.loc[:ix]).iloc[-1]
        data = data.iloc[35:]
        return data
    return scaling_function

def scale_fn_creator(scaler):
    def scale_function(data):
        result = pd.DataFrame(scaler().fit_transform(data), index=data.index, columns=data.columns)
        return result
    return apply_scaler(scale_function)

COLUMN_SCALERS = {k: scale_fn_creator(SCALERS[k]) for k in SCALERS}
@UNIVERSE_DATA_CACHING.cache_on_arguments(namespace='broad_indices')
def apex__broad_index_bloomberg_field_factor(field_name):
    indices = ['AMEI Index',
        'AMZ Index',
        'AMZI Index',
        'DJSOES Index',
        'HHO Index',
        'MXWD0EN Index',
        'MXWO0EG Index',
        'MXWO0EN Index',
        'NEX Index',
        'S12OILR Index',
        'S15OILR Index',
        'S5OILE Index',
        'S5OILP Index',
        'S5OILR Index',
        'S6OILR Index',
        'SPGNEUP Index',
        'SPGSSINR Index',
        'SPGTAE Index',
        'SPMLP Index',
        'SPSIOP Index',
        'SPSIOS Index',
        'SPTRSC10 Index',
        'SPX Index',
    ]
    bbg = ApexBloomberg()
    result = bbg.history(indices, field_name)
    result.columns = result.columns.droplevel(1)
    return result[indices]

def adv_minus_decl():
    adv_minus_decl = apex__broad_index_bloomberg_field_factor('NUM_DAILY_ADV_MINUS_DECL').dropna(how='all').ewm(span=20).mean()
    adv_minus_decl.columns = [x + ' A-D' for x in adv_minus_decl.columns]
    return adv_minus_decl

def get_global_macro_data(return_tensor=False):
    """
    Most important indicators for the economy (forward looking) that we can have on a daily basis that affects MLPS
    - VIX
    - S&P 500 Momentum
    - Crude Oil
    - Natural Gas Prices
    - EIA Storage Numbers
    - Interest Rates
    - Crude Oil Volatility
    - AMZ/AMEI/SPMLP Volatility
    - Russell 2000 Volatility
    - Russell 2000 Momentum
    - Etc

    Now... since I want to forecast *changes* in volatility, i can't have any information that might tell the NN
    where we are in the sample in terms of time.

    So... need to diff everything so that there's no levels.
    """
    BASE_INDICES = [
        'SPX Index',
        'VIX Index',
        'RTY Index',
        'NG1 Comdty',
        'CL1 Comdty',
        'AMZ Index',
        'SPMLP Index',
        'SPXAD Index',
        'RUT Index',
        'DXY Index',

        'USYC2Y10 Index', # 2-10s Spread
        'LPGSMBPE Index', #Ethane
        'LPGSMBPP Index', # Propane
        'LPGSMBNB Index', # Normal Butane Gasoline
        'LPGSMBIB Index', # Butane
        'LPGSMBNG Index', # Natural Gasoline
        'CRKS321C Index', # Crack Spread

        'LF98TRUU Index',
        'LUACTRUU Index',
        'LG30TRUU Index',
        'LEGATRUU Index',
        'LGTRTRUU Index',
        'LF94TRUU Index',
        'LGCPTRUH Index',
        'LGCPTRUU Index',
        'BEBGTRUU Index',
        'BASPTDSP Index',
        'ENASENA2 Index',
        '.CL1-2 G Index',
        '.WIDOW G Index',
        '.CL1DXYCO G Index',
        '.CL/NG G Index',
        '.CLO G Index',
        '.CLSPR12 G Index',
        '.CL1-CL4 G Index'

    ]
    indices = BASE_INDICES.copy()

    result = {}
    economic_data = apex__adjusted_market_data(*indices, parse=True)['px_last'].fillna(method='ffill', limit=5)

    result['S&P500 Realized Variance'] = economic_data['SPX Index'].pct_change().pow(2) * 100 * 260
    result['S&P500 Volatility Premium'] = economic_data['VIX Index'] - economic_data['SPX Index'].pct_change().pow(2).rolling(3).mean().pow(0.5) * 100 * np.sqrt(260)

    result['AMZ Realized Variance'] = economic_data['AMZ Index'].pct_change().pow(2) * 100 * 260
    bbg = ApexBloomberg()
    iv = bbg.history('AMJ US Equity', 'HIST_CALL_IMP_VOL')
    iv = iv[iv.columns[0]]
    result['AMZ Volatility Premium'] = iv - economic_data['AMZ Index'].pct_change().pow(2).rolling(20).mean().pow(0.5) * 100 * np.sqrt(260)

    oil_iv = bbg.history('CL1 Comdty', 'HIST_CALL_IMP_VOL')
    oil_iv = oil_iv[oil_iv.columns[0]]
    result['Oil Volatility Premium'] = oil_iv - economic_data['CL1 Comdty'].pct_change().pow(2).rolling(3).mean().pow(0.5) * 100 * np.sqrt(260)

    rut_iv = bbg.history('RUT Index', 'HIST_CALL_IMP_VOL')
    rut_iv = rut_iv[rut_iv.columns[0]]
    result['Russell 2000 VRP'] = rut_iv - economic_data['RUT Index'].pct_change().pow(2).rolling(3).mean().pow(0.5) * 100 * np.sqrt(260)

    result['Butane Realized Variance'] = economic_data['LPGSMBIB Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['Ethane Realized Variance'] = economic_data['LPGSMBPE Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['Propane Realized Variance'] = economic_data['LPGSMBPP Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['Natural Gasoline Realized Variance'] = economic_data['LPGSMBNG Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['Normal Butane Realized Variance'] = economic_data['LPGSMBNB Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['Crack Spread Realized Variance'] = economic_data['CRKS321C Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260

    result['Crack Spread RV 20D Diff'] = (economic_data['CRKS321C Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)
    result['Ethane RV 20D Diff'] = (economic_data['LPGSMBPE Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)
    result['Natural Gasoline RV 20D Diff'] = (economic_data['LPGSMBNG Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)


    result['SPMLP Realized Variance'] = economic_data['SPMLP Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
    result['SPMLP RV vs S&P 500 RV'] = result['SPMLP Realized Variance'] - result['S&P500 Realized Variance']
    result['SPMLP Volatility Premium'] = iv - economic_data['SPMLP Index'].pct_change().pow(2).rolling(3).mean().pow(0.5) * 100 * np.sqrt(260)


    ### All series that need to be diffed
    SERIES_TO_PCT_CHG = [
        'SPX Index',
        'VIX Index',
        'RTY Index',
        'AMZ Index',
        'DXY Index',
        'SPMLP Index',
        'NG1 Comdty',
        'CL1 Comdty',
        'LPGSMBPE Index', #Ethane
        'LPGSMBPP Index', # Propane
        'LPGSMBNB Index', # Normal Butane Gasoline
        'LPGSMBIB Index', # Butane
        'LPGSMBNG Index', # Natural Gasoline
        'CRKS321C Index', # Crack Spread
        'LF98TRUU Index',
        'LUACTRUU Index',
        'LG30TRUU Index',
        'LEGATRUU Index',
        'LGTRTRUU Index',
        'LF94TRUU Index',
        'LGCPTRUH Index',
        'LGCPTRUU Index',
        'BEBGTRUU Index',
        'BASPTDSP Index',
        'ENASENA2 Index',
    ]

    COLUMN_NAMES = {
        'SPX Index': 'S&P 500',
        'VIX Index': 'VIX',
        'DXY Index': 'Dollar Index',
        'RTY Index': 'Russell 2000',
        'SPMLP Index': 'S&P MLP Index',
        'AMZ Index': 'AMZ Index',
        'CL1 Comdty': 'Crude Oil 1st',
        'NG1 Comdty': 'Natural Gas 1st',
        'USYC2Y10 Index': 'US 2yr-10yr Spread',
        'LPGSMBPE Index': 'Ethane', #Ethane
        'LPGSMBPP Index': 'Propane', # Propane
        'LPGSMBNB Index': 'Normal Butane', # Normal Butane Gasoline
        'LPGSMBIB Index': 'Butane', # Butane
        'LPGSMBNG Index': 'Natural Gasoline', # Natural Gasoline
        'CRKS321C Index': 'Crack Spread', # Crack Spread,
        'LF98TRUU Index': 'US Corporate HY Bonds',
        'LUACTRUU Index': 'US Corporate Bonds',
        'LG30TRUU Index': 'Global HY Bonds',
        'LEGATRUU Index': 'Global Aggregate Bonds',
        'LGTRTRUU Index': 'Global Aggregate Treasuries',
        'LF94TRUU Index': 'Global Inflation-Linked Bonds',
        'LGCPTRUH Index': 'Global Aggregate Corporate Bonds (Hedged USD)',
        'LGCPTRUU Index': 'Global Aggregate Corporate Bonds (Unhedged)',
        'BEBGTRUU Index': 'EM HY Bonds',
        'BASPTDSP Index': 'TED Spread',
        'ENASENA2 Index': 'WTI Crude 12M Strip',
        '.CL1-2 G Index': 'CL1-CL2 Spread',
        '.WIDOW G Index': 'NG Widowmaker',
        '.CL1DXYCO G Index': 'Oil-Dollar Correlation',
        '.CL/NG G Index': 'Oil/Natgas',
        '.CLO G Index': 'CLO Spread',
        '.CLSPR12 G Index': 'CL1-CL12 Spread',
        '.CL1-CL4 G Index': 'CL1-CL4 Spread'
    }

    SERIES_TO_DIFF = [
        'USYC2Y10 Index',
        '.CL1-2 G Index',
        '.CL1-2 G Index',
        '.WIDOW G Index'
    ]

    MOMENTUM_SERIES = {}
    for momentum_period in [1, 2, 3, 4, 5, 10, 20, 40, 50, 100, 200]:
        for series in SERIES_TO_PCT_CHG:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Momentum'] = series_data.pct_change().rolling(momentum_period).sum()
        for series in SERIES_TO_PCT_CHG:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Skew'] = series_data.pct_change().rolling(momentum_period).skew()
        for series in SERIES_TO_PCT_CHG:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Kurtosis'] = series_data.pct_change().rolling(momentum_period).kurt()

        for series in SERIES_TO_DIFF:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Difference'] = series_data.diff().rolling(momentum_period).sum()
        for series in SERIES_TO_DIFF:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Skew'] = series_data.diff().rolling(momentum_period).skew()
        for series in SERIES_TO_DIFF:
            series_data = economic_data[series].dropna()
            MOMENTUM_SERIES[f'{COLUMN_NAMES[series]} - {momentum_period}D Kurtosis'] = series_data.diff().rolling(momentum_period).kurt()
    MOMENTUM_SERIES = pd.DataFrame(MOMENTUM_SERIES)
    VARIANCE_SERIES = {}
    for vp in [1, 2, 3, 4, 5, 10, 20, 40, 50, 100, 200]:
        for series in SERIES_TO_PCT_CHG + SERIES_TO_DIFF:
            series_data = economic_data[series].dropna()
            VARIANCE_SERIES[f'{COLUMN_NAMES[series]} - {vp}D Realized Variance'] = series_data.pct_change().pow(2).rolling(vp).mean() * 100 * 260
    VARIANCE_SERIES = pd.DataFrame(VARIANCE_SERIES)
    data = pd.concat([MOMENTUM_SERIES, VARIANCE_SERIES], axis=1)
    data = pd.concat([data, pd.DataFrame(result)], axis=1).fillna(method='ffill', limit=10)
    scaler = COLUMN_SCALERS['standard']
    return {
        'raw': data,
        'scaled': scaler(data)
    }


class NeuralNetGlobalMacroForecaster(nn.Module):
    def __init__(self, in_features, n_output, n_hidden):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.predict(x)

def get_forecaster_network():
    return NeuralNetGlobalMacroForecaster(in_features=1250, n_hidden=64, n_output=275)

@dataclass
class GlobalMacroForecaster:
    """
    This is the forecaster for global macro.

    It will be trained along-side the risk forecaster and also the single name forecaster.

    Anywhere we forecast volatility and implied volatility, what we mean is forecast change
    in values.

    Different forecasters embedded in the main neural network. Target after arrows.
    a) Global Macro -> Different momentum values for different macro variables
    b) Commodities -> Different momentum, volatility, implied volatility, profitability of option sale.
    c) Bonds & Credit Markets -> Momentum, Volatility, Default Rate
    d) Global & US Equities -> Momentum, Volatility, Implied Volatility
    e) Energy Subsectors -> Momentum, Excess Returns, Ranking, Volatility, Implied Volatility
    f) Energy Rowe Factors -> Momentum, Excess Returns, Ranking, Volatility, Implied Volatility
    g) Midstream -> Momentum, Excess Returns, Ranking, Volatility, Implied Volatility
    h) Deep Factor Models -> Momentum, Excess Returns, Ranking, Volatility, Implied Volatility
    i) Tech stocks -> Momentum, Excess Returns, Ranking, Volatility, Implied Volatility


    Goal for model:
        Forecast Volatility Changes, Total Returns, Excess Returns for our universe

    This forecaster will be used as input to portfolio construction, risk management, and factor tilting portfolios.

    For each day, you will forecast the output necessary for computing portfolio, compute portfolio,
    That will be Apex V1:
        1) Global Macro + Midrtream Stock Selection Factor Model

    Apex v2 will start introducing additional datasets, and will also have a reinforcement learning agent.

    Then I'll combine all of these, aim for 100% turnover per year,
    """
    network: typing.Any = field(init=False)
    dataset: typing.Any = field(init=False)
    def __post_init__(self):
        self.network = get_forecaster_network()
        self.dataset = self.build_dataset()

    def train(self, num_epochs=10000, cuda=True, cuda_index=0, verbose=True, loss=torch.nn.MSELoss):
        net = self.network
        X_train = self.dataset['X']['train']
        X_val = self.dataset['X']['validation']
        y_train = self.dataset['y']['train']
        y_val = self.dataset['y']['validation']

        optimizer = Eve(net.parameters(), lr=1e-5)
        loss_func = loss()  # this is for regression mean squared loss
        if cuda:
            device = torch.device(f'cuda:{cuda_index}')
        else:
            device = torch.device('cpu')
        X = X_train.to(device)
        y = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        net = net.to(device)

        try:
            for t in range(num_epochs):
                def closure():
                    optimizer.zero_grad()
                    output = net(X)
                    output[torch.abs(y) <= 1e-8] = 0.0
                    loss = loss_func(output, y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)        # apply gradients

                """prediction = net(X)     # input x and predict based on x

                loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                """
                if t % 100 == 0 and verbose:
                    print('Train:', loss)
                    net = net.eval()
                    y_val_pred = net.predict(X_val)
                    loss = loss_func(y_val_pred, y_val)
                    print('Validation:', loss)
                    net = net.train()
        except KeyboardInterrupt:
            pass
        self.network = net
        return net

    def build_dataset(self):
        global_macro_data = get_global_macro_data()
        macro_data = global_macro_data['scaled']
        raw_data = global_macro_data['raw']
        momentum_columns = [x for x in macro_data.columns if 'Momentum' in x]
        mom_col_to_day = {x: int(re.match('.* - (?P<momentum_days>\d+)D .*', x).groupdict()['momentum_days']) for x in momentum_columns}
        targets = {}
        for col in mom_col_to_day:
            target_series = macro_data[col].shift(-mom_col_to_day[col])
            targets[col] = target_series

        targets = pd.DataFrame(targets).dropna(how='all')
        macro_data = macro_data.reindex(targets.index)
        X = torch.from_numpy(macro_data.iloc[:-400].fillna(0).values)
        y = torch.from_numpy(targets.iloc[:-400].fillna(0.0).values)
        X_val = torch.from_numpy(macro_data.iloc[-400:].fillna(0).values)
        y_val = torch.from_numpy(targets.iloc[-400:].values)

        return {
            'scaled_data': global_macro_data['scaled'],
            'raw_data': raw_data,
            'X': {
                'train': X,
                'validation': X_val
            },
            'y': {
                'train': y,
                'validation': y_val
            }
        }