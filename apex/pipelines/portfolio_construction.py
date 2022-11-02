import re
import time
import typing
import uuid
from collections import ChainMap, OrderedDict, defaultdict, deque
from pathlib import Path

import boltons as bs
import dogpile.cache as dc
import funcy as fy
import numba as nb

# Default imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy as sc
import pendulum

# Other
import toolz as tz
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as to
import torch.tensor as tt

# Others
from dataclasses import dataclass, field
import joblib
import pygmo as pg
import pyomo as po

from apex.pipelines.covariance import default_covariance



def score_portfolio_construction(dates, signal, market_data, methodology='risk_budget', minimum_securities=6, volatility_target=0.2):
    result = {}
    weights = None
    all_securities = set(market_data.columns.get_level_values(1).tolist())
    securities_signal = set(signal.loc[list(dates)].dropna(how='all', axis=1).columns.tolist())
    securities = sorted(all_securities.intersection(securities_signal))
    market_data = market_data.copy().loc[:dates[-1]]
    market_data.index = [x.strftime("%Y-%m-%d") for x in market_data.index]
    market_data = market_data.swaplevel(axis=1)[securities].swaplevel(axis=1).sort_index(axis=1)
    closes = market_data['px_last']
    returns = market_data['returns']

    for date in dates:
        dt = date.strftime('%Y-%m-%d')
        day_signal = signal.loc[dt].dropna()
        day_signal = day_signal.reindex(securities).fillna(0)
        day_signal[day_signal.abs() < 1e-5] = np.nan
        day_signal = day_signal.dropna()
        day_returns = returns.iloc[-512:-1][day_signal.index.tolist()]
        covariance = default_covariance(day_returns)
        weights = construct_portfolio(day_signal,
                                    covariance,
                                    methodology=methodology,
                                    minimum_securities=minimum_securities,
                                    volatility_target=volatility_target)
        weights[closes.loc[dt].isnull()] = np.nan
        result[dt] = weights.reindex(all_securities).fillna(0)
    result = pd.DataFrame(result).T
    result[result.abs() < 1e-7] = np.nan
    result = result.fillna(0)
    key = uuid.uuid4().hex
    filename = f'/mnt/data/experiments/alpha/compute/{key}.ending.{dt}.parquet'
    result.to_parquet(filename)
    return filename
