import concurrent.futures as cf
import datetime as dt
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

import boltons as bs
import dogpile.cache as dc
import funcy as fy
import numba as nb
# Default imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygmo as pg
import scipy as sc
import sklearn as sk
# Other
import toml
import toolz as tz
# Others
from dataclasses import dataclass, field
from scipy.stats import rankdata
from toolz import compose, curry, pipe

import pyomo as po
import statsmodels.api as sm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


def decay_linear(series, window):
    weights = np.arange(1, window + 1)[::-1]
    weights = weights / np.sum(weights)
    return series.rolling(window).apply(lambda x: np.dot(x, weights))

def signed_power(series, a):
    return series.pow(a) * np.sign(series)

def rolling_zscore(x, days):
    return (x - x.rolling(days).mean()) / x.rolling(days).std()

def rank_signal(result):
    result = result.copy()
    result = result.rank(axis=1, pct=True)
    result = result - 0.5
    return result

def scale(df):
    return df.divide(df.abs().sum(axis=1), axis=0)


@nb.jit
def timeseries_rank_fn(x):
    array = np.array(x)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks[-1] + 1


def call_market_alpha(market_alpha, market_data, **kwargs):
    return market_alpha(highs=market_data['px_high'].fillna(method='ffill', limit=2),
                  opens=market_data['px_open'].fillna(method='ffill', limit=2),
                  lows=market_data['px_low'].fillna(method='ffill', limit=2),
                  closes=market_data['px_last'].fillna(method='ffill', limit=2),
                  returns=market_data['returns'].fillna(0),
                  volumes=market_data['px_volume'].fillna(method='ffill', limit=2),
                  **kwargs)



def alpha_1(argmax_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - average_rank

    So basically, if returns < 0 I take the std dev of returns over past 20 days.
    Else I take the close.

    Then I square that, but keep the signs.

    Then I pick the day that maximizes it over the past 5 days.

    And rank it.
    """
    result = closes.copy()
    std_dev = returns.ewm(span=20).std()
    result[returns < 0] = std_dev[returns < 0]
    result = result.pow(2) * np.sign(result)
    result = result.dropna(how='all', axis=0).fillna(method='ffill')
    result = result.rolling(argmax_period).apply(lambda x: np.argmax(x))
    return result



def alpha_2(rolling_corr_period=6, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

     -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)

    The alpha is simply the correlation between the ranks of change in volume vs rank in (close - open)/open over the past 6 days.

    """
    log_volume_diff_rank = np.log(volumes).diff(2).rank(axis=1)
    open_close_rets_rank = ((closes - opens) / opens).rank(axis=1)

    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': log_volume_diff_rank[c], 'rank_id_rets': open_close_rets_rank[c]})
        col_res = col_df.rolling(rolling_corr_period).corr().loc[(slice(None), 'rank_id_rets'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -pd.DataFrame(result)



def alpha_3(rolling_corr_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    (-1 * correlation(rank(open), rank(volume), 10))

    The alpha is simply the correlation between the ranks of open vs ranks of volume

    """
    volume_rank = rank_signal(volumes)
    opens_rank = rank_signal(opens)

    result = {}
    for c in opens.columns:
        col_df = pd.DataFrame({'rank_vol': volume_rank[c], 'rank_opens': opens_rank[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'rank_opens'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = -col_res

    return rank_signal(pd.DataFrame(result))

def alpha_4(rolling_apply_period=9, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

     (-1 * Ts_Rank(rank(low), 9))
    """
    result = -rank_signal(lows).rolling(rolling_apply_period).apply(lambda x: rankdata(x)[-1])
    return result


def alpha_5(vwap_rolling_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))

    So. Lets assume VWAP = (open*0.4 + 0.1*high + 0.1*low + 0.4*close)/4 (seems ok to me)

    The first term is the rank of the open - average vwap over 10 days
    The second term is -1 * absolute value of the rank of close - vwap
    """
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows
    vwaps_mean = vwaps.ewm(span=vwap_rolling_period).mean()
    first_term = rank_signal(opens - vwaps_mean)
    second_term = -rank_signal(closes-vwaps).abs()
    return first_term * second_term


def alpha_6(rolling_corr_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    (-1 * correlation(open, volume, 10))
    """
    result = {}
    for c in opens.columns:
        col_df = pd.DataFrame({'open': opens[c], 'volume': volumes[c]})
        col_res = col_df.rolling(rolling_corr_period).corr().loc[(slice(None), 'volume'), 'open']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -pd.DataFrame(result)


def alpha_7(delta_close_period=7, rolling_apply_period=60, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))

    So. If adv < volume, we take the rank over the past 60 days of the changes in 7 day closes and multiply it by the last diff.
    Else -1
    """
    delta_close = closes.diff(delta_close_period)

    # First term: ts_rank(abs(delta(close, 7)), 60))
    first_term = -delta_close.abs().rolling(rolling_apply_period).apply(lambda x: rankdata(x)[-1])

    # Second term: sign(delta(close, 7))
    second_term = np.sign(delta_close)

    result = first_term * second_term
    return rank_signal(result)


def alpha_8(rolling_sum_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank (
        (sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)
    )
    """
    opens_5d_sum = opens.rolling(rolling_sum_period).sum()
    returns_5d_sum = returns.rolling(rolling_sum_period).sum()

    open_times_returns = opens_5d_sum * returns_5d_sum
    result = open_times_returns - open_times_returns.shift(rolling_sum_period * 2)
    return rank_signal(result)



def alpha_9(delta_close_period=5, rolling_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((0 < ts_min(delta(close, 1), 5)) ?

     delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?

                      delta(close, 1) : (-1 * delta(close, 1))))
    """
    delta_closes = closes.diff(delta_close_period)
    min_delta_closes = delta_closes.rolling(rolling_period).min()
    max_delta_closes = delta_closes.rolling(rolling_period).max()

    second_term = delta_closes.copy()
    second_term[max_delta_closes < 0] = second_term[max_delta_closes > 0]

    result = delta_closes.copy()
    result[min_delta_closes > 0] = second_term

    return -result

def alpha_10(rolling_period=4, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     rank(
         ts_min(delta(close, 1), 4) > 0 ? delta(close, 1) :
             ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))
         )
    """
    delta_closes = closes.diff()
    min_delta_closes = delta_closes.rolling(rolling_period).min()
    max_delta_closes = delta_closes.rolling(rolling_period).max()

    second_term = delta_closes.copy()
    second_term[max_delta_closes > 0] = -second_term[max_delta_closes > 0]
    result = delta_closes.copy()
    result[min_delta_closes > 0] = second_term

    return rank_signal(result)


def alpha_11(rolling_period=5, volume_period=2, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
        rank(delta(volume, 3))
    """
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows

    first_term = rank_signal((vwaps-closes).rolling(rolling_period).max())
    second_term = rank_signal((vwaps-closes).rolling(rolling_period).min())

    third_term = rank_signal(volumes.ewm(span=volume_period).mean().diff())

    return rank_signal(first_term + second_term + third_term)


def alpha_12(volume_diff=2, closes_diff=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
       (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    first_term = np.sign(volumes.diff(volume_diff))
    second_term = -(closes.diff(closes_diff))

    return rank_signal(first_term + second_term)

def alpha_13(rolling_cov_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
        (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    volume_rank = rank_signal(volumes)
    closes_rank = rank_signal(closes)

    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': volume_rank[c], 'rank_closes': closes_rank[c]})
        col_res = col_df.ewm(span=rolling_cov_period).cov().loc[(slice(None), 'rank_closes'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -rank_signal(pd.DataFrame(result))

def alpha_14(rolling_corr_period=10, return_mean_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
        (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
    """
    corr = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': volumes[c], 'rank_id_rets': opens[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'rank_id_rets'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        corr[c] = col_res
    corr = pd.DataFrame(corr)

    first_term = -rank_signal(returns.ewm(span=return_mean_period).mean().diff())
    return first_term * corr


def alpha_15(rolling_corr_period=8, rolling_mean_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
         -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    """
    rank_highs = rank_signal(highs)
    rank_volumes = rank_signal(volumes)

    result = {}
    for c in volumes.columns:
        col_df = pd.DataFrame({'high': rank_highs[c], 'vol': rank_volumes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = rank_signal(pd.DataFrame(result))

    return result.ewm(span=rolling_mean_period).mean()

def alpha_16(rolling_cov_period=8, rolling_mean_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
         -1 * sum(rank(cov(rank(high), rank(volume), 3)), 3)
    """
    rank_highs = rank_signal(highs)
    rank_volumes = rank_signal(volumes)

    result = {}
    for c in volumes.columns:
        col_df = pd.DataFrame({'high': rank_highs[c], 'vol': rank_volumes[c]})
        col_res = col_df.ewm(span=rolling_cov_period).cov().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = rank_signal(pd.DataFrame(result))

    return result.ewm(span=rolling_mean_period).mean()

def alpha_17(rolling_apply_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *

    rank(ts_rank((volume / adv20), 5))
    """
    adv = volumes.rolling(20).mean()

    first_term = -(rank_signal(closes.rolling(rolling_apply_period).apply(lambda x: rankdata(x)[-1]) + 0.5))
    second_term = rank_signal(closes.diff().diff())
    third_term = rank_signal((volumes/adv).rolling(rolling_apply_period).apply(lambda x: rankdata(-x)[-1]))
    result = first_term * second_term * third_term
    return result


def alpha_18(rolling_corr_period=10, rolling_std_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    """
    first_term = (closes - opens).abs().ewm(span=rolling_std_period).std() * np.sqrt(261)
    second_term = closes - opens
    third_term = {}

    for c in closes.columns:
        col_df = pd.DataFrame({'open': opens[c], 'close': closes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'open'), 'close']
        col_res.index = col_res.index.get_level_values(0)
        third_term[c] = col_res
    third_term = pd.DataFrame(third_term)
    result = -rank_signal(rank_signal(first_term) + rank_signal(second_term) + rank_signal(third_term))
    return result

def alpha_19(shift_period=7, momentum_period=250, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    """
    first_term = -np.sign(closes - closes.shift(shift_period) + closes.diff(shift_period))
    second_term = 1 + rank_signal(1+returns.ewm(span=momentum_period).mean())
    return first_term * second_term

def alpha_20(shift_period=1, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -
delay(low, 1)))
    """
    first_term = -rank_signal(opens - highs.shift(shift_period))
    snd_term = rank_signal(opens - closes.shift(shift_period))
    trd_term = rank_signal(opens - lows.shift(shift_period))
    return first_term * snd_term * trd_term


def alpha_21(rolling_mean_period=8, comparison_period=2, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     IF TRUE: (sum(close, 8) / 8) + stddev(close, 8) < (sum(close, 2) / 2)
     THEN: -1
     ELSE:
         IF TRUE: (sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))
         THEN: 1
         ELSE:
         IF (1 < (volume / adv20)) OR (volume / adv20) == 1)
         THEN 1
         ELSE -1
    """
    first_if_clause = closes.ewm(span=rolling_mean_period).mean() + closes.diff().ewm(span=rolling_mean_period).std() < closes.ewm(comparison_period).mean()
    second_if_clause = closes.ewm(span=rolling_mean_period).mean() - closes.diff().ewm(span=rolling_mean_period).std() > closes.ewm(comparison_period).mean()

    result = pd.DataFrame(1, index=closes.index, columns=closes.columns)
    result[second_if_clause] = -1
    result[first_if_clause] = 1
    return result
#############


def alpha_22(rolling_corr_period=10, diff_period=5, std_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
    """
    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'high': highs[c], 'vol': volumes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res

    first_term = -pd.DataFrame(first_term).diff(diff_period)
    second_term = rank_signal(returns.ewm(span=std_period).std())
    return first_term * second_term

def alpha_23(base_period=20, diff_period=2, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    IF sum(high, 20) / 20 < high
    THEN -1 * delta(high, 2)
    ELSE 0
    """
    first_term = highs.ewm(span=base_period).mean() < highs
    result = pd.DataFrame(0, index=highs.index, columns=highs.columns)
    result[first_term] = -highs.diff(diff_period)
    return result

def alpha_24(base_period=100, diff_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    IF  delta((sum(close, 100) / 100), 100)/delay(close, 100) <= 0.05
    THEN -1 * (close - ts_min(close, 100))
    ELSE -1 * delta(close, 3)
    """
    first_term = closes.ewm(span=100).mean().diff(base_period) / closes.shift(base_period) <= 0.05
    second_term = closes - closes.ewm(span=base_period).mean()
    third_term = closes.diff(diff_period)

    result = pd.DataFrame(0, index=closes.index, columns=closes.columns)
    result[first_term] = -1 * second_term
    result[~first_term] = -1 * third_term
    return result


def alpha_25(volume_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    adv = volumes.ewm(span=volume_period).mean()

    vwap = closes * 0.6 + 0.26*opens + 0.07 * lows + 0.07 * highs

    result = -returns * adv * vwap *((highs - closes))
    return rank_signal(result)


def alpha_26(ts_rank_period=5, rolling_corr_period=5, max_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    -ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    """
    first_ts_rank = volumes.rolling(ts_rank_period).apply(timeseries_rank_fn)
    second_ts_rank = highs.rolling(ts_rank_period).apply(timeseries_rank_fn)

    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'high': second_ts_rank[c], 'vol': first_ts_rank[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)
    return - first_term.rolling(max_period).max()

def alpha_27(rolling_corr_period=6, rolling_mean_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    IF rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)) > 0.5
    THEN -1
    ELSE 1
    """
    adv = volumes.rolling(20).mean()

    vwap = closes * 0.6 + 0.26*opens + 0.07 * lows + 0.07 * highs
    vwap_rank = rank_signal(vwap)
    volume_rank = rank_signal(volumes)

    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'vwap': vwap_rank[c], 'vol': volume_rank[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vol'), 'vwap']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res

    first_term = rank_signal(pd.DataFrame(first_term).ewm(span=rolling_mean_period).mean()) > 0.5

    result = pd.DataFrame(1, index=first_term.index, columns=first_term.columns)
    result[first_term] = -1
    return result

def alpha_28(rolling_corr_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     scale(
         (
             (correlation(adv20, low, 5) + ((high + low) / 2))
             - close
          )
        )
     scale = rescaling so that sum(abs(x)) = 1
    """
    adv = volumes.rolling(20).mean()

    vwap = closes * 0.6 + 0.26*opens + 0.07 * lows + 0.07 * highs
    vwap_rank = rank_signal(vwap)
    volume_rank = rank_signal(volumes)

    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'low': lows[c], 'vol': adv[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vol'), 'low']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)

    result = first_term + (highs + lows)*0.5 - closes
    result = result.divide(result.abs().sum(axis=1), axis=0)
    return result


def alpha_29(base_period=23, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    min(product(
        rank(rank(scale(log(sum(ts_min(rank(rank((
        -1 * rank(delta((close - 1), 5))

        ))), 2), 1))))), 1), 5)
    + ts_rank(delay((-1 * returns), 6), 5)

    I modified it.
    """
    first_term = rank_signal(rank_signal(scale(rank_signal(-rank_signal((closes-1).diff(2))).rolling(2).min()))).rolling(3).max()

    return rank_signal(first_term)


def alpha_30(num_shifts=5, rolling_volume_period=10, rolling_mean_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3))))))
     *
     sum(volume, 5)) / sum(volume, 20)

    """
    first_term = np.sign(closes - closes.shift(1))
    for i in range(1, num_shifts):
        first_term += np.sign(closes.shift(i) - closes.shift(i + 1))

    second_term = volumes/volumes.rolling(rolling_volume_period).sum()

    return rank_signal(-(first_term * second_term).ewm(span=rolling_mean_period).mean())

def alpha_31(base_period=10, close_diff_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      rank(decay_linear(-rank(delta(close, 10)), 10))

      + rank((-1 * delta(close, 3)))

      + sign(correlation(adv20, low, 12))

    Not implementing last part. Don't think its necessary.

    """
    first_term = rank_signal(decay_linear(-rank_signal(closes.diff(base_period)), base_period))
    second_term = rank_signal(-closes.diff(close_diff_period))

    return rank_signal(-first_term + second_term)

def alpha_32(base_close_period=7, close_delay_period=5, rolling_corr_period=230, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (
          scale(
              ((sum(close, 7) / 7) - close)
               ) + (20 * scale(correlation(vwap, delay(close, 5), 230)))
        )
    """

    adv = volumes.rolling(20).mean()

    first_term = scale(closes.rolling(base_close_period).mean() - closes)

    vwap = closes * 0.6 + 0.26*opens + 0.07 * lows + 0.07 * highs

    second_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'vwap': vwap[c], 'close_delayed': closes[c].shift(close_delay_period)})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'vwap'), 'close_delayed']
        col_res.index = col_res.index.get_level_values(0)
        second_term[c] = col_res
    second_term = 20*scale(pd.DataFrame(second_term))

    return first_term + second_term

def alpha_33(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank((-1 * ((1 - (open / close))^1)))
    Why power of 1?
    """
    return rank_signal((-(1-opens/closes)))

def alpha_34(fst_std_period=3, snd_std_period=5, diff_period=1, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    """
    first_term = 1+rank_signal(returns.rolling(fst_std_period).std()/returns.rolling(snd_std_period).std())
    second_term = 1-rank_signal(closes.diff(diff_period))

    return rank_signal(first_term + second_term)

def alpha_35(base_period=32, rolling_mean_period=3, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    """
    first_term = volumes.rolling(base_period).apply(timeseries_rank_fn)
    third_term = 1 - returns.rolling(base_period).apply(timeseries_rank_fn)
    return rank_signal(rank_signal(first_term).ewm(span=rolling_mean_period).mean() + rank_signal(third_term))


def alpha_36(rolling_corr_period=6, rolling_min_period=16, adv_period=180, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    """
    adv180 = volumes.rolling(adv_period).mean()
    vwaps = (highs + lows) / 2

    first_term = rank_signal(vwaps - vwaps.rolling(rolling_min_period).min())

    snd_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': vwaps[c], 'rank_first': adv180[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'rank_first'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        snd_term[c] = col_res
    snd_term = rank_signal(pd.DataFrame(snd_term))
    return (first_term < snd_term).astype(int)


def alpha_37(shift_period=1, rolling_corr_period=200, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    """
    first_term_corr = (opens-closes).shift(shift_period)

    first_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'first_term': first_term_corr[c], 'close': closes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'close'), 'first_term']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)
    second_term = opens-closes
    return rank_signal(rank_signal(first_term) + rank_signal(second_term))

def alpha_38(rolling_apply_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    """
    first_term = rank_signal(-closes.rolling(rolling_apply_period).apply(timeseries_rank_fn))
    second_term = rank_signal(closes/opens)
    return first_term * second_term

def alpha_39(delta_close_period=7, decay_linear_period=9, momentum_calc_period=50, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    """
    adv = volumes.rolling(20).mean()

    first_term = -rank_signal(closes.diff(delta_close_period) * (1-rank_signal(decay_linear(volumes/adv, decay_linear_period))))
    second_term = (1+rank_signal(returns.ewm(span=momentum_calc_period).mean() * momentum_calc_period))
    return first_term * second_term

def alpha_40(std_dev_period=10, rolling_corr_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    """
    first_term = -rank_signal(highs.ewm(span=std_dev_period).std())
    second_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'high': highs[c], 'volume': volumes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'high'), 'volume']
        col_res.index = col_res.index.get_level_values(0)
        second_term[c] = col_res
    second_term = pd.DataFrame(second_term)
    return -first_term*second_term

def alpha_41(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (((high * low)^0.5) - vwap)
    """
    adv = volumes.rolling(20).mean()
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows

    return rank_signal((highs*lows).pow(0.5) - vwaps)


def alpha_42(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (rank((vwap - close)) / rank((vwap + close)))
    """
    adv = volumes.rolling(20).mean()
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows
    signal = rank_signal((rank_signal(vwaps - closes) + 0.5) / (rank_signal(vwaps + closes) + 0.5)).fillna(0.0)
    return signal

def alpha_43(vol_to_adv_period=20, close_diff_period=7, close_rolling_apply_period=8, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    """
    adv = volumes.rolling(20).mean()

    first_term = (volumes/adv).rolling(vol_to_adv_period).apply(timeseries_rank_fn)
    second_term = (-closes.diff(close_diff_period)).rolling(close_rolling_apply_period).apply(timeseries_rank_fn)

    return rank_signal(first_term * second_term)


def alpha_44(base_period=20, base_delay=5, rolling_corr_period=2, fst_period=5, snd_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    """
    first_term = rank_signal(closes.shift(base_delay).ewm(span=base_period).mean())

    second_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'close': highs[c], 'volume': volumes[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'close'), 'volume']
        col_res.index = col_res.index.get_level_values(0)
        second_term[c] = col_res
    second_term = pd.DataFrame(second_term)

    third_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'close_fst': closes[c].ewm(span=fst_period).mean(),
                               'close_snd': closes[c].ewm(span=snd_period).mean()})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'close_fst'), 'close_snd']
        col_res.index = col_res.index.get_level_values(0)
        third_term[c] = col_res
    third_term = rank_signal(pd.DataFrame(third_term))

    return first_term * second_term * third_term

def alpha_45(base_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    Actually 44

    (-1 * correlation(high, rank(volume), 5))
    """
    first_term = {}
    volume_rank = rank_signal(volumes)
    for c in highs.columns:
        corr_df = pd.DataFrame({'highs': highs[c], 'volume_rank': volume_rank[c]})
        col_res = corr_df.ewm(span=base_period).corr().loc[(slice(None), 'highs'), 'volume_rank']
        col_res.index = col_res.index.droplevel(1)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)

    return first_term


def alpha_46(base_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """

    IF ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)))
    THEN -1
    ELSE IF (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0)
        THEN 1
        ELSE ((-1 * 1) * (close - delay(close, 1)))))
    """
    adv = volumes.rolling(20).mean()

    first_term = -((closes.shift(base_period) - closes.shift(base_period/2))/base_period/2 - (closes.shift(base_period/2) - closes)/base_period/2 > 0.25).astype(int)
    second_term = ((closes.shift(base_period) - closes.shift(base_period/2))/base_period/2 - (closes.shift(base_period/2) - closes)/base_period/2 < 0).astype(int)
    second_term[second_term == 0] = - (closes - closes.shift(1))[second_term == 0]
    first_term[first_term == 0] = second_term[first_term == 0]
    return first_term

def alpha_47(adv_period=200, base_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    Alpha#47:
    ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    """
    adv = volumes.rolling(adv_period).mean()

    vwaps = (highs + lows)/2

    first_term = rank_signal(1/closes)*volumes/adv
    second_term = highs * rank_signal(highs-closes)/highs.rolling(base_period).mean()
    third_term = rank_signal(vwaps-vwaps.shift(base_period))
    res = first_term * second_term - third_term
    return res



def alpha_48(rolling_apply_period=10, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    """
    first_term = (closes - lows) - (highs - closes)
    first_term = first_term / ((highs - lows) * volumes)
    first_term = scale(rank_signal(first_term))
    second_term = scale(rank_signal(closes.rolling(rolling_apply_period).apply(lambda x: np.argmax(x))))
    return rank_signal(first_term * second_term)


def alpha_49(base_period=10, cutoff=0.1, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    first_term = (closes.shift(base_period*2) - closes.shift(base_period))/base_period - (closes.shift(base_period) - closes)/base_period
    result = first_term.copy()
    result[first_term < -cutoff] = 1
    result[first_term >= -cutoff] = -(closes - closes.shift(1))
    return result

def alpha_50(rolling_corr_period=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    """
    vwaps = (highs + lows + closes)/3.0

    rank_vol = rank_signal(volumes)
    rank_vwap = rank_signal(vwaps)
    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': rank_vol[c], 'rank_vwaps': vwaps[c]})
        col_res = col_df.ewm(span=rolling_corr_period).corr().loc[(slice(None), 'rank_vwaps'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    result = pd.DataFrame(result)
    return result

def alpha_51(base_period=10, cutoff=0.05, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    first_term = (closes.shift(base_period*2) - closes.shift(base_period))/base_period - (closes.shift(base_period) - closes)/base_period
    result = (first_term < cutoff).astype(int)
    result[result == 0] = -(closes - closes.shift(1))
    return result

def alpha_52(base_period=5, momentum_slow_period=240, momentum_fast_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    """
    first_term = -lows.rolling(base_period).min() + lows.rolling(base_period).min().shift(base_period)
    second_term = rank_signal((returns.fillna(0).rolling(momentum_slow_period).sum() - returns.fillna(0).rolling(momentum_fast_period).sum())/(momentum_slow_period - momentum_fast_period))
    ts_rank_vol = volumes.rolling(base_period).apply(timeseries_rank_fn)
    return first_term * second_term * ts_rank_vol


def alpha_53(diff_period=9, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    """
    first_term = (closes - lows) - (highs-closes)
    first_term = first_term / (closes - lows)

    result = -(first_term.replace([np.inf, -np.inf], np.nan).fillna(method='ffill', limit=1)).diff(diff_period)
    return result


def alpha_54(base_power_exponent=5, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    """
    first_term = (lows - closes) * (opens.pow(base_power_exponent))
    second_term = (lows - highs) * (closes.pow(base_power_exponent))

    return -(first_term / second_term)

def alpha_55(base_period=6, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (-1 * correlation(
         rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))),
         rank(volume), 6))
    """
    first_term = rank_signal((closes - lows.rolling(base_period * 2).min())/(highs.rolling(base_period * 2).max() - lows.rolling(base_period * 2).min()))
    second_term = rank_signal(volumes)

    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': second_term[c], 'rank_first': first_term[c]})
        col_res = col_df.ewm(span=base_period).corr().loc[(slice(None), 'rank_first'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = pd.DataFrame(result)

    return -result


def alpha_56(slow_period=10, fast_period=2, fast_period_offset=1, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
      gonna change from cap to closes * volumes
    """
    first_term = rank_signal(returns.rolling(slow_period).sum()/returns.rolling(fast_period).sum().rolling(fast_period+fast_period_offset).sum())
    second_term = rank_signal(returns * closes * volumes)
    return -(first_term * second_term)

def alpha_57(base_period=5, rolling_apply_period=20, highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      ((0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))

    """
    vwaps = (highs + lows)/2
    first_term = closes - vwaps
    second_term = decay_linear(rank_signal(closes.rolling(rolling_apply_period).apply(lambda x: np.argmax(x))), base_period)

    return -(first_term / second_term)


# In[184]:



def alpha_normalization_ls(alpha_data):
    alpha_data = alpha_data.subtract(alpha_data.mean(axis=1), axis=0)
    alpha_data = alpha_data.divide(alpha_data.abs().sum(axis=1), axis=0)
    return alpha_data

def alpha_normalization_lo(alpha_data):
    alpha_data = alpha_data.rank(axis=1)
    alpha_data = alpha_data.divide(alpha_data.abs().sum(axis=1), axis=0)
    return alpha_data

@dataclass
class ApexAlphaManager:
    market_alphas: dict = field(default_factory=dict)
    alphas_by_family: dict = field(default_factory=dict)
    def __post_init__(self):
        self.build_market_alpha_families()

    def build_market_alpha_families(self):
        # Number of alphas:
        self.alphas_by_family['a1'] = self.build_a1_family()
        self.alphas_by_family['a2'] = self.build_a2_family()
        self.alphas_by_family['a3'] = self.build_a3_family()
        self.alphas_by_family['a4'] = self.build_a4_family()
        self.alphas_by_family['a5'] = self.build_a5_family()
        self.alphas_by_family['a6'] = self.build_a6_family()
        self.alphas_by_family['a7'] = self.build_a7_family()
        self.alphas_by_family['a8'] = self.build_a8_family()
        self.alphas_by_family['a9'] = self.build_a9_family()
        self.alphas_by_family['a10'] = self.build_a10_family()
        self.alphas_by_family['a11'] = self.build_a11_family()
        self.alphas_by_family['a12'] = self.build_a12_family()
        self.alphas_by_family['a13'] = self.build_a13_family()
        self.alphas_by_family['a14'] = self.build_a14_family()
        self.alphas_by_family['a15'] = self.build_a15_family()
        self.alphas_by_family['a16'] = self.build_a16_family()
        self.alphas_by_family['a17'] = self.build_a17_family()
        self.alphas_by_family['a18'] = self.build_a18_family()
        self.alphas_by_family['a19'] = self.build_a19_family()
        self.alphas_by_family['a20'] = self.build_a20_family()
        self.alphas_by_family['a21'] = self.build_a21_family()
        self.alphas_by_family['a22'] = self.build_a22_family()
        self.alphas_by_family['a23'] = self.build_a23_family()
        self.alphas_by_family['a24'] = self.build_a24_family()
        self.alphas_by_family['a25'] = self.build_a25_family()
        self.alphas_by_family['a26'] = self.build_a26_family()
        self.alphas_by_family['a27'] = self.build_a27_family()
        self.alphas_by_family['a28'] = self.build_a28_family()
        self.alphas_by_family['a29'] = self.build_a29_family()
        self.alphas_by_family['a30'] = self.build_a30_family()
        self.alphas_by_family['a31'] = self.build_a31_family()
        self.alphas_by_family['a32'] = self.build_a32_family()
        self.alphas_by_family['a33'] = self.build_a33_family()
        self.alphas_by_family['a41'] = self.build_a41_family()
        self.alphas_by_family['a42'] = self.build_a42_family()
        self.alphas_by_family['a34'] = self.build_a34_family()
        self.alphas_by_family['a35'] = self.build_a35_family()
        self.alphas_by_family['a36'] = self.build_a36_family()
        self.alphas_by_family['a37'] = self.build_a37_family()
        self.alphas_by_family['a38'] = self.build_a38_family()
        self.alphas_by_family['a39'] = self.build_a39_family()
        self.alphas_by_family['a40'] = self.build_a40_family()
        self.alphas_by_family['a43'] = self.build_a43_family()
        self.alphas_by_family['a44'] = self.build_a44_family()
        self.alphas_by_family['a45'] = self.build_a45_family()
        self.alphas_by_family['a46'] = self.build_a46_family()
        self.alphas_by_family['a47'] = self.build_a47_family()
        self.alphas_by_family['a48'] = self.build_a48_family()
        self.alphas_by_family['a49'] = self.build_a49_family()
        self.alphas_by_family['a50'] = self.build_a50_family()
        self.alphas_by_family['a51'] = self.build_a51_family()
        self.alphas_by_family['a52'] = self.build_a52_family()
        self.alphas_by_family['a53'] = self.build_a53_family()
        self.alphas_by_family['a54'] = self.build_a54_family()
        self.alphas_by_family['a55'] = self.build_a55_family()
        self.alphas_by_family['a56'] = self.build_a56_family()
        self.alphas_by_family['a57'] = self.build_a57_family()

    def build_a1_family(self):
        result = {}
        for argmax_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_1__argmax_period={argmax_period}'
            alpha_fn = curry(alpha_1, argmax_period=argmax_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a2_family(self):
        result = {}
        for rolling_corr_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_2__rolling_corr_period={rolling_corr_period}'
            alpha_fn = curry(alpha_2, rolling_corr_period=rolling_corr_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a3_family(self):
        result = {}
        for rolling_corr_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_3__rolling_corr_period={rolling_corr_period}'
            alpha_fn = curry(alpha_3, rolling_corr_period=rolling_corr_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a4_family(self):
        result = {}
        for rolling_apply_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_4__rolling_apply_period={rolling_apply_period}'
            alpha_fn = curry(alpha_4, rolling_apply_period=rolling_apply_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a5_family(self):
        result = {}
        for vwap_rolling_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_5__vwap_rolling_period={vwap_rolling_period}'
            alpha_fn = curry(alpha_5, vwap_rolling_period=vwap_rolling_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a6_family(self):
        result = {}
        for rolling_corr_period in np.arange(2, 41, 4):
            alpha_name = f'alpha_6__rolling_corr_period={rolling_corr_period}'
            alpha_fn = curry(alpha_6, rolling_corr_period=rolling_corr_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a7_family(self):
        result = {}
        for rolling_apply_period in np.arange(10, 150, 10):
            for delta_close_period in np.arange(2, 20, 4):
                alpha_name = f'alpha_7__rap={rolling_apply_period}_dcp={delta_close_period}'
                alpha_fn = curry(alpha_7, rolling_apply_period=rolling_apply_period, delta_close_period=delta_close_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a8_family(self):
        result = {}
        for rolling_sum_period in np.arange(2, 24, 4):
            alpha_name = f'alpha_8__rolling_sum_period={rolling_sum_period}'
            alpha_fn = curry(alpha_8, rolling_sum_period=rolling_sum_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a9_family(self):
        result = {}
        for rolling_period in np.arange(10, 150, 10):
            for delta_close_period in np.arange(5, 45, 5):
                alpha_name = f'alpha_9__rp={rolling_period}_dcp={delta_close_period}'
                alpha_fn = curry(alpha_9, rolling_period=rolling_period, delta_close_period=delta_close_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a10_family(self):
        result = {}
        for rolling_period in np.arange(2, 42, 4):
            alpha_name = f'alpha_10__rolling_period={rolling_period}'
            alpha_fn = curry(alpha_10, rolling_period=rolling_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a11_family(self):
        result = {}
        for rolling_period in np.arange(2, 42, 4):
            for volume_period in np.arange(5, 50, 5):
                alpha_name = f'alpha_11__rp={rolling_period}_vp={volume_period}'
                alpha_fn = curry(alpha_11, rolling_period=rolling_period, volume_period=volume_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a12_family(self):
        result = {}
        for volume_diff in np.arange(2, 15, 2):
            for closes_diff in np.arange(2, 15, 2):
                alpha_name = f'alpha_12__vd={volume_diff}_cd={closes_diff}'
                alpha_fn = curry(alpha_12, volume_diff=volume_diff, closes_diff=closes_diff)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a13_family(self):
        result = {}
        for rolling_cov_period in np.arange(5, 105, 20):
            alpha_name = f'alpha_13__rolling_cov_period={rolling_cov_period}'
            alpha_fn = curry(alpha_13, rolling_cov_period=rolling_cov_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a14_family(self):
        result = {}
        for rolling_corr_period in np.arange(10, 100, 20):
            for return_mean_period in np.arange(2, 42, 4):
                alpha_name = f'alpha_14__rcp={rolling_corr_period}_rmp={return_mean_period}'
                alpha_fn = curry(alpha_14, rolling_corr_period=rolling_corr_period, return_mean_period=return_mean_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result


    def build_a15_family(self):
        result = {}
        for rolling_corr_period in np.arange(10, 100, 20):
            for rolling_mean_period in np.arange(2, 42, 4):
                alpha_name = f'alpha_15__rcp={rolling_corr_period}_rmp={rolling_mean_period}'
                alpha_fn = curry(alpha_15, rolling_corr_period=rolling_corr_period, rolling_mean_period=rolling_mean_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result


    def build_a16_family(self):
        result = {}
        for rolling_cov_period in np.arange(10, 100, 20):
            for rolling_mean_period in np.arange(2, 42, 4):
                alpha_name = f'alpha_16__rcp={rolling_cov_period}_rmp={rolling_mean_period}'
                alpha_fn = curry(alpha_16, rolling_cov_period=rolling_cov_period, rolling_mean_period=rolling_mean_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a17_family(self):
        result = {}
        for rolling_apply_period in np.arange(5, 105, 20):
            alpha_name = f'alpha_17__rolling_apply_period={rolling_apply_period}'
            alpha_fn = curry(alpha_17, rolling_apply_period=rolling_apply_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result


    def build_a18_family(self):
        result = {}
        for rolling_corr_period in np.arange(4, 52, 4):
            for rolling_std_period in np.arange(5, 45, 5):
                alpha_name = f'alpha_18__rcp={rolling_corr_period}_rstdp={rolling_std_period}'
                alpha_fn = curry(alpha_18, rolling_corr_period=rolling_corr_period, rolling_std_period=rolling_std_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result


    def build_a19_family(self):
        result = {}
        for shift_period in np.arange(0, 20, 4):
            for momentum_period in np.arange(20, 400, 20):
                alpha_name = f'alpha_19__sp={shift_period}_mp={momentum_period}'
                alpha_fn = curry(alpha_19, shift_period=shift_period, momentum_period=momentum_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a20_family(self):
        result = {}
        for shift_period in np.arange(1, 26, 4):
            alpha_name = f'alpha_20__shift_period={shift_period}'
            alpha_fn = curry(alpha_20, shift_period=shift_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a21_family(self):
        result = {}
        for rolling_mean_period in np.arange(2, 62, 10):
            for comparison_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_21__rmp={rolling_mean_period}_cp={comparison_period}'
                alpha_fn = curry(alpha_21, rolling_mean_period=rolling_mean_period, comparison_period=comparison_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a22_family(self):
        result = {}
        for rolling_corr_period in np.arange(2, 62, 10):
            for diff_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_22__rcp={rolling_corr_period}_dp={diff_period}'
                alpha_fn = curry(alpha_22, rolling_corr_period=rolling_corr_period, diff_period=diff_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a23_family(self):
        result = {}
        for base_period in np.arange(5, 205, 20):
            for diff_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_23__bp={base_period}_dp={diff_period}'
                alpha_fn = curry(alpha_23, base_period=base_period, diff_period=diff_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a24_family(self):
        result = {}
        for base_period in np.arange(5, 205, 20):
            for diff_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_24__bp={base_period}_dp={diff_period}'
                alpha_fn = curry(alpha_24, base_period=base_period, diff_period=diff_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a25_family(self):
        result = {}
        for volume_period in np.arange(1, 26, 4):
            alpha_name = f'alpha_25__volume_period={volume_period}'
            alpha_fn = curry(alpha_25, volume_period=volume_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a26_family(self):
        result = {}
        for ts_rank_period in np.arange(5, 20, 5):
            for rolling_corr_period in np.arange(5, 50, 10):
                for max_period in np.arange(1, 10, 2):
                    alpha_name = f'alpha_26__tsrp={ts_rank_period}_mp={max_period}_rcp={rolling_corr_period}'
                    alpha_fn = curry(alpha_26, ts_rank_period=ts_rank_period, rolling_corr_period=rolling_corr_period, max_period=max_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a27_family(self):
        result = {}
        for rolling_corr_period in np.arange(5, 100, 20):
            for rolling_mean_period in np.arange(1, 15, 4):
                alpha_name = f'alpha_27__rcp={rolling_corr_period}_dp={rolling_mean_period}'
                alpha_fn = curry(alpha_27, rolling_corr_period=rolling_corr_period, rolling_mean_period=rolling_mean_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a28_family(self):
        result = {}
        for rolling_corr_period in np.arange(1, 26, 4):
            alpha_name = f'alpha_28__rolling_corr_period={rolling_corr_period}'
            alpha_fn = curry(alpha_28, rolling_corr_period=rolling_corr_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a29_family(self):
        result = {}
        for base_period in np.arange(1, 26, 4):
            alpha_name = f'alpha_29__base_period={base_period}'
            alpha_fn = curry(alpha_29, base_period=base_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a30_family(self):
        result = {}
        for num_shifts in np.arange(3, 20, 4):
            for rolling_volume_period in np.arange(5, 25, 5):
                for rolling_mean_period in np.arange(1, 15, 5):
                    alpha_name = f'alpha_30__ns={num_shifts}_rmp={rolling_mean_period}_rvp={rolling_volume_period}'
                    alpha_fn = curry(alpha_30, num_shifts=num_shifts, rolling_volume_period=rolling_volume_period, rolling_mean_period=rolling_mean_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a31_family(self):
        result = {}
        for base_period in np.arange(5, 105, 20):
            for close_diff_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_31__bp={base_period}_cdp={close_diff_period}'
                alpha_fn = curry(alpha_31, base_period=base_period, close_diff_period=close_diff_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result


    def build_a32_family(self):
        result = {}
        for base_close_period in np.arange(2, 10, 2):
            for rolling_corr_period in np.arange(10, 250, 50):
                for close_delay_period in np.arange(1, 5, 2):
                    alpha_name = f'alpha_32__bcp={base_close_period}_cdp={close_delay_period}_rcp={rolling_corr_period}'
                    alpha_fn = curry(alpha_32, base_close_period=base_close_period, rolling_corr_period=rolling_corr_period, close_delay_period=close_delay_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a33_family(self):
        result = {}
        self.market_alphas['alpha_33'] = alpha_33

    def build_a41_family(self):
        result = {}
        self.market_alphas['alpha_41'] = alpha_41

    def build_a42_family(self):
        result = {}
        self.market_alphas['alpha_42'] = alpha_42

    def build_a34_family(self):
        result = {}
        for fst_std_period in np.arange(5, 250, 50):
            for snd_std_period in np.arange(5, 100, 20):
                for diff_period in np.arange(1, 10, 2):
                    alpha_name = f'alpha_34__bcp={fst_std_period}_cdp={diff_period}_rcp={snd_std_period}'
                    alpha_fn = curry(alpha_34, fst_std_period=fst_std_period, snd_std_period=snd_std_period, diff_period=diff_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a35_family(self):
        result = {}
        for base_period in np.arange(5, 105, 20):
            for rolling_mean_period in np.arange(1, 15, 3):
                alpha_name = f'alpha_35__bp={base_period}_rmp={rolling_mean_period}'
                alpha_fn = curry(alpha_35, base_period=base_period, rolling_mean_period=rolling_mean_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a36_family(self):
        result = {}
        for rolling_corr_period in np.arange(5, 20, 5):
            for adv_period in np.arange(10, 250, 50):
                for rolling_min_period in np.arange(5, 40, 10):
                    alpha_name = f'alpha_36__rmp={rolling_min_period}_rcp={rolling_corr_period}_advp={adv_period}'
                    alpha_fn = curry(alpha_36, rolling_min_period=rolling_min_period, adv_period=adv_period, rolling_corr_period=rolling_corr_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a37_family(self):
        result = {}
        for rolling_corr_period in np.arange(5, 105, 20):
            for shift_period in np.arange(1, 15, 2):
                alpha_name = f'alpha_37__sp={shift_period}_rcp={rolling_corr_period}'
                alpha_fn = curry(alpha_37, shift_period=shift_period, rolling_corr_period=rolling_corr_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a38_family(self):
        result = {}
        for rolling_apply_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_38__rolling_apply_period={rolling_apply_period}'
            alpha_fn = curry(alpha_38, rolling_apply_period=rolling_apply_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a39_family(self):
        result = {}
        for delta_close_period in np.arange(2, 10, 2):
            for decay_linear_period in np.arange(3, 25, 5):
                for momentum_calc_period in np.arange(5, 250, 50):
                    alpha_name = f'alpha_39__mcp={momentum_calc_period}_dcp={delta_close_period}_dlp={decay_linear_period}'
                    alpha_fn = curry(alpha_39, momentum_calc_period=momentum_calc_period, decay_linear_period=decay_linear_period, delta_close_period=delta_close_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a40_family(self):
        result = {}
        for std_dev_period in np.arange(5, 105, 40):
            for rolling_corr_period in np.arange(5, 105, 40):
                alpha_name = f'alpha_40__rcp={rolling_corr_period}_sdp={std_dev_period}'
                alpha_fn = curry(alpha_40, rolling_corr_period=rolling_corr_period, std_dev_period=std_dev_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a43_family(self):
        result = {}
        for close_rolling_apply_period in np.arange(2, 26, 5):
            for close_diff_period in np.arange(3, 25, 5):
                for vol_to_adv_period in np.arange(5, 250, 50):
                    alpha_name = f'alpha_43__mcp={vol_to_adv_period}_dcp={close_rolling_apply_period}_dlp={close_diff_period}'
                    alpha_fn = curry(alpha_43, vol_to_adv_period=vol_to_adv_period, close_diff_period=close_diff_period, close_rolling_apply_period=close_rolling_apply_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a44_family(self):
        result = {}
        for base_period in np.arange(5, 40, 10):
            for base_delay in np.arange(5, 40, 10):
                for rolling_corr_period in np.arange(5, 50, 10):
                    for fst_period in np.arange(5, 55, 10):
                        snd_period = fst_period * 4
                        alpha_name = f'alpha_44__fp={fst_period}_bp={base_period}_rcp={rolling_corr_period}_bdp={base_delay}'
                        alpha_fn = curry(alpha_44, base_period=base_period,
                            base_delay=base_delay,
                            fst_period=fst_period,
                            snd_period=snd_period,
                            rolling_corr_period=rolling_corr_period)
                        self.market_alphas[alpha_name] = alpha_fn
                        result[alpha_name] = alpha_fn
        return result

    def build_a45_family(self):
        result = {}
        for base_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_45__base_period={base_period}'
            alpha_fn = curry(alpha_45, base_period=base_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a46_family(self):
        result = {}
        for base_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_46__base_period={base_period}'
            alpha_fn = curry(alpha_46, base_period=base_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a47_family(self):
        result = {}
        for adv_period in np.arange(5, 205, 20):
            for base_period in np.arange(2, 25, 4):
                alpha_name = f'alpha_47__bp={base_period}_advp={adv_period}'
                alpha_fn = curry(alpha_47, base_period=base_period, adv_period=adv_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a48_family(self):
        result = {}
        for rolling_apply_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_48__rolling_apply_period={rolling_apply_period}'
            alpha_fn = curry(alpha_48, rolling_apply_period=rolling_apply_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result

    def build_a49_family(self):
        result = {}
        for cutoff in np.arange(0.0, 0.5, 0.1):
            for base_period in np.arange(2, 25, 4):
                alpha_name = f'alpha_49__bp={base_period}_cutoff={cutoff}'
                alpha_fn = curry(alpha_49, base_period=base_period, cutoff=cutoff)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result

    def build_a50_family(self):
        result = {}
        for rolling_corr_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_50__rolling_corr_period={rolling_corr_period}'
            alpha_fn = curry(alpha_50, rolling_corr_period=rolling_corr_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result


    def build_a51_family(self):
        result = {}
        for cutoff in np.arange(0.0, 0.5, 0.1):
            for base_period in np.arange(2, 25, 4):
                alpha_name = f'alpha_51__bp={base_period}_advp={cutoff}'
                alpha_fn = curry(alpha_51, base_period=base_period, cutoff=cutoff)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result


    def build_a52_family(self):
        result = {}
        for base_period in np.arange(2, 24, 5):
            for momentum_fast_period in np.arange(4, 52, 10):
                for momentum_slow_period in np.arange(50, 250, 10):
                    alpha_name = f'alpha_52__mcp={momentum_slow_period}_dcp={base_period}_dlp={momentum_fast_period}'
                    alpha_fn = curry(alpha_52, momentum_slow_period=momentum_slow_period, momentum_fast_period=momentum_fast_period, base_period=base_period)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result


    def build_a53_family(self):
        result = {}
        for diff_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_53__diff_period={diff_period}'
            alpha_fn = curry(alpha_53, diff_period=diff_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result


    def build_a54_family(self):
        result = {}
        for base_power_exponent in np.arange(2, 8, 1):
            alpha_name = f'alpha_54__base_power_exponent={base_power_exponent}'
            alpha_fn = curry(alpha_54, base_power_exponent=base_power_exponent)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result


    def build_a55_family(self):
        result = {}
        for base_period in np.arange(2, 26, 4):
            alpha_name = f'alpha_55__base_period={base_period}'
            alpha_fn = curry(alpha_55, base_period=base_period)
            self.market_alphas[alpha_name] = alpha_fn
            result[alpha_name] = alpha_fn
        return result


    def build_a56_family(self):
        result = {}
        for fast_period_offset in np.arange(1, 5, 1):
            for fast_period in np.arange(2, 20, 4):
                for slow_period in np.arange(20, 100, 20):
                    alpha_name = f'alpha_56__sp={slow_period}_fpo={fast_period_offset}_fp={fast_period}'
                    alpha_fn = curry(alpha_56, slow_period=slow_period, fast_period=fast_period, fast_period_offset=fast_period_offset)
                    self.market_alphas[alpha_name] = alpha_fn
                    result[alpha_name] = alpha_fn
        return result

    def build_a57_family(self):
        result = {}
        for rolling_apply_period in np.arange(5, 60, 5):
            for base_period in np.arange(2, 25, 2):
                alpha_name = f'alpha_57__bp={base_period}_advp={rolling_apply_period}'
                alpha_fn = curry(alpha_57, base_period=base_period, rolling_apply_period=rolling_apply_period)
                self.market_alphas[alpha_name] = alpha_fn
                result[alpha_name] = alpha_fn
        return result
