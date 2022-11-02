import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
from dataclasses import dataclass
import typing
import numba as nb


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


def alpha_1(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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
    std_dev = returns.rolling(20).std()
    result[returns < 0] = std_dev[returns < 0]
    result = result.pow(2) * np.sign(result)
    result = result.dropna(how='all', axis=0).fillna(method='ffill')
    for c in result.columns:
        result[c] = result[c].rolling(5).apply(lambda x: pd.Series(x).idxmax())

    return rank_signal(result)

def alpha_2(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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
        col_res = col_df.rolling(6).corr().loc[(slice(None), 'rank_id_rets'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -pd.DataFrame(result)

def alpha_3(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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

def alpha_4(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

     (-1 * Ts_Rank(rank(low), 9))


    """
    result = rank_signal(lows).rolling(9).apply(lambda x: pd.Series(x).rank(ascending=True).iloc[-1])
    return result

def alpha_5(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))

    So. Lets assume VWAP = (open*0.4 + 0.1*high + 0.1*low + 0.4*close)/4 (seems ok to me)

    The first term is the rank of the open - average vwap over 10 days
    The second term is -1 * absolute value of the rank of close - vwap
    """
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows
    vwaps_mean = vwaps.rolling(10).mean()
    first_term = rank_signal(opens - vwaps_mean)
    second_term = -rank_signal(closes-vwaps).abs()
    return first_term * second_term

def alpha_6(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1

    (-1 * correlation(open, volume, 10))
    """
    result = {}
    for c in opens.columns:
        col_df = pd.DataFrame({'open': opens[c], 'volume': volumes[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'volume'), 'open']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -pd.DataFrame(result)

def alpha_7(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    DELAY = 1
    ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))

    So. If adv < volume, we take the rank over the past 60 days of the changes in 7 day closes and multiply it by the last diff.
    Else -1
    """
    delta_close = closes.diff(7)

    # First term: ts_rank(abs(delta(close, 7)), 60))
    first_term = -delta_close.abs().rolling(60).apply(lambda x: pd.Series(x).rank(ascending=True).iloc[-1])

    # Second term: sign(delta(close, 7))
    second_term = np.sign(delta_close)

    result = first_term * second_term
    return rank_signal(result)

def alpha_8(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank (
        (sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)
    )
    """
    opens_5d_sum = opens.rolling(5).sum()
    returns_5d_sum = returns.rolling(5).sum()

    open_times_returns = opens_5d_sum * returns_5d_sum
    result = open_times_returns - open_times_returns.shift(10)
    return rank_signal(result)

def alpha_9(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((0 < ts_min(delta(close, 1), 5)) ?

     delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?

                      delta(close, 1) : (-1 * delta(close, 1))))
    """
    delta_closes = closes.diff(5)
    min_delta_closes = delta_closes.rolling(15).min()
    max_delta_closes = delta_closes.rolling(15).max()

    second_term = delta_closes.copy()
    second_term[max_delta_closes < 0] = second_term[max_delta_closes > 0]

    result = delta_closes.copy()
    result[min_delta_closes > 0] = second_term

    return result

def alpha_10(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     rank(
         ts_min(delta(close, 1), 4) > 0 ? delta(close, 1) :
             ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))
         )
    """
    delta_closes = closes.diff()
    min_delta_closes = delta_closes.rolling(4).min()
    max_delta_closes = delta_closes.rolling(4).max()

    second_term = delta_closes.copy()
    second_term[max_delta_closes > 0] = -second_term[max_delta_closes > 0]
    result = delta_closes.copy()
    result[min_delta_closes > 0] = second_term

    return rank_signal(result)


def alpha_11(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
        rank(delta(volume, 3))
    """
    vwaps = opens*0.4 + closes * 0.4 + 0.1 * highs + 0.1 * lows

    first_term = rank_signal((vwaps-closes).rolling(5).max())
    second_term = rank_signal((vwaps-closes).rolling(5).min())

    third_term = rank_signal(volumes.diff(3))

    return rank_signal(first_term + second_term + third_term)


def alpha_12(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
       (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    """
    first_term = np.sign(volumes.diff())
    second_term = -(closes.diff(2))

    return rank_signal(first_term + second_term)

def alpha_13(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
        (-1 * rank(covariance(rank(close), rank(volume), 5)))
    """
    volume_rank = rank_signal(volumes)
    closes_rank = rank_signal(closes)

    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': volume_rank[c], 'rank_closes': closes_rank[c]})
        col_res = col_df.rolling(10).cov().loc[(slice(None), 'rank_closes'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    return -rank_signal(pd.DataFrame(result))

def alpha_14(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
        (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
    """
    corr = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': volumes[c], 'rank_id_rets': opens[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'rank_id_rets'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        corr[c] = col_res
    corr = pd.DataFrame(corr)

    first_term = -rank_signal(returns.rolling(3).mean().diff())
    return first_term * corr


def alpha_15(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
         -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    """
    rank_highs = rank_signal(highs)
    rank_volumes = rank_signal(volumes)

    result = {}
    for c in volumes.columns:
        col_df = pd.DataFrame({'high': rank_highs[c], 'vol': rank_volumes[c]})
        col_res = col_df.rolling(3).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = rank_signal(pd.DataFrame(result))

    return -(-result.rolling(3).sum())

def alpha_16(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (-1 * rank(covariance(rank(high), rank(volume), 5)))
    """
    rank_highs = rank_signal(highs)
    rank_volumes = rank_signal(volumes)

    result = {}
    for c in volumes.columns:
        col_df = pd.DataFrame({'high': rank_highs[c], 'vol': rank_volumes[c]})
        col_res = col_df.rolling(5).cov().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = -rank_signal(pd.DataFrame(result))
    return result

def alpha_17(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *

    rank(ts_rank((volume / adv20), 5))
    """
    adv = volumes.rolling(20).mean()

    first_term = -(rank_signal(closes.rolling(10).apply(lambda x: pd.Series(x).rank(ascending=True).iloc[-1])) + 0.5)
    second_term = rank_signal(closes.diff().diff())
    third_term = rank_signal((volumes/adv).rolling(5).apply(lambda x: pd.Series(x).rank(ascending=False).iloc[-1]))
    result = first_term * second_term * third_term
    return result

def alpha_18(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    """
    first_term = (closes - opens).abs().rolling(5).std() * np.sqrt(261)
    second_term = closes - opens
    third_term = {}

    for c in closes.columns:
        col_df = pd.DataFrame({'open': opens[c], 'close': closes[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'open'), 'close']
        col_res.index = col_res.index.get_level_values(0)
        third_term[c] = col_res
    third_term = pd.DataFrame(third_term)
    result = -rank_signal(rank_signal(first_term) + rank_signal(second_term) + rank_signal(third_term))
    return result

def alpha_19(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    """
    first_term = -np.sign(closes - closes.shift(7) + closes.diff(7))
    second_term = 1 + rank_signal(1+returns.rolling(250).sum())
    return first_term * second_term

def alpha_20(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -
delay(low, 1)))
    """
    first_term = -rank_signal(opens - highs.shift(1))
    snd_term = rank_signal(opens - closes.shift(1))
    trd_term = rank_signal(opens - lows.shift(1))
    return first_term * snd_term * trd_term


def alpha_21(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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
    adv = volumes.rolling(20).mean()

    first_if_clause = closes.rolling(8).mean() + closes.rolling(8).std() < closes.rolling(2).mean()
    second_if_clause = closes.rolling(8).mean() - closes.rolling(8).std() > closes.rolling(2).mean()

    result = pd.DataFrame(1, index=closes.index, columns=closes.columns)
    result[second_if_clause] = -1
    result[first_if_clause] = 1

    return result

def alpha_22(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
    """
    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'high': highs[c], 'vol': volumes[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res

    first_term = -pd.DataFrame(first_term).diff(5)
    second_term = rank_signal(returns.rolling(20).std())
    return first_term * second_term

def alpha_23(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    IF sum(high, 20) / 20 < high
    THEN -1 * delta(high, 2)
    ELSE 0
    """
    first_term = highs.rolling(20).mean() < highs
    result = pd.DataFrame(0, index=highs.index, columns=highs.columns)
    result[first_term] = -highs.diff(2)
    return result

def alpha_24(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    IF  delta((sum(close, 100) / 100), 100)/delay(close, 100) <= 0.05
    THEN -1 * (close - ts_min(close, 100))
    ELSE -1 * delta(close, 3)
    """
    first_term = closes.rolling(100).mean().diff(100) / closes.shift(100) <= 0.05
    second_term = closes - closes.rolling(100).mean()
    third_term = closes.diff(3)

    result = pd.DataFrame(0, index=closes.index, columns=closes.columns)
    result[first_term] = -1 * second_term
    result[~first_term] = -1 * third_term
    return result


def alpha_25(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    """
    adv = volumes.rolling(20).mean()

    vwap = closes * 0.6 + 0.26*opens + 0.07 * lows + 0.07 * highs

    result = -returns * adv * vwap *((highs - closes))
    return rank_signal(result)

def alpha_26(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    -ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    """
    first_ts_rank = volumes.rolling(5).apply(timeseries_rank_fn)
    second_ts_rank = highs.rolling(5).apply(timeseries_rank_fn)

    first_term = {}
    for c in highs.columns:
        col_df = pd.DataFrame({'high': second_ts_rank[c], 'vol': first_ts_rank[c]})
        col_res = col_df.rolling(5).corr().loc[(slice(None), 'vol'), 'high']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)
    return - first_term.rolling(3).max()

def alpha_27(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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
        col_res = col_df.rolling(6).corr().loc[(slice(None), 'vol'), 'vwap']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res

    first_term = rank_signal(pd.DataFrame(first_term)) > 0.5

    result = pd.DataFrame(1, index=first_term.index, columns=first_term.columns)
    result[first_term] = -1
    return result

def alpha_28(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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
        col_res = col_df.rolling(5).corr().loc[(slice(None), 'vol'), 'low']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)

    result = first_term + (highs + lows)*0.5 - closes
    result = result.divide(result.abs().sum(axis=1), axis=0)
    return result


def alpha_29(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
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


def alpha_30(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3))))))
     *
     sum(volume, 5)) / sum(volume, 20)

    """
    first_term = np.sign(closes - closes.shift(1))
    first_term += np.sign(closes.shift(1) - closes.shift(2))
    first_term += np.sign(closes.shift(2) - closes.shift(3))
    first_term += np.sign(closes.shift(3) - closes.shift(4))

    second_term = volumes/volumes.rolling(10).sum()

    return rank_signal(-(first_term * second_term).rolling(3).mean()).rolling(2).mean()

def alpha_31(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      rank(decay_linear(-rank(delta(close, 10)), 10))

      + rank((-1 * delta(close, 3)))

      + sign(correlation(adv20, low, 12))

    Not implementing last part. Don't think its necessary.

    """
    first_term = rank_signal(decay_linear(-rank_signal(closes.diff(100)), 20))
    second_term = rank_signal(-closes.diff(3))

    return rank_signal(-first_term + second_term)

def alpha_32(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (
          scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))
        )
    """

    adv = volumes.rolling(20).mean()

    first_term = scale(closes.rolling(7).mean() - closes)
    first_term += scale(opens.rolling(7).mean() - opens)
    first_term += scale(closes.rolling(12).mean() - closes)


    return rank_signal(first_term)

def alpha_33(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank((-1 * ((1 - (open / close))^1)))
    Why power of 1?
    """
    return rank_signal((-(1-opens/closes)))

def alpha_34(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    """
    first_term = 1+rank_signal(returns.rolling(2).std()/returns.rolling(5).std())
    second_term = 1-rank_signal(closes.diff())

    return rank_signal(first_term + second_term)

def alpha_35(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    """
    first_term = volumes.rolling(32).apply(timeseries_rank_fn)
    third_term = 1 - returns.rolling(32).apply(timeseries_rank_fn)
    return rank_signal(rank_signal(first_term).rolling(3).mean() + rank_signal(third_term))

def alpha_36(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    Seems a bit overfit because of the constants.

    Gonna pass on this
      (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15)))
      + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    """
    return pd.DataFrame(0, index=closes.index, columns=closes.columns)

def alpha_37(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    """
    adv = volumes.rolling(20).mean()

    first_term_corr = (opens-closes).shift(1)

    first_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'first_term': first_term_corr[c], 'close': closes[c]})
        col_res = col_df.rolling(200).corr().loc[(slice(None), 'close'), 'first_term']
        col_res.index = col_res.index.get_level_values(0)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)
    second_term = opens-closes
    return rank_signal(rank_signal(first_term) + rank_signal(second_term))

def alpha_38(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    """
    first_term = rank_signal(-closes.rolling(10).apply(timeseries_rank_fn))
    second_term = rank_signal(closes/opens)
    filter_term = closes - closes.rolling(200).mean() < 0
    return first_term * second_term * filter_term

def alpha_39(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    """
    adv = volumes.rolling(20).mean()

    first_term = -rank_signal(closes.diff(7) * (1-rank_signal(decay_linear(volumes/adv, 9))))
    second_term = (1+rank_signal(returns.rolling(50).sum()))
    filter_term = closes - closes.rolling(200).mean() < 0
    filter_term = filter_term.astype(int)
    return rank_signal(first_term * second_term).rolling(10).min() * filter_term + rank_signal(first_term * second_term) * (1 - filter_term)

def alpha_40(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    """
    first_term = -rank_signal(highs.rolling(10).std())
    second_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'high': highs[c], 'volume': volumes[c]})
        col_res = col_df.rolling(10).corr().loc[(slice(None), 'high'), 'volume']
        col_res.index = col_res.index.get_level_values(0)
        second_term[c] = col_res
    second_term = pd.DataFrame(second_term)
    return -rank_signal(first_term*second_term)

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

    return rank_signal((rank_signal(vwaps - closes) + 0.5) / (rank_signal(vwaps + closes) + 0.5)).fillna(0.0)

def alpha_43(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    """
    adv = volumes.rolling(20).mean()

    first_term = (volumes/adv).rolling(20).apply(timeseries_rank_fn)
    second_term = (-closes.diff(7)).rolling(8).apply(timeseries_rank_fn)

    return rank_signal(first_term * second_term)


def alpha_44(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    """
    adv = volumes.rolling(20).mean()

    first_term = rank_signal(closes.shift(5).rolling(20).mean())

    # Lets try something diff second term
    volumes_m = volumes.mean(axis=1)
    returns_m = returns.mean(axis=1)
    corr_df = pd.DataFrame({'returns': returns_m.cumsum(), 'volume': volumes_m})
    corr_res = corr_df.rolling(2).corr().loc[(slice(None), 'returns'), 'volume']
    corr_res.index = corr_res.index.droplevel(1)
    res = rank_signal(first_term).multiply(corr_res, axis=0)
    res = rank_signal(res).rolling(60).min()
    filter_res = (closes - closes.rolling(200).mean()) < 0
    filter_res_two = returns.rolling(100).sum() < 0

    return rank_signal(res * filter_res * filter_res_two) - rank_signal(res)

def alpha_45(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    Actually 44

    (-1 * correlation(high, rank(volume), 5))
    """
    adv = volumes.rolling(20).mean()

    first_term = {}
    volume_rank = rank_signal(volumes)
    for c in highs.columns:
        corr_df = pd.DataFrame({'highs': highs[c], 'volume_rank': volume_rank[c]})
        col_res = corr_df.rolling(5).corr().loc[(slice(None), 'highs'), 'volume_rank']
        col_res.index = col_res.index.droplevel(1)
        first_term[c] = col_res
    first_term = pd.DataFrame(first_term)

    return first_term


def alpha_46(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """

    IF ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)))
    THEN -1
    ELSE IF (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0)
        THEN 1
        ELSE ((-1 * 1) * (close - delay(close, 1)))))
    """
    adv = volumes.rolling(20).mean()

    first_term = -((closes.shift(20) - closes.shift(10))/10 - (closes.shift(10) - closes)/10 > 0.25).astype(int)
    second_term = ((closes.shift(20) - closes.shift(10))/10 - (closes.shift(10) - closes)/10 < 0).astype(int)
    second_term[second_term == 0] = - (closes - closes.shift(1))[second_term == 0]
    first_term[first_term == 0] = second_term[first_term == 0]
    return first_term

def alpha_47(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    Alpha#47:
    ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    """
    adv = volumes.rolling(200).mean()

    vwaps = (highs + lows)/2

    first_term = rank_signal(1/closes)*volumes/adv
    second_term = highs * rank_signal(highs-closes)/highs.rolling(5).mean()
    third_term = rank_signal(vwaps-vwaps.shift(5))
    res = first_term * second_term + third_term
    volatility = returns.rolling(250).std()/returns.expanding().std()
    return rank_signal(res).rolling(13).min() + rank_signal(-volatility)


def alpha_48(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

    Cn't implement indneutralize
    """
    return pd.DataFrame(0, index=closes.index, columns=closes.columns)

def alpha_49(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    first_term = (closes.shift(20) - closes.shift(10))/10 - (closes.shift(10) - closes)/10
    result = first_term.copy()
    result[first_term < -0.1] = 1
    result[first_term >= -0.1] = -(closes - closes.shift(1))

    return result

def alpha_50(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    """
    vwaps = (highs + lows + closes)/3.0

    rank_vol = rank_signal(volumes)
    rank_vwap = rank_signal(vwaps)
    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': rank_vol[c], 'rank_vwaps': vwaps[c]})
        col_res = col_df.rolling(5).corr().loc[(slice(None), 'rank_vwaps'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res

    result = pd.DataFrame(result)
    return result

def alpha_51(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
    (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    """
    first_term = (closes.shift(20) - closes.shift(10))/10 - (closes.shift(10) - closes)/10
    result = (first_term < 0.05).astype(int)
    result[result == 0] = -(closes - closes.shift(1))
    return result

def alpha_52(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    """
    first_term = -lows.rolling(5).min() + lows.rolling(5).min().shift(5)
    second_term = rank_signal((returns.rolling(240).sum() - returns.rolling(20).sum())/220)
    ts_rank_vol = volumes.rolling(5).apply(timeseries_rank_fn)
    return first_term * second_term * ts_rank_vol


def alpha_53(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    """
    first_term = (closes - lows) - (highs-closes)
    first_term = first_term / (closes - lows)

    result = -(first_term.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')).diff(9)
    return rank_signal(result).rolling(3).mean()


def alpha_54(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    """
    first_term = (lows - closes) * (opens.pow(5))
    second_term = (lows - highs) * (closes.pow(5))

    return rank_signal(-(first_term / second_term).rolling(250).mean())

def alpha_55(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (-1 * correlation(
         rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))),
         rank(volume), 6))
    """
    first_term = rank_signal((closes - lows.rolling(12).min())/(highs.rolling(12).max() - lows.rolling(12).min()))
    second_term = rank_signal(volumes)

    result = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': second_term[c], 'rank_first': first_term[c]})
        col_res = col_df.rolling(6).corr().loc[(slice(None), 'rank_first'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        result[c] = col_res
    result = pd.DataFrame(result)

    return -result


def alpha_56(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
      gonna change from cap to closes * volumes
    """
    first_term = rank_signal(returns.rolling(10).sum()/returns.rolling(2).sum().rolling(3).sum())
    second_term = rank_signal(returns * closes * volumes)
    return -(first_term * second_term)

def alpha_57(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
      ((0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))

    """
    vwaps = (highs + lows)/2
    first_term = closes - vwaps
    second_term = decay_linear(rank_signal(closes.rolling(30).apply(lambda x: np.argmax(x))), 5)

    return -(first_term / second_term)

def alpha_58(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
       (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))

    Cannot implement
    """
    return pd.DataFrame(0, index=closes.index, columns=closes.columns)

def alpha_59(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
        (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

    Cannot implement
    """
    return pd.DataFrame(0, index=closes.index, columns=closes.columns)

def alpha_60(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    """
    first_term = (closes - lows) - (highs - closes)
    first_term = first_term / ((highs - lows) * volumes)
    first_term = scale(rank_signal(first_term))
    second_term = scale(rank_signal(closes.rolling(10).apply(lambda x: np.argmax(x))))

    return rank_signal(first_term * second_term)

def alpha_61(highs=None, opens=None, lows=None, closes=None, returns=None, volumes=None):
    """
     (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    """
    adv180 = volumes.rolling(180).mean()
    vwaps = (highs + lows) / 2

    first_term = rank_signal(vwaps - vwaps.rolling(16).min())

    snd_term = {}
    for c in closes.columns:
        col_df = pd.DataFrame({'rank_vol': vwaps[c], 'rank_first': adv180[c]})
        col_res = col_df.rolling(6).corr().loc[(slice(None), 'rank_first'), 'rank_vol']
        col_res.index = col_res.index.get_level_values(0)
        snd_term[c] = col_res
    snd_term = rank_signal(pd.DataFrame(snd_term))
    return (first_term < snd_term).astype(int)

MARKET_ALPHAS = [
    alpha_1,
    alpha_2,
    alpha_3,
    alpha_4,
    alpha_5,
    alpha_6,
    alpha_7,
    alpha_8,
    alpha_9,
    alpha_10,
    alpha_11,
    alpha_12,
    alpha_13,
    alpha_14,
    alpha_15,
    alpha_16,
    alpha_17,
    alpha_18,
    alpha_19,
    alpha_20,
    alpha_21,
    alpha_22,
    alpha_23,
    alpha_24,
    alpha_25,
    alpha_26,
    alpha_27,
    alpha_28,
    alpha_29,
    alpha_30,
    alpha_31,
    alpha_32,
    alpha_33,
    alpha_34,
    alpha_35,
    alpha_37,
    alpha_38,
    alpha_39,
    alpha_40,
    alpha_41,
    alpha_42,
    alpha_43,
    alpha_44,
    alpha_45,
    alpha_46,
    alpha_47,
    alpha_48,
    alpha_49,
    alpha_50,
    alpha_51,
    alpha_52,
    alpha_53,
    alpha_54,
    alpha_55,
    alpha_56,
    alpha_57,
    alpha_60,
    alpha_61,
]

