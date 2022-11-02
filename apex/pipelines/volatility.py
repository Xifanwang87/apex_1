import numpy as np


def parkinson_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices.px_high
    lows = undl_prices.px_low

    result = 1.0 / (4 * np.log(2))
    result *= ((np.log(highs / lows)**2))
    return result

def rogers_satchell_yoon_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices.px_high
    lows = undl_prices.px_low
    opens = undl_prices.px_open
    closes = undl_prices.px_last

    first_term = np.log(highs / closes)
    second_term = np.log(highs / opens)
    third_term = np.log(lows / closes)
    fourth_term = np.log(lows / opens)

    return (first_term * second_term + third_term * fourth_term)


def garman_klass_vol_estimate(undl_prices):
    """
    Page 22 Gatheral
    """
    highs = undl_prices.px_high
    lows = undl_prices.px_low
    closes = undl_prices.px_last
    closes_shifted = undl_prices.px_last.shift(1)

    first_term = (0.5 * (np.log(highs / lows).pow(2)))
    second_term = ((2 * np.log(2) - 1) *
                    (np.log(closes / closes_shifted).pow(2)))

    return first_term - second_term


def default_volatility(undl_prices, vol_days=252):
    gk_vol = garman_klass_vol_estimate(undl_prices)
    rsy_vol = rogers_satchell_yoon_vol_estimate(undl_prices)
    p_vol = parkinson_vol_estimate(undl_prices)
    gk_vol = (gk_vol.rolling(vol_days).mean().pow(0.5)) * np.sqrt(252)
    rsy_vol = (rsy_vol.rolling(vol_days).mean().pow(0.5)) * np.sqrt(252)
    p_vol = (p_vol.rolling(vol_days).mean().pow(0.5)) * np.sqrt(252)

    return (gk_vol + rsy_vol + p_vol)/3.0