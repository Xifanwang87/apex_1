from apex.system.v12.data import ds_to_df, df_to_ds
from apex.system.v11.backtest import (
    apex__backtest_portfolio_weights,
    apex__compute_strategy_returns,
    apex__performance_stats
)
import numpy as np
import pandas as pd
import numba as nb
import xarray as xr
import pygmo as pg
from scipy.optimize import minimize
from apex.system.v12.data import (ApexBlackboard, ApexTemporaryBlackboard,
                                  ApexUniverseBlackboard)

@nb.jit
def _nb_fill_nas_mtx(arr, inplace=False):
    if not isinstance(arr, np.ndarray):
        arr = arr.values
    if inplace:
        out = arr
    else:
        out = arr.copy()
    arr_last_ix = arr.shape[0] - 1
    last_valid_indices = arr_last_ix - (~np.isnan(arr[::-1])).argmax(axis=0) - 1 ## Not isnan, reversed, argmax. Meaning first index? yep. exactly.
    for row_idx in range(1, out.shape[0]):
        for col_idx in range(0, out.shape[1]):
            if np.isnan(out[row_idx, col_idx]) and last_valid_indices[col_idx] > row_idx:
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    return out

def apex__forward_fill_nulls(data, inplace=False):
    import numpy as np
    import numba as nb

    if isinstance(data, np.ndarray):
        return _nb_fill_nas_mtx(data, inplace=inplace)
    elif isinstance(data, pd.DataFrame):
        return pd.DataFrame(_nb_fill_nas_mtx(data.values, inplace=inplace), index=data.index, columns=data.columns)
    elif isinstance(data, xr.DataArray):
        return xr.DataArray(_nb_fill_nas_mtx(data.values, inplace=inplace), dims=['time', 'ticker'], coords=[data.time, data.ticker])
    elif isinstance(data, xr.Dataset):
        return data.apply(_nb_fill_nas_mtx, inplace=inplace)


@nb.njit
def apex_nb__exponential_smoothing(series, alpha):
    out_arr = series.copy()

    n, m = out_arr.shape

    for c_ix in range(m):
        for day in range(1, n):
            if np.isnan(out_arr[day - 1, c_ix]):
                continue
            out_arr[day, c_ix] = alpha[c_ix] * series[day, c_ix] + out_arr[day - 1, c_ix] * (1-alpha[c_ix])
    return out_arr

def apex__exponential_smoothing_fit(input_data, forecast_period):
    """
    Fits exponential smoothing for particular forecast period.
    """
    assert isinstance(input_data, (xr.DataArray, pd.DataFrame))
    input_data = apex__forward_fill_nulls(input_data)
    if isinstance(input_data, xr.DataArray):
        target_data = input_data.values
        in_data = input_data.shift(time=forecast_period).values
        alpha_names = input_data.ticker.values.tolist()
    elif isinstance(input_data, pd.DataFrame):
        target_data = input_data.values
        in_data = input_data.shift(forecast_period).values
        alpha_names = input_data.columns.tolist()

    def loss_fn(alpha):
        smoothed = apex_nb__exponential_smoothing(in_data, alpha)
        errors = target_data - smoothed
        errors = np.nansum(errors ** 2)
        return errors

    n, m = input_data.shape
    start_alpha_guess = np.zeros(m) + 0.5

    alphas = minimize(loss_fn, start_alpha_guess, method='L-BFGS-B', bounds=[(0, 1)] * m).x
    result = apex_nb__exponential_smoothing(input_data.values, alphas)

    if isinstance(input_data, xr.DataArray):
        result = xr.DataArray(result, dims=['time', 'ticker'], coords=[input_data.time, input_data.ticker])
    elif isinstance(input_data, pd.DataFrame):
        result = pd.DataFrame(result, index=input_data.index, columns=input_data.columns)
    return {
        'data': result,
        'alphas': pd.Series(alphas, index=alpha_names)
    }
