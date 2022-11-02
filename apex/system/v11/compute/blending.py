import bottleneck as bn
import numba as nb
import numpy as np
import pandas as pd

import xarray as xr


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
                new_port[day - 1] = portfolio[day - 1]/np.nansum(np.abs(portfolio[day - 1]))
                initialized = True

        curr_pos = new_port[day - 1] * rets[day]
        curr_val = np.nansum(np.abs(curr_pos))
        curr_wt = curr_pos/curr_val

        new_pos = curr_wt + (portfolio[day] - curr_wt) * blend_multiplier[day]
        new_pos[~availability[day]] = 0
        port_val = np.nansum(np.abs(new_pos))
        if port_val <= 1e-10:
            if np.nansum(np.abs(new_port[day - 1])) >= 1.0:
                new_pos = new_port[day - 1] * rets[day]
            else:
                new_pos = portfolio[day]
            port_val = np.nansum(np.abs(new_pos))
        new_port[day] = new_pos/port_val
    return new_port


def apex__compute_drifting_turnover_constrained_portfolio(dataset, portfolio, turnover_period,
                                                          maximum_gap_close=0.5,
                                                          availability='default_availability',
                                                          vol_metric='ewm_vol_20d_hl'):
    var = dataset[vol_metric]
    var = var.median(axis=1).dropna()
    var_exp_median = var.expanding().median()

    # compute number of median days equivalent to rebalancing days
    turns_per_variance_day = 1/turnover_period
    blend_multiplier = var/var_exp_median * turns_per_variance_day

    # Now let's set up everything for numpy

    np_returns = dataset['returns'].reindex(portfolio.index)[portfolio.columns].fillna(0).values
    np_availability = dataset[availability].reindex(portfolio.index)[portfolio.columns].values
    np_blend_multiplier = np.minimum(blend_multiplier.reindex(portfolio.index), maximum_gap_close).fillna(0).values
    np_port = portfolio[dataset[availability]].fillna(0).values
    result = apex__compute_low_turnover_weights(np_returns, np_availability, np_port, np_blend_multiplier)
    result = pd.DataFrame(result, index=portfolio.index, columns=portfolio.columns)[dataset[availability]]
    return result