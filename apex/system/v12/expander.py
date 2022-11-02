
TS_TRANSFORMS = {
    'identity': lambda x: x,

    'ewm1': lambda x: x.ewm(halflife=1).mean(),
    'ewm3': lambda x: x.ewm(halflife=3).mean(),
    'ewm5': lambda x: x.ewm(halflife=5).mean(),
    'ewm10': lambda x: x.ewm(halflife=10).mean(),
    'ewm20': lambda x: x.ewm(halflife=20).mean(),
    'ewm50': lambda x: x.ewm(halflife=50).mean(),
    'ewm100': lambda x: x.ewm(halflife=100).mean(),
    'ewm200': lambda x: x.ewm(halflife=200).mean(),
}

TS_DIFFED_TRANSFORMS = {
#     'ewm5_diffed': lambda x: x.ewm(halflife=5).mean().diff(),
#     'ewm20_diffed': lambda x: x.ewm(halflife=5).mean().diff(),
#     'ewm50_diffed': lambda x: x.ewm(halflife=40).mean().diff(),
#     'ewm100_diffed': lambda x: x.ewm(halflife=100).mean().diff(),
}

@nb.njit
def apex__nb_compute_low_turnover_weights(returns, availability, portfolio, blend_multiplier):
    """
    Blend every day at rate = blend multiplier / 2

    portfolio: np.ndarray shaped (n_time, n_securities)
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

def apex__blend_transformer(dataset, universe, portfolio_pipeline, blend_reference_quantile=0.99, blend_var_halflife=1, blend_turnover_days=1):
    """
    Transforms the portfolios in pipeline into blended ones

        blending_options = {
            'blend_reference_quantile': 0.99,
            'blend_var_halflife': 1,
            'blend_turnover_days': 1
        }
    """
    blend_var_halflife = 1
    var = xr.DataArray(dataset.returns.to_pandas().ewm(halflife=blend_var_halflife).var())
    var = var.median('ticker')
    var_exp_median = xr.DataArray(var.to_pandas().expanding().quantile(blend_reference_quantile))

    # compute number of median days equivalent to rebalancing days
    turns_per_variance_day = 1/blend_turnover_days
    blend_multiplier = np.minimum(var/var_exp_median * turns_per_variance_day, 1)

    np_rets = dataset.returns.values
    np_av = dataset[universe].to_pandas().fillna(False).astype(bool).values
    np_bld = blend_multiplier.values
    for p in portfolio_pipeline:
        try:
            p = p.to_pandas()
            np__port = apex__nb_compute_low_turnover_weights(np_rets, np_av, p.values, np_bld)
        except:
            print(p.shape, availability.shape, blend_multiplier.shape)
            continue
        yield xr.DataArray(np__port, dims=['time', 'ticker'], coords=[dataset.time, dataset.ticker])



def apex__signal_expander(dataset, cutoff=0, winsorize_limit=0.05, long_short=False, universe='universe:base', prefix=None, blending_options=None):
    """
    The timeseries expander takes a signal, expands it across time, and yields back all the expansions
    This signal must be normalized to center at zero per stock

    It is always long-only
    """
    vol = dataset['volatility']
    signal = dataset['alpha']

    availability = dataset[universe]
    signal = signal * availability
    signal = signal.rank('ticker', pct=True) # (0 -> 1)
    signal = (signal - signal.min('ticker')) / (signal.max('ticker') - signal.min('ticker')) # (0, 1)

    signal_df = signal.to_pandas()
    signal_ds = {}

    MULTIPLIERS = {
        'base': 1,
        'vol': vol,
        'inv_vol': 1/vol
    }


    def compute_derived_portfolios(t_signal):
        t_signals = []
        # Not as important
        # for multiplier_name, multiplier in MULTIPLIERS.items():
        #     base_port_t = (np.sign(t_signal) * multiplier)
        #     t_signals.append(base_port_t)
        for multiplier_name, multiplier in MULTIPLIERS.items():
            base_port_t = (t_signal * multiplier)
            t_signals.append(base_port_t)
        return t_signals


    transforms = merge(TS_TRANSFORMS, TS_DIFFED_TRANSFORMS)
    transformed_signals = []
    for transform_name, transform in transforms.items():
        """
        Now you need to optimize this.
        Put all t_signals on a single dataset (everything right before normalization/multiplying by multipliers)
        """
        # Transform
        t_signal = transform(signal_df)
        t_signal = xr.DataArray(t_signal, dims=['time', 'ticker'])
        transformed_signals += compute_derived_portfolios(t_signal)
        transformed_signals += compute_derived_portfolios(-t_signal)

        t_signal = transform(np.sign(signal_df))
        t_signal = xr.DataArray(t_signal, dims=['time', 'ticker'])
        transformed_signals += compute_derived_portfolios(t_signal)
        transformed_signals += compute_derived_portfolios(-t_signal)


    rank_fn = lambda x: x.rank('ticker') * availability
    normalize_fn = lambda x: (x - x.min('ticker')) / (x.max('ticker') - x.min('ticker'))
    winsorize_fn = lambda signal: xr.DataArray(winsorize(signal.values, limits=[winsorize_limit, winsorize_limit]),
                                               dims=['time', 'ticker'],
                                               coords={'time': signal.time.values, 'ticker': signal.ticker.values})

    t_ds_dict = {f't_signal={ix}': data for ix, data in enumerate(transformed_signals)}
    t_ds = xr.Dataset(data_vars=t_ds_dict)
    # t_ds = rank_fn(t_ds)
    t_ds = normalize_fn(t_ds * availability)

    if long_short:
        t_ds = t_ds * 2 - 1
    t_ds = t_ds.where(np.abs(t_ds) > cutoff) # (cutoff, 1) or (-1, -cutoff) and (cutoff, 1)
    t_ds = t_ds / np.abs(t_ds).sum('ticker')
    t_ds_dvs = [f'{prefix}_{x}' for x in t_ds.data_vars.keys()]
    t_ds_dvs = dict(zip(t_ds.data_vars.keys(), t_ds_dvs))
    t_ds_dict = {t_ds_dvs[x]: t_ds[x] for x in t_ds_dict}
    t_ds.rename(name_dict=t_ds_dvs, inplace=True)

    blending_options = {
        'blend_reference_quantile': 0.5,
        'blend_var_halflife': 1,
        'blend_turnover_days': 0.25,
    }
    blended_portfolios = apex__blend_transformer(dataset, universe, t_ds_dict.values(), **blending_options)
    t_ds_bld = dict(zip(t_ds_dict.keys(), blended_portfolios))
    t_ds_bld = {x + '_blended': t_ds_bld[x] for x in t_ds_dict}
    t_ds_bld = xr.Dataset(data_vars=t_ds_bld) * availability
    t_ds_bld = t_ds_bld / np.abs(t_ds_bld).sum('ticker')


    portfolios = xr.merge([t_ds_bld, t_ds])
    return portfolios

    blending_options = {
        'blend_reference_quantile': 0.99,
        'blend_var_halflife': 1,
        'blend_turnover_days': 0.25
    }
    blended_portfolios = apex__blend_transformer(dataset, universe, t_ds_dict.values(), **blending_options)
    t_ds_bld = dict(zip(t_ds_dict.keys(), blended_portfolios))
    t_ds_bld = {x + '_blended': t_ds_bld[x] for x in t_ds_dict}
    t_ds_bld = xr.Dataset(data_vars=t_ds_bld) * availability
    t_ds_bld = t_ds_bld / np.abs(t_ds_bld).sum('ticker')
    return xr.merge([t_ds_bld, t_ds_bld_slow])


