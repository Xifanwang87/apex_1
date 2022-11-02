def large_caps_availability(dataset, num_stocks=15, min_num_stocks=3):
    """
    Default availability for simplification in future
    """
    availability = dataset['default_availability']
    market_cap = dataset['cur_mkt_cap'] * availability
    availability_mkt_cap = (-market_cap.rolling(time=250).median()).rank('ticker') <= num_stocks
    availability = availability & availability_mkt_cap

    num_stocks_filter = availability.sum('ticker') >= min_num_stocks
    availability = availability * num_stocks_filter
    return availability

def mid_caps_availability(dataset, num_stocks_skip=0, num_stocks=30, min_num_stocks=5):
    """
    Default availability for simplification in future
    """
    availability = dataset['default_availability']
    market_cap = dataset['cur_mkt_cap'] * availability
    availability_mkt_cap = (-market_cap.rolling(time=250).median()).rank('ticker') >= num_stocks_skip
    availability_mkt_cap = availability_mkt_cap & ((-market_cap.rolling(time=250).median()).rank('ticker') <= num_stocks + num_stocks_skip)
    availability = availability & availability_mkt_cap
    num_stocks_filter = availability.sum('ticker') >= min_num_stocks
    availability = availability * num_stocks_filter
    return availability


def small_caps_availability(dataset, num_stocks=30, min_num_stocks=10):
    """
    Default availability for simplification in future
    """
    availability = dataset['default_availability']
    market_cap = dataset['cur_mkt_cap'] * availability
    availability_mkt_cap = (market_cap.rolling(time=250).median()).rank('ticker') <= num_stocks
    availability = availability & availability_mkt_cap
    num_stocks_filter = availability.sum('ticker') >= min_num_stocks
    availability = availability * num_stocks_filter
    return availability

def canadians_availability(dataset):
    """
    Default availability for simplification in future
    """
    tickers = dataset.ticker
    tickers = tickers.to_pandas().index.tolist()
    non_canadians = [x for x in tickers if x.split(' ')[1] != 'CN']
    default_availability = dataset['default_availability'].copy()
    default_availability.loc[{'ticker': default_availability.ticker.isin(non_canadians)}] = False
    return default_availability

def noncanadians_availability(dataset):
    """
    Default availability for simplification in future
    """
    tickers = dataset.ticker
    tickers = tickers.to_pandas().index.tolist()
    non_canadians = [x for x in tickers if x.split(' ')[1] == 'CN']
    default_availability = dataset['default_availability'].copy()
    default_availability.loc[{'ticker': default_availability.ticker.isin(non_canadians)}] = False
    return default_availability


def ccorp_availability(dataset):
    from apex.toolz.bloomberg import ApexBloomberg
    bbg = ApexBloomberg()
    tickers = dataset.ticker
    tickers = tickers.to_pandas().index.tolist()
    security_types = bbg.reference(tickers, 'security_typ')['security_typ']
    mlps = security_types[security_types == 'MLP']
    mlps = sorted(set(mlps.index.tolist()))

    default_availability = dataset['default_availability'].copy()
    default_availability.loc[{'ticker': default_availability.ticker.isin(mlps)}] = False
    return default_availability

def mlp_availability(dataset):
    from apex.toolz.bloomberg import ApexBloomberg
    bbg = ApexBloomberg()
    tickers = dataset.ticker
    tickers = tickers.to_pandas().index.tolist()
    security_types = bbg.reference(tickers, 'security_typ')['security_typ']
    non_mlps = security_types[security_types != 'MLP']
    non_mlps = sorted(set(non_mlps.index.tolist()))

    default_availability = dataset['default_availability'].copy()
    default_availability.loc[{'ticker': default_availability.ticker.isin(non_mlps)}] = False
    return default_availability


def create_strategy_subuniverses(universe_data, strategy_cache, tickers):
    base_universe = universe_data['default_availability'].copy()
    base_universe.loc[{'ticker': ~base_universe.ticker.isin(tickers)}] = False

    large_caps_universe = large_caps_availability(universe_data)
    large_caps_universe = large_caps_universe & base_universe
    large_caps_universe.name = 'large_caps_universe'

    mid_caps_universe = mid_caps_availability(universe_data)
    mid_caps_universe = mid_caps_universe & base_universe
    mid_caps_universe.name = 'mid_caps_universe'

    small_caps_universe = small_caps_availability(universe_data)
    small_caps_universe = small_caps_universe & base_universe
    small_caps_universe.name = 'small_caps_universe'

    canadians_universe = canadians_availability(universe_data)
    canadians_universe = canadians_universe & base_universe
    canadians_universe.name = 'canadians_universe'

    noncanadians_universe = noncanadians_availability(universe_data)
    noncanadians_universe = noncanadians_universe & base_universe
    noncanadians_universe.name = 'noncanadians_universe'

    ccorp_universe = ccorp_availability(universe_data)
    ccorp_universe = ccorp_universe & base_universe
    ccorp_universe.name = 'ccorp_universe'

    mlp_universe = mlp_availability(universe_data)
    mlp_universe = mlp_universe & base_universe
    mlp_universe.name = 'mlp_universe'

    strategy_cache['universe:large_cap'] = large_caps_universe
    strategy_cache['universe:mid_cap'] = mid_caps_universe
    strategy_cache['universe:small_cap'] = small_caps_universe
    strategy_cache['universe:canadian'] = canadians_universe
    strategy_cache['universe:noncanadian'] = noncanadians_universe
    strategy_cache['universe:ccorp'] = ccorp_universe
    strategy_cache['universe:mid_cap_ccorps'] = ccorp_universe & mid_caps_universe
    strategy_cache['universe:mlp'] = mlp_universe
    strategy_cache['universe:base'] = base_universe
    return [
        'universe:large_cap',
        'universe:mid_cap',
        'universe:small_cap',
        'universe:canadian',
        'universe:noncanadian',
        'universe:ccorp',
        'universe:mlp',
        'universe:base',
        'universe:mid_cap_ccorps',
    ]
