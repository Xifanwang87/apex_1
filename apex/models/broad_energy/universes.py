from apex.toolz.bloomberg import get_index_members_multiday


def integrateds_universe():
    result = list(get_index_members_multiday('S15IOIL Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S5IOIL Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('SYIOIL Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('R2GOIDM Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

def equipment_oilfield_svcs_universe():
    result = list(get_index_members_multiday('S15OILE Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S5OILE Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('SPSIOS Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('DJSOES Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

def enp_universe():
    result = list(get_index_members_multiday('S15OILP Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S5OILP Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('SPSIOP Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S12OILP Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

def downstream_universe():
    result = list(get_index_members_multiday('S15OILR Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S4OILR Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('S12OILR Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('SOILR Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

def midstream_universe():
    result = list(get_index_members_multiday('AMNA Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('AMZ Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('AMEI Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

def broad_universe():
    result = list(get_index_members_multiday('AMNA Index', start_date='2000-01-01', freq='Q'))
    result += list(get_index_members_multiday('AMZ Index', start_date='2000-01-01', freq='Q'))
    return sorted(set(result))

