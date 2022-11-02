from apex.toolz.universe import *

def index_universe(index):
    return pd.DataFrame(get_index_member_weights_multiday(index, start_date='2010-01-01', freq='Q')).index.tolist()

APEX_UNIVERSES = {
    'AMNA': ApexUniverse.amna.tickers,
    'AMUS': ApexUniverse.amus.tickers,
    'Energy Infrastructure': sorted(set(ApexUniverse.all.energy.infrastructure.tickers + ApexUniverse.amna.tickers)),
    'Midstream': ApexUniverse.all.energy.infrastructure.midstream.tickers,
    'MLPs': ApexUniverse.all.energy.infrastructure.mlps.tickers,
    'AMZ': ApexUniverse.all.benchmarks.amz.tickers,
    'AMEI': ApexUniverse.all.benchmarks.amei.tickers,
    #'GLI': ApexGlobalListedInfrastructure(),

}

from toolz import valmap
from apex.security import ApexSecurity
APEX_UNIVERSES = valmap(lambda x: [ApexSecurity.from_id(y).id for y in x if ApexSecurity.from_id(y).id != ''], APEX_UNIVERSES)

def ccorp_ei_universe():
    base_universe = APEX_UNIVERSES['Energy Infrastructure']
    bbg = ApexBloomberg()
    security_types = bbg.reference(base_universe, 'security_typ')['security_typ']
    non_mlps = security_types[security_types != 'MLP']
    return sorted(set(non_mlps.index.tolist()))


#APEX_UNIVERSES['C-Corps EI'] = ccorp_ei_universe()