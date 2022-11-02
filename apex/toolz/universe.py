import datetime
import re
from functools import reduce

import funcy
import pandas as pd
import toml
from dataclasses import dataclass, field
from funcy import mapcat, partial
from toolz import keyfilter, keymap, valmap

from apex.config.base import ApexBaseConfig
from apex.toolz.bloomberg import (BLOOMBERG_METADATA_DEFAULT_KWARGS,
                                  ApexBloomberg, fix_tickers_with_bbg,
                                  get_index_members_multiday)
from apex.toolz.bloomberg import get_security_metadata as bbg_metadata
from apex.toolz.caches import METADATA_DB_CACHE
from apex.toolz.storage import ApexMinio
from apex.toolz.strings import camelcase_to_snakecase
import typing
from apex.security import ApexSecurity


AMNA_AVAILABILITY_REMOVALS = {
    'BBG0083Y0TZ2': pd.Timestamp(day=28, year=2018, month=12),
    'BBG0059H9QV0': pd.Timestamp(day=28, year=2018, month=12),
    'BBG0068FF7N7': pd.Timestamp(day=28, year=2018, month=12),
    'BBG000BB8J57': pd.Timestamp(day=28, year=2018, month=12),
    'BBG000BJJ834': pd.Timestamp(day=28, year=2018, month=12),
    'BBG000JV7CL3': pd.Timestamp(day=28, year=2019, month=1),
    'BBG000BDMKQ1': pd.Timestamp(day=15, year=2019, month=6),
    'BBG001CWR5R3': pd.Timestamp(day=28, year=2019, month=7),

}


def is_date_the_nth_friday_of_month(nth, date=None):
    #nth is an integer representing the nth weekday of the month
    #date is a datetime.datetime object, which you can create by doing datetime.datetime(2016,1,11) for January 11th, 2016

    if not date:
        #if date is None, then use today as the date
        date = datetime.datetime.today()

    if date.weekday() == 4:
        #if the weekday of date is Friday, then see if it is the nth Friday
        if (date.day - 1) // 7 == (nth - 1):
            return True
    return False


def get_alerian_new_index_membership(excel_file, current_latest='x^5'):
    sheet_names = excel_file.sheet_names
    rgx = r'(?P<year>[0-9]+)\.(?P<month>[0-9]+)(?P<addl>x.*)?'
    announcement_day_weights = {}
    for sheet in sheet_names:
        match = re.match(rgx, sheet).groupdict()
        if match['addl'] is not None:
            if match['addl'] != current_latest:
                continue
        year = match['year']
        month = match['month']
        year = int(year)
        month = int(month)
        day = 1
        date_range = pd.date_range(
            pd.Timestamp(year, month, day),
            pd.Timestamp(year, month, day) + pd.DateOffset(months=1, days=1)
        )
        second_friday = list(
            filter(lambda x: is_date_the_nth_friday_of_month(2, x), date_range)
        )[0].date()
        announcement_day_weights[second_friday + pd.DateOffset(days=3)] = \
            excel_file.parse(sheet, skiprows=4)[['Ticker', 'Weight']].set_index('Ticker')['Weight']

    weights = pd.concat(announcement_day_weights, axis=1, sort=True).T
    new_cols = []
    for c in weights.columns:
        if len(c.split(' ')) == 1:
            c = c + ' US Equity'
        else:
            c = c + ' Equity'
        new_cols.append(c)
    weights.columns = new_cols
    weights = weights.groupby(level=0, axis=1).sum()
    return weights


def ApexUniverseAMNA(return_weights=False):
    minio = ApexMinio()
    AMNA_XLS = pd.ExcelFile(minio.get('universe', 'AMNAmembers-2018.08.10.xls'))
    amna = get_alerian_new_index_membership(AMNA_XLS)
    if return_weights:
        return amna
    amna_members = sorted(set(amna.columns.tolist()))
    # pull_members_fn = partial(get_index_members_multiday, freq='Q')
    # amna_members += list(pull_members_fn('AMNA Index', start_date='2018-01-01'))
    # amna_members = sorted(set(amna_members))

    # Members from last year
    # result = [
    #     ApexSecurity.from_id(x).parsekyable_des for x in get_index_members_multiday('AMNA Index',
    #         start_date=pd.Timestamp.now() - pd.DateOffset(years=2),
    #         end_date=pd.Timestamp.now(), freq='Q')
    # ]
    # amna_members += result
    # amna_members = sorted(set(amna_members))
    return ApexBaseConfig.from_dict('amna_universe', {'tickers': amna_members})

def ApexUniverseAMUS(return_weights=False):
    minio = ApexMinio()
    AMUS_XLS = pd.ExcelFile(minio.get('universe', 'AMUSmembers-2018.08.10.xls'))
    amus = get_alerian_new_index_membership(AMUS_XLS)
    if return_weights:
        return amus
    amus_members = sorted(set(amus.columns.tolist()))
    return ApexBaseConfig.from_dict('amus_universe', {'tickers': amus_members})

def ApexUniverseBenchmarks():
    minio = ApexMinio()
    settings = minio.get('universe', 'universe.toml').read().decode('utf8')
    return ApexBaseConfig.from_dict('universe', toml.loads(settings))


def get_updated_security_universe():
    BENCHMARKS = ApexUniverseBenchmarks()
    def build_sector_tickers(sector_dictionary):
        tickers = set()
        result = {}
        pull_members_fn = partial(get_index_members_multiday, freq='Q')
        for k, v in sector_dictionary.items():
            if k == 'sources':
                try:
                    tickers.update(sorted(mapcat(pull_members_fn, v)))
                except:
                    print(sorted(mapcat(pull_members_fn, v)))
            else:
                assert isinstance(v, dict)
                result[k] = build_sector_tickers(v)
                tickers.update(result[k]['tickers'])
        result['tickers'] = sorted(tickers)
        return result

    def build_universe_tickers_config(benchmark_universe):
        data = benchmark_universe.to_dict()
        result = build_sector_tickers(data)
        return ApexBaseConfig.from_dict(benchmark_universe.name + '_securities', result)

    t = build_universe_tickers_config(BENCHMARKS)
    return t


# This is the base configuration for apex.
@METADATA_DB_CACHE.cache_on_arguments(namespace='universe_indices')
def ApexUniverseIndices():
    minio = ApexMinio()
    settings = minio.get('universe', 'important_indices.toml').read().decode('utf8')
    settings = toml.loads(settings)['indices']
    settings = {'tickers': [x['parsekyable_des'] for x in settings]}
    return ApexBaseConfig.from_dict('universe_indices', settings)


@METADATA_DB_CACHE.cache_on_arguments(namespace='apex_universe_secs')
def _ApexUniverseSecurities():
    minio = ApexMinio()
    result = get_updated_security_universe()
    result = result.to_dict()
    result_data = toml.dumps(result).encode('utf8')
    minio.set('universe', 'universe_securities.toml', result_data)
    return result

def ApexUniverseSecurities(update=False):
    """
    Even when updating we have to cache it.
    """
    if not update:
        try:
            minio = ApexMinio()
            settings = minio.get('universe', 'universe_securities.toml').read().decode('utf8')
            return ApexBaseConfig.from_dict('universe_securities', toml.loads(settings))
        except:
            pass
    return ApexBaseConfig.from_dict('universe_securities', _ApexUniverseSecurities())

@METADATA_DB_CACHE.cache_on_arguments(namespace='apex_gli_secs')
def ApexGlobalListedInfrastructure():
    TICKERS = {'1052 HK Equity',
    '1083 HK Equity',
    '1193 HK Equity',
    '1199 HK Equity',
    '144 HK Equity',
    '1539941D US Equity',
    '177 HK Equity',
    '2688 HK Equity',
    '3 HK Equity',
    '371 HK Equity',
    '384 HK Equity',
    '392 HK Equity',
    '3IN LN Equity',
    '548 HK Equity',
    '576 HK Equity',
    '694 HK Equity',
    '737 HK Equity',
    '788 HK Equity',
    '855 HK Equity',
    '9531 JP Equity',
    '9533 JP Equity',
    '9706 JP Equity',
    'ABE SM Equity',
    'ABFOF US Equity',
    'ACE IM Equity',
    'ACKDF US Equity',
    'ADP FP Equity',
    'AENA SM Equity',
    'AEOXF US Equity',
    'AIA NZ Equity',
    'ALX AU Equity',
    'AMT US Equity',
    'APA AU Equity',
    'APAJF US Equity',
    'ASR US Equity',
    'AST AU Equity',
    'AT IM Equity',
    'ATASF US Equity',
    'ATL IM Equity',
    'ATO US Equity',
    'AWK US Equity',
    'AWR US Equity',
    'BIP US Equity',
    'BJCHF US Equity',
    'BJINF US Equity',
    'CCI US Equity',
    'CGHOF US Equity',
    'CHG US Equity',
    'CLNX SM Equity',
    'CMHHF US Equity',
    'CNP US Equity',
    'CPGX US Equity',
    'CWT US Equity',
    'DG FP Equity',
    'DPW DU Equity',
    'DPWRF US Equity',
    'DUE AU Equity',
    'DUETF US Equity',
    'ED US Equity',
    'EEQ US Equity',
    'EIT IM Equity',
    'EIX US Equity',
    'ELI BB Equity',
    'ELIAF US Equity',
    'ENB CN Equity',
    'ENB US Equity',
    'ENF CN Equity',
    'ENG SM Equity',
    'ENGGF US Equity',
    'ENLC US Equity',
    'ENV AU Equity',
    'ES US Equity',
    'ETL FP Equity',
    'EUTLF US Equity',
    'FCGYF US Equity',
    'FER SM Equity',
    'FGX AU Equity',
    'FHZN SW Equity',
    'FPRUF US Equity',
    'FRA GR Equity',
    'FRRVF US Equity',
    'FTS CN Equity',
    'FTS US Equity',
    'GAS US Equity',
    'GET FP Equity',
    'GLPR LI Equity',
    'GMAAF US Equity',
    'GRPTF US Equity',
    'H CN Equity',
    'HCTPF US Equity',
    'HHFA GR Equity',
    'HHULF US Equity',
    'HICL LN Equity',
    'HICLF US Equity',
    'HIFR US Equity',
    'HOKCF US Equity',
    'HPHT SP Equity',
    'ICAYY US Equity',
    'IG IM Equity',
    'INW IM Equity',
    'IPL CN Equity',
    'ITC US Equity',
    'JAIRF US Equity',
    'JEXYF US Equity',
    'KEY CN Equity',
    'KEYUF US Equity',
    'KMI US Equity',
    'KML CN Equity',
    'KMR US Equity',
    'LNG US Equity',
    'MAQAF US Equity',
    'NG/ LN Equity',
    'NGGTF US Equity',
    'NI US Equity',
    'NJR US Equity',
    'NWE US Equity',
    'NWN US Equity',
    'OGS US Equity',
    'OHL SM Equity',
    'OKE US Equity',
    'OMAB US Equity',
    'PAC US Equity',
    'PBA US Equity',
    'PCG US Equity',
    'PEGRF US Equity',
    'PNN LN Equity',
    'PNY US Equity',
    'POM US Equity',
    'PPL CN Equity',
    'RDEIF US Equity',
    'REE SM Equity',
    'RWAY IM Equity',
    'SAUNF US Equity',
    'SBAC US Equity',
    'SBS US Equity',
    'SEMG US Equity',
    'SESG FP Equity',
    'SFDPF US Equity',
    'SGBAF US Equity',
    'SIS IM Equity',
    'SIZAF US Equity',
    'SJW US Equity',
    'SKI AU Equity',
    'SNMRF US Equity',
    'SR US Equity',
    'SRE US Equity',
    'SRG IM Equity',
    'SVT LN Equity',
    'SVTRF US Equity',
    'SWX US Equity',
    'SYD AU Equity',
    'SYDDF US Equity',
    'TCL AU Equity',
    'TERRF US Equity',
    'TGASF US Equity',
    'THOGF US Equity',
    'TKGSF US Equity',
    'TRAUF US Equity',
    'TRGP US Equity',
    'TRN IM Equity',
    'TRP CN Equity',
    'TRP US Equity',
    'UIL US Equity',
    'UTL US Equity',
    'UU/ LN Equity',
    'UUGWF US Equity',
    'UZAPF US Equity',
    'VOPKF US Equity',
    'VPK NA Equity',
    'VSN CN Equity',
    'WGL US Equity',
    'WMB US Equity',
    'WTE CN Equity',
    'WTR US Equity',
    'WTSHF US Equity',
    'XNGSF US Equity',
    'ZHEXF US Equity'}
    return TICKERS
    from apex.toolz.bloomberg import (apex__adjusted_market_data, ApexBloomberg,
                                    fix_security_name_index, get_security_metadata,
                                    get_index_member_weights_on_day)
    tickers = get_index_member_weights_on_day('DJBGIT Index', pd.Timestamp.now().strftime("%Y-%m-%d"))
    if tickers is not None:
        TICKERS.update(fix_security_name_index(tickers).index.tolist())
    return TICKERS

def ApexCustomRoweIndicesUniverse():
    minio = ApexMinio()
    data = pd.read_excel(minio.get('universe', 'custom_mlp_indices.xlsx'), sheet_name='Indices').dropna(axis=1, how='all')
    result = {
        k: data[k].dropna().tolist() for k in data.columns
    }
    tickers = sorted(set(reduce(lambda x, y: x + y, result.values(), [])))

    tickers = funcy.zipdict(tickers, fix_tickers_with_bbg(*tickers).values())
    result = {
        k: [tickers[x] for x in result[k] if x in tickers] for k in result
    }
    result['e&p'] = result['EnP']
    del result['EnP']
    result = keymap(camelcase_to_snakecase, result)
    del result['other']
    del result['indices']
    del result['commodity']
    final_res = {
        'custom_indices': result,
        'tickers': sorted(tickers.values())
    }
    return ApexBaseConfig.from_dict('rowe_indices', final_res)


@dataclass
class ApexUniverse:
    amna: typing.Any = field(default_factory=ApexUniverseAMNA)
    amus: typing.Any = field(default_factory=ApexUniverseAMUS)
    benchmarks: typing.Any = field(default_factory=ApexUniverseBenchmarks)
    all: typing.Any = field(default_factory=ApexUniverseSecurities)
    def __call__(self):
        return self

ApexUniverse = ApexUniverse()
