import pickle
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from tempfile import NamedTemporaryFile

import dogpile.cache as dc
import pandas as pd
from dogpile.cache import make_region
from toolz import curry, partition_all

from .dask import ApexDaskClient, compute_delayed
from .deco import retry
from .service_client import BROKER_ADDR, ApexServiceClient
from .caches import METADATA_DB_CACHE, INDEX_WEIGHTS_CACHING, PORTFOLIO_DB_CACHE, MARKET_DATA_CACHING
import funcy, toolz


from functools import reduce
from toolz import valfilter
from funcy import isnone
from apex.toolz.functools import isnotnone


def should_cache_market_data_fn(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return True
    if isinstance(x, bool):
        return False
    if not x:
        return False
    return False

def fix_raw_ticker_us(x):
    values_to_replace = (
        ' UW ',
        ' UN ',
        ' UQ ',
        ' UA ',
        ' UV ',
        ' UR ',
    )
    u = ' US '
    for v in values_to_replace:
        x = x.replace(v, u)
    return x


def fix_raw_ticker_cn(x):
    values_to_replace = (
        ' CT ',
    )
    u = ' CN '
    for v in values_to_replace:
        x = x.replace(v, u)
    return x



BLOOMBERG_MARKET_DATA_FIELDS = [
    'px_last',
    'px_high',
    'px_low',
    'px_open',
    'px_volume',
    'day_to_day_tot_return_gross_dvds',
]

BLOOMBERG_ADJUSTED_KWARGS = {
    'ignore_security_error': True,
    'ignore_field_error': True,
    'adjustment_split': True,
    'adjustment_abnormal': True,
    'non_trading_day_fill_method': 'PREVIOUS_VALUE',
    'non_trading_day_fill_option': 'ACTIVE_DAYS_ONLY',
    'currency': 'USD'
}

BLOOMBERG_DEFAULT_KWARGS = {
    'ignore_security_error': True,
    'ignore_field_error': True,
    'currency': 'USD'
}

BLOOMBERG_QUARTERLY_FUNDAMENTAL_FIELDS = [
    'announcement_dt',
    'bs_cur_asset_report',
    'bs_cur_liab',
    'bs_gross_fix_asset',
    'bs_lt_borrow',
    'bs_lt_invest',
    'bs_net_fix_asset',
    'tot_common_eqy',
    'bs_other_lt_liabilities',
    'bs_retain_earn',
    'bs_sh_out',
    'bs_st_borrow',
    'bs_tot_asset',
    'bs_tot_liab2',
    'tangible_assets',
    'tot_liab_and_eqy',
    'total_equity',
    'working_capital',
    'total_invested_capital',
    'cf_cash_from_oper',
    'cf_free_cash_flow',
    'cf_incr_lt_borrow',
    'ebit',
    'cf_net_inc',
    'cf_incr_invest',
    'cf_dvd_paid',
    'cf_net_chng_cash',
    'cf_ffo',
    'cf_ffo_net_income',
    'is_abnormal_item',
    'is_comp_net_income_adjust',
    'is_diluted_eps',
    'is_eps',
    'is_inc_bef_xo_item',
    'is_int_expense',
    'is_net_non_oper_loss',
    'quick_ratio',
    'is_oper_inc',
    'is_operating_expn',
    'is_div_per_shr',
    'is_cogs_to_fe_and_pp_and_g',
    'capital_expend',
    'ebitda',
    'net_debt',
    'altman_z_score',
    'book_val_per_sh',
    'cash_dvd_coverage',
    'dvd_payout_ratio',
    'oper_margin',
    'prof_margin',
    'return_on_asset',
    'fncl_lvrg',
    'tot_debt_to_tot_asset',
    'cf_ffo_per_sh',
    'cf_ffo_per_sh_diluted',
    'sales_rev_turn',
    'net_income',
    'ev_to_t12m_ebitda',
    'net_debt_to_ebitda',
    'net_debt_to_shrhldr_eqty',
    'operating_roic',
    'cap_expend_to_sales',
    'return_com_eqy',
    'return_on_inv_capital',
    'sales_growth',
    'tot_debt_to_tot_eqy',
    'tot_debt_to_tang_book_val',
    'wacc',
    'wacc_cost_debt',
    'return_on_capital_employed',
    'fcf_to_total_debt',
]

BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS = [
    'average_dividend_yield',
    'average_price_to_book_ratio',
    'best_analyst_rating',
    'best_analyst_rec',
    'best_analyst_recs_bulk',
    'best_ask',
    'best_bid',
    'best_bps',
    'best_calculated_fcf',
    'best_capex',
    'best_cps',
    'best_cur_ev_to_ebitda',
    'best_current_ev_best_ebit',
    'best_current_ev_best_sales',
    'best_div_yld',
    'best_dps',
    'best_ebit',
    'best_ebitda',
    'best_edps_cur_yr',
    'best_edps_nxt_yr',
    'best_eeps_cur_yr',
    'best_eeps_nxt_yr',
    'best_eps_3mo_pct_chg',
    'best_eps_4wk_chg',
    'best_eps_4wk_dn',
    'best_eps_4wk_pct_chg',
    'best_eps_4wk_up',
    'best_eps_6mo_pct_chg',
    'best_eps_gaap',
    'best_eps_median',
    'best_eps_numest',
    'best_eps_nxt_yr',
    'best_eps_stddev',
    'best_eps_yoy_gth',
    'best_eps',
    'best_esales_nxt_yr',
    'best_est_ebitda_nxt_yr_mean',
    'best_est_eps_fy3',
    'best_est_long_term_growth',
    'best_est_pe_cur_yr',
    'best_est_pe_nxt_yr',
    'best_estimate_fcf',
    'best_ev_to_best_ebit',
    'best_ev_to_best_ebitda',
    'best_ev',
    'best_gross_margin',
    'best_net_debt',
    'best_net_income',
    'best_opp',
    'best_pe_cur_yr',
    'best_pe_nxt_yr',
    'best_pe_ratio',
    'best_peg_ratio',
    'best_period_end_date',
    'best_px_bps_ratio',
    'best_px_cps_ratio',
    'best_px_sales_ratio',
    'best_roe',
    'best_sales_yoy_gth',
    'best_sales',
    'best_target_hi',
    'best_target_lo',
    'best_target_median',
    'best_target_price',
    'capital_yield',
    'cash_ratio',
    'cur_ratio',
    'current_ev_to_t12m_ebitda',
    'current_px_to_free_cash_flow',
    'dvd_payout_ratio',
    'earn_yld_hist',
    'earn_yld',
    'ebit_ev_yield',
    'ebit_yield',
    'ebit_yld',
    'ebita_ev_yield',
    'eff_ratio',
    'enterprise_val_1_yr_growth',
    'enterprise_val_5_yr_growth'
    'eqy_dvd_yld_ind_net',
    'est_ev_next_yr_ebitda',
    'ev_est_ebitda_fy3_aggte',
    'ev_est_ebitda_next_yr_aggte',
    'ev_to_t12m_free_cash_flow',
    'fcf_yield_with_cur_entp_val',
    'fcf_yield_with_cur_mkt_cap',
    'five_year_avg_ev_to_t12_ebitda',
    'free_cash_flow_yield',
    'iest_finl_tangible_bps',
    'iest_gas_production',
    'iest_ngl_production',
    'iest_oil_production',
    'iest_production',
    'iest_total_production',
    'information_ratio',
    'interest_coverage_ratio',
    'pe_ratio',
    'pegy_ratio',
    'price_to_boe_reserves',
    'px_to_book_ratio',
    'px_to_cash_flow',
    'px_to_ebitda',
    'px_to_eps_before_abnormal_items',
    'px_to_est_ebitda',
    'px_to_ffo_ratio',
    'px_to_free_cash_flow',
    'px_to_sales_ratio',
    'px_to_tang_bv_per_sh',
    'return_com_eqy',
    'shareholder_yield_cff',
    'shareholder_yield_ex_debt',
    'shareholder_yield',
    'short_int_ratio',
    'sortino_ratio',
    't12m_fcf_to_firm_yield',
]

BLOOMBERG_METADATA_DEFAULT_KWARGS = {
    'ignore_security_error': True,
    'ignore_field_error': True,
}

BLOOMBERG_METADATA_FIELDS = [
    'alternate_form_bbid',
    'bb_country_code',
    'composite_exch_code',
    'composite_id_bb_global',
    'eqy_fund_ind',
    'eqy_fund_ticker',
    'eqy_prim_security_comp_exch',
    'eqy_prim_security_crncy',
    'eqy_prim_security_prim_exch',
    'eqy_prim_security_ticker',
    'id_bb_company',
    'id_bb_global_company',
    'id_bb_global_parent_co',
    'id_bb_global_rt',
    'id_bb_global_ult_parent_co_name',
    'id_bb_global',
    'id_bb_parent_co',
    'id_bb_prim_security_flag',
    'id_bb_sec_num_des',
    'id_bb_sec_num_src',
    'id_bb_sec_num',
    'id_bb_sec_number_description_rt',
    'id_bb_security',
    'id_bb_unique',
    'id_bb',
    'id_common',
    'id_cusip_id_num',
    'id_cusip',
    'id_exch_symbol',
    'id_full_exchange_symbol',
    'id_isin',
    'id_mic_prim_exch',
    'id_sedol1',
    'id_sedol2',
    'id_stock_exchange',
    'isin_ticker_exch_list',
    'issuer_alt_bb_id'
    'market_sector_des',
    'market_sector',
    'parsekyable_des_source',
    'parsekyable_des',
    'prim_security_comp_id_bb_global',
    'ticker',
]

class ApexBloomberg(ApexServiceClient):
    def __init__(self, verbose=False):
        super().__init__(BROKER_ADDR, verbose=verbose)

    @retry(Exception, tries=3)
    def history(self, tickers, fields, start_date=None, end_date=None, kwargs=None):
        today = pd.Timestamp.now(tz='America/Chicago')
        if start_date is None:
            start_date = pd.to_datetime('1/1/1980').tz_localize('America/Chicago').strftime('%Y-%m-%d')
        if end_date is None:
            end_date = today.strftime('%Y-%m-%d')
        req = {'service': 'historical', 'args': [
            tickers, fields, start_date, end_date]}
        if kwargs is not None:
            req['kwargs'] = kwargs
        else:
            req['kwargs'] = BLOOMBERG_DEFAULT_KWARGS.copy()
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def reference(self, tickers, fields, kwargs=None):
        req = {'service': 'reference_data', 'args': [tickers, fields]}
        if kwargs is not None:
            req['kwargs'] = kwargs
        else:
            req['kwargs'] = BLOOMBERG_DEFAULT_KWARGS.copy()
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def intraday_tick(self, tickers, events, start_date, end_date):
        req = {'service': 'historical_intraday_tick',
               'args': [tickers, events, start_date, end_date]}
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def intraday_bar(self, tickers, fields, start_date, end_date, interval):
        req = {'service': 'historical_intraday_bar', 'args': [
            tickers, fields, start_date, end_date, interval], 'kwargs': {'gap_fill_initial_bar': True}}
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def portfolio_data(self, portfolio_id, date):
        req = {'service': 'portfolio_data', 'args': [portfolio_id, date]}
        self.send('bbg_account_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def index_reference_data(self, tickers, fields, kwargs=None):
        req = {'service': 'reference_data', 'args': [tickers, fields]}
        if kwargs is not None:
            req['kwargs'] = kwargs
        else:
            req['kwargs'] = BLOOMBERG_DEFAULT_KWARGS.copy()
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    def quarterly_fundamental_data(self, securities, start_date=None, end_date=None):
        pool = ThreadPoolExecutor()
        # Quarterly
        qpull = list(partition_all(25, BLOOMBERG_QUARTERLY_FUNDAMENTAL_FIELDS))
        data = []
        for batch in qpull:
            data.append(pool.submit(historical_bloomberg_data_pull, securities, batch, start_date=start_date, end_date=end_date))
        for ix, result in enumerate(data):
            data[ix] = result.result()
        data = list(toolz.remove(lambda x: x is False, data))
        data = pd.concat(data, axis=1)
        result = {}
        for security in securities:
            sec_data = data[security]
            sec_data = sec_data.dropna(how='all')
            announcement_date = sec_data['announcement_dt'].dropna()
            announcement_date = pd.to_datetime(announcement_date, format='%Y%m%d')
            sec_data = sec_data.reindex(announcement_date.index)
            sec_data.index = announcement_date.values
            sec_data = sec_data.fillna(limit=1, method='ffill')[~sec_data.index.duplicated(keep='last')]
            result[security] = sec_data

        result = pd.concat(result, axis=1)
        result.index.rename('date', inplace=True)
        return result

    @retry(Exception, tries=3)
    def fundamentals(self, securities, fields, start_date=None, end_date=None):
        pool = ThreadPoolExecutor()
        # Quarterly
        if isinstance(fields, str):
            fields = [fields]
        original_fields = fields.copy()
        fields.append('announcement_dt')
        qpull = list(partition_all(25, fields))
        data = []
        for batch in qpull:
            data.append(pool.submit(historical_bloomberg_data_pull, securities, batch, start_date=start_date, end_date=end_date))
        for ix, result in enumerate(data):
            data[ix] = result.result()
        data = list(toolz.remove(lambda x: x is False, data))
        data = pd.concat(data, axis=1)
        result = {}
        for security in securities:
            sec_data = data[security]
            sec_data = sec_data.dropna(how='all')
            announcement_date = sec_data['announcement_dt'].dropna()
            try:
                announcement_date = pd.to_datetime(announcement_date, format='%Y%m%d')
            except:
                announcement_date = pd.to_datetime(announcement_date)
            sec_data = sec_data[original_fields].reindex(announcement_date.index)
            sec_data.index = announcement_date.values
            sec_data = sec_data.fillna(limit=1, method='ffill')[~sec_data.index.duplicated(keep='last')]
            result[security] = sec_data[original_fields]

        result = pd.concat(result, axis=1)
        result.index.rename('date', inplace=True)
        return result


    def daily_fundamental_data(self, securities, start_date=None, end_date=None):
        pool = ThreadPoolExecutor()
        # Quarterly
        qpull = list(partition_all(25, BLOOMBERG_DAILY_FUNDAMENTAL_FIELDS))
        data = []
        for batch in qpull:
            data.append(pool.submit(historical_bloomberg_data_pull, securities, batch, start_date=start_date, end_date=end_date))
        for ix, result in enumerate(data):
            data[ix] = result.result()

        data = list(toolz.remove(lambda x: x is False, data))
        data = pd.concat(data, axis=1)
        result = {}
        for security in securities:
            sec_data = data[security]
            sec_data = sec_data.dropna(how='all')
            sec_data = sec_data.fillna(limit=1, method='ffill')[~sec_data.index.duplicated(keep='last')]
            result[security] = sec_data

        result = pd.concat(result, axis=1)
        result.index.rename('date', inplace=True)
        return result

    def fundamental_data(self, securities, start_date=None, end_date=None):
        quarterly = self.quarterly_fundamental_data(securities, start_date=start_date, end_date=end_date)
        daily = self.daily_fundamental_data(securities, start_date=start_date, end_date=end_date)

        return {
            'quarterly': quarterly,
            'daily': daily
        }

    def index_data(self, index, start_date=None, end_date=None, freq='B'):
        return get_index_member_weights_multiday(index, start_date=start_date, end_date=end_date, freq='B')

    @retry(Exception, tries=3)
    def market_data(self, securities, start_date=None, end_date=None):
        today = pd.Timestamp.now(tz='America/Chicago')
        if start_date is None:
            start_date = pd.to_datetime('1/1/1980').tz_localize('America/Chicago').strftime('%Y-%m-%d')
        if end_date is None:
            end_date = today.strftime('%Y-%m-%d')
        adjusted_data = self.history(
            securities,
            BLOOMBERG_MARKET_DATA_FIELDS,
            start_date=start_date,
            end_date=end_date,
            kwargs=BLOOMBERG_ADJUSTED_KWARGS
        )
        unadjusted_data = self.history(
            securities,
            BLOOMBERG_MARKET_DATA_FIELDS,
            start_date=start_date,
            end_date=end_date,
            kwargs=BLOOMBERG_DEFAULT_KWARGS
        )
        return {
            'adjusted': adjusted_data,
            'unadjusted': unadjusted_data
        }

    @retry(Exception, tries=3)
    def adjusted_market_data(self, securities, start_date=None, end_date=None):
        today = pd.Timestamp.now(tz='America/Chicago')
        if start_date is None:
            start_date = pd.to_datetime('1/1/1980').tz_localize('America/Chicago').strftime('%Y-%m-%d')
        if end_date is None:
            end_date = today.strftime('%Y-%m-%d')
        adjusted_data = self.history(
            securities,
            BLOOMBERG_MARKET_DATA_FIELDS,
            start_date=start_date,
            end_date=end_date,
            kwargs=BLOOMBERG_ADJUSTED_KWARGS
        )
        return adjusted_data

    @retry(Exception, tries=3)
    def unadjusted_market_data(self, securities, start_date=None, end_date=None):
        today = pd.Timestamp.now(tz='America/Chicago')
        if start_date is None:
            start_date = pd.to_datetime('1/1/1980').tz_localize('America/Chicago').strftime('%Y-%m-%d')
        if end_date is None:
            end_date = today.strftime('%Y-%m-%d')
        adjusted_data = self.history(
            securities,
            BLOOMBERG_MARKET_DATA_FIELDS,
            start_date=start_date,
            end_date=end_date,
            kwargs=BLOOMBERG_DEFAULT_KWARGS
        )
        return adjusted_data

    @retry(Exception, tries=3)
    def field_search(self, search_string):
        req = {'service': 'search_fields', 'args': [search_string]}
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])

    @retry(Exception, tries=3)
    def security_search(self, search_string, max_results=100):
        req = {'service': 'search_securities',
               'args': [search_string, max_results]}
        self.send('bloomberg_data_service', pickle.dumps(req))
        return pickle.loads(self.recv()[0])


def historical_bloomberg_data_pull(tickers, fields, start_date=None, end_date=None):
    today = pd.Timestamp.now(tz='America/Chicago')
    if start_date is None:
        start_date = pd.to_datetime('1/1/1980').tz_localize('America/Chicago').strftime('%Y-%m-%d')
    if end_date is None:
        end_date = today
    data = []
    bbg = ApexBloomberg()
    return bbg.history(tickers, fields, start_date=start_date, end_date=end_date)


def metadata_to_dict(metadata):
    if not isinstance(metadata, dict):
        data = metadata.fillna('').to_dict()
    else:
        data = metadata
    try:
        data['isin_ticker_exch_list'] =  data['isin_ticker_exch_list']['Ticker and Exchange Code'].values.tolist()
    except:
        pass
    try:
        data['related_composite_sec'] =  data['related_composite_sec']['Securities'].values.tolist()
    except:
        pass
    try:
        data['related_equities'] =  data['related_equities']['Ticker and Exchange Code'].values.tolist()
    except:
        pass
    try:
        data['id_bb_parent_co'] = str(int(data['id_bb_parent_co']))
    except:
        pass
    try:
        data['primary_security_ticker'] = f"""{data['eqy_prim_security_ticker']} {data['eqy_prim_security_comp_exch']} {data['market_sector_des']}"""
    except:
        pass
    return data

@METADATA_DB_CACHE.cache_on_arguments(should_cache_fn=isnotnone)
def get_single_security_metadata(identifier):
    bbg = ApexBloomberg()
    result = bbg.reference(
        identifier,
        BLOOMBERG_METADATA_FIELDS,
        kwargs=BLOOMBERG_METADATA_DEFAULT_KWARGS
    ).T
    return metadata_to_dict(result[identifier])


@METADATA_DB_CACHE.cache_multi_on_arguments(asdict=True, should_cache_fn=should_cache_market_data_fn)
@retry(Exception, tries=3)
def get_security_metadata(*identifiers):
    bbg = ApexBloomberg()
    if len(identifiers) < 100:
        batches = partition_all(30, identifiers)
        metadata = {}
        for batch in batches:
            result = bbg.reference(
                batch,
                BLOOMBERG_METADATA_FIELDS,
                kwargs=BLOOMBERG_METADATA_DEFAULT_KWARGS
            ).T
            result = {x: metadata_to_dict(result[x]) for x in result.columns}
            metadata.update(result)
        return result
    else:
        pool = ThreadPoolExecutor()
        result = funcy.zipdict(identifiers, pool.map(get_single_security_metadata, identifiers))
        return result


@METADATA_DB_CACHE.cache_multi_on_arguments(namespace='security_ids', asdict=True, should_cache_fn=isnotnone)
def get_security_id(*identifiers):
    metadata = get_security_metadata(*identifiers)
    result = {}
    for identifier in identifiers:
        if metadata[identifier]['composite_id_bb_global'] != '':
            identifier_id = metadata[identifier]['composite_id_bb_global']
        elif metadata[identifier]['id_bb_global'] != '':
            identifier_id = metadata[identifier]['id_bb_global']
        elif metadata[identifier]['id_isin'] != '':
            identifier_id = f'\/isin\/{metadata[identifier]["id_isin"]}'
        else:
            raise NotImplementedError("Security id not found.")
        result[identifier] = identifier_id
    return result

@METADATA_DB_CACHE.cache_multi_on_arguments(namespace='full_metadata', asdict=True, should_cache_fn=isnotnone)
def get_security_metadata_full(*identifiers):
    metadata = get_security_metadata(*identifiers)
    addl_metadata_ids = set()
    for x in metadata:
        security_metadata = metadata[x]
        if security_metadata['prim_security_comp_id_bb_global'] != '':
            addl_metadata_ids.add(security_metadata['prim_security_comp_id_bb_global'])
        if security_metadata['composite_id_bb_global'] != '':
            addl_metadata_ids.add(security_metadata['composite_id_bb_global'])
        if security_metadata['id_bb_global_parent_co'] != '':
            addl_metadata_ids.add(security_metadata['id_bb_global_parent_co'])

    addl_metadata = get_security_metadata(*list(addl_metadata_ids))
    addl_metadata_result = {x: metadata_to_dict(addl_metadata[x]) for x in addl_metadata}
    for x in metadata:
        security_metadata = metadata[x]
        if security_metadata['prim_security_comp_id_bb_global'] != '':
            security_metadata['global_primary_security_composite_metadata'] = addl_metadata_result.get(security_metadata['prim_security_comp_id_bb_global'], {})
        if security_metadata['composite_id_bb_global'] != '':
            security_metadata['global_composite_security_metadata'] = addl_metadata_result.get(security_metadata['composite_id_bb_global'], {})
        if security_metadata['id_bb_global_parent_co'] != '':
            security_metadata['global_parent_metadata'] = addl_metadata_result.get(security_metadata['id_bb_global_parent_co'], {})
    return metadata

def process_bloomberg_fundamental_data(data):
    data = data.copy()
    tickers = sorted(set(data.columns.get_level_values(0)))
    dataset = []
    for ticker in tickers:
        ticker_data = data[ticker]
        ticker_data = ticker_data.stack().reset_index(drop=False).rename(columns={'level_1': 'field', 0: 'value'}).dropna()
        ticker_data['identifier'] = ticker
        dataset.append(ticker_data)
    try:
        dataset = pd.concat(dataset).sort_index().sort_values(by=['date', 'field']).reset_index(drop=True).dropna()
    except:
        return data
    return dataset

def get_batch_fundamental_data(batch):
    bbg = ApexBloomberg()
    group_result = bbg.fundamental_data(batch)
    daily = process_bloomberg_fundamental_data(group_result['daily'])
    quarterly = process_bloomberg_fundamental_data(group_result['quarterly'])
    dataset = pd.concat([daily, quarterly])
    return dataset

@MARKET_DATA_CACHING.cache_multi_on_arguments(namespace='sec_fnd_data', asdict=True, should_cache_fn=isnotnone)
def get_security_fundamental_data(*identifiers):
    result = []
    split = list(partition_all(5, identifiers))
    pool = ThreadPoolExecutor()
    for group in split:
        result.append(pool.submit(get_batch_fundamental_data, group))
    return pd.concat([x.result() for x in result]).reset_index(drop=True)


@METADATA_DB_CACHE.cache_multi_on_arguments(namespace='ticker_fix', asdict=True, should_cache_fn=isnotnone)
@retry(Exception, tries=3)
def fix_tickers_with_bbg(*ticker_list):
    bbg = ApexBloomberg()
    tickers = bbg.reference(ticker_list, 'parsekyable_des', kwargs=BLOOMBERG_METADATA_DEFAULT_KWARGS)
    from funcy import zipdict
    result = zipdict(ticker_list, tickers['parsekyable_des'].tolist())
    return result

def process_bloomberg_market_data(data):
    data = data.copy()
    adjusted_data = data['adjusted'].copy()
    unadjusted_data = data['unadjusted'].copy()
    adjusted_data = adjusted_data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'}, level=1)
    unadjusted_data = unadjusted_data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'}, level=1)
    tickers = sorted(set(adjusted_data.columns.get_level_values(0)))
    dataset = []
    for ticker in tickers:
        adj_ticker_data = adjusted_data[ticker]
        adj_ticker_data['returns'] = adj_ticker_data['returns'] / 100
        adj_ticker_data = adj_ticker_data.stack().reset_index(drop=False).rename(columns={'level_1': 'field', 0: 'value'}).dropna()
        adj_ticker_data['identifier'] = ticker
        adj_ticker_data['adjusted'] = True

        unadj_ticker_data = unadjusted_data[ticker]
        unadj_ticker_data['returns'] = unadj_ticker_data['returns'] / 100
        unadj_ticker_data = unadj_ticker_data.stack().reset_index(drop=False).rename(columns={'level_1': 'field', 0: 'value'}).dropna()
        unadj_ticker_data['identifier'] = ticker
        unadj_ticker_data['adjusted'] = False
        dataset.append(pd.concat([adj_ticker_data, unadj_ticker_data]).sort_index())
    dataset = pd.concat(dataset).sort_index().sort_values(by=['date', 'adjusted', 'field']).reset_index(drop=True)
    return dataset


def get_security_historical_data(*identifiers, fields=[]):
    if len(fields) == 0:
        raise ValueError("Need to specify fields")
    bbg = ApexBloomberg()
    split = partition_all(25, identifiers)
    result = []
    for group in split:
        try:
            result.append(bbg.history(group, fields))
        except:
            result.append(bbg.history(group, fields))
    result = pd.concat(result, axis=1)
    return result

@MARKET_DATA_CACHING.cache_multi_on_arguments(asdict=True, should_cache_fn=isnotnone)
def get_security_market_data(*identifiers):
    bbg = ApexBloomberg()
    result = {}
    for ticker in identifiers:
        ticker_data = bbg.market_data(ticker)
        result[ticker] = process_bloomberg_market_data(ticker_data)
    result = pd.concat(result).sort_values(by=['date', 'adjusted', 'field']).reset_index(drop=True)

    try:
        result['date'] = pd.DatetimeIndex(pd.to_datetime(result['date'])).tz_localize('America/Chicago', errors='coerce')
    except:
        result['date'] = pd.DatetimeIndex(pd.to_datetime(result['date'])).tz_convert('America/Chicago')

    return result[['date', 'identifier', 'field', 'value', 'adjusted']]

from joblib import Parallel, delayed, parallel_backend


@MARKET_DATA_CACHING.cache_multi_on_arguments(namespace='trade_status', asdict=True, should_cache_fn=should_cache_market_data_fn)
def apex__trade_status(*tickers):
    bbg = ApexBloomberg()
    data = bbg.reference(tickers, 'TRADE_STATUS')
    return (data['TRADE_STATUS'] == 'Y').to_dict()

def apex__adjusted_market_data(*tickers, parse=False, cache_key='apex_adjusted_mkt_data:2.4'):
    @MARKET_DATA_CACHING.cache_multi_on_arguments(namespace=cache_key, asdict=True, should_cache_fn=should_cache_market_data_fn)
    def _apex__adjusted_market_data(*identifiers):
        result = {}
        def get_data_fn(identifier):
            bbg = ApexBloomberg()
            return bbg.adjusted_market_data(identifier)

        with parallel_backend('threading', n_jobs=20):
            result = Parallel()(delayed(get_data_fn)(i) for i in identifiers)
        result = funcy.zipdict(identifiers, result)
        return result


    data = _apex__adjusted_market_data(*tickers)
    if parse:
        data = pd.concat(data.values(), axis=1)
        data = data.swaplevel(axis=1)
        data = data.sort_index(axis=1)
        data = data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'})
        data['returns'] = data['returns'] / 100

    return data


def apex__adjusted_market_data(*tickers, parse=False, cache_key='apex_adjusted_mkt_data:2.4'):
    from apex.toolz.bloomberg import MARKET_DATA_CACHING, should_cache_market_data_fn, parallel_backend, Parallel, delayed
    import funcy

    @MARKET_DATA_CACHING.cache_multi_on_arguments(namespace=cache_key, asdict=True, should_cache_fn=should_cache_market_data_fn)
    def _apex__adjusted_market_data(*identifiers):
        result = {}
        def get_data_fn(identifier):
            bbg = ApexBloomberg()
            return bbg.adjusted_market_data(identifier)

        with parallel_backend('threading', n_jobs=10):
            result = Parallel()(delayed(get_data_fn)(i) for i in identifiers)
        result = funcy.zipdict(identifiers, result)
        return result


    data = _apex__adjusted_market_data(*tickers)
    missing = [x for x in data if isinstance(data[x], bool)]
    tries=0
    while len(missing) > 0:
        data.update(_apex__adjusted_market_data(*missing))
        tries += 1
        if tries > 10:
            raise ConnectionError('Check connection to bloomberg.')

    if parse:
        data = pd.concat(data.values(), axis=1)
        data = data.swaplevel(axis=1)
        data = data.sort_index(axis=1)
        data = data.rename(columns={'day_to_day_tot_return_gross_dvds': 'returns'})
        data['returns'] = data['returns'] / 100

    return data

PORTFOLIO_ID_TO_ACCOUNT = {
    'U15262333-61': 'MN',
    'U15262333-25': 'SMMI',
    'U15262333-22': 'HEB',
    'U15262333-13': 'SMLPX',
    'U15262333-14': 'SMM',
    'U15262333-23': 'TR',
    'U15262333-20': 'TR TE',
    'U15262333-42': 'SMA',
    'U15262333-59': 'HIGH CONVICTION',
    'U15262333-18': 'LP',
}

ACCOUNT_TO_PORTFOLIO_ID = {k: v for v, k in PORTFOLIO_ID_TO_ACCOUNT.items()}
ACCOUNTS = set(ACCOUNT_TO_PORTFOLIO_ID.keys())

PORTFOLIO_ID_TO_INCEPTION_DATE = {
    'U15262333-61': '07/26/2017',
    'U15262333-25': '10/18/2016',
    'U15262333-22': '10/12/2016',
    'U15262333-13': '09/20/2015',
    'U15262333-14': '12/31/2015',
    'U15262333-23': '09/30/2015',
    'U15262333-20': '10/12/2016',
    'U15262333-42': '12/31/2015',
    'U15262333-59': '12/31/2015',
    'U15262333-18': '10/12/2016',
}


@INDEX_WEIGHTS_CACHING.cache_on_arguments(namespace='idx_members_8', should_cache_fn=isnotnone)
def get_index_member_weights_on_day(index, day):
    if index is None:
        raise NotImplementedError("Benchmark cannot be none.")
    bbg = ApexBloomberg()
    day = pd.to_datetime(day)
    try:
        weights = bbg.index_reference_data(index, 'INDX_MWEIGHT_HIST', kwargs={'END_DATE_OVERRIDE': day.strftime('%Y%m%d')})['INDX_MWEIGHT_HIST'][index]
    except TypeError:
        return None

    weights['Index Member'] += ' Equity'
    weights = weights.set_index('Index Member')['Percent Weight'] / 100
    return weights

def fix_security_name_index(df):
    df = df.copy()
    from apex.security import ApexSecurity
    result = {}
    for x in df.index:
        try:
            ticker = ApexSecurity.from_id(x).parsekyable_des
            result[x] = ticker
        except:
            pass
    new_indices = sorted(result.keys())
    df = df.loc[new_indices]
    df = df.rename(index=result)
    return df

def fix_security_name_columns(df):
    df = df.copy()
    from apex.security import ApexSecurity
    df.columns = [ApexSecurity.from_id(x).parsekyable_des for x in df.columns]
    return df


@INDEX_WEIGHTS_CACHING.cache_on_arguments(namespace='positions.v2')
def get_index_positions_on_day(index, day):
    if index is None:
        raise NotImplementedError("Benchmark cannot be none.")
    bbg = ApexBloomberg()
    day = pd.to_datetime(day)
    try:
        weights = bbg.reference(index, 'INDX_MWEIGHT_PX', kwargs={'END_DATE_OVERRIDE': day.strftime('%Y%m%d')}).iloc[0].iloc[0]
    except TypeError:
        return None

    weights['Index Member'] += ' Equity'
    return fix_security_name_index(weights.set_index('Index Member'))


from toolz import get_in
ticker_locs = [
    ['global_primary_security_composite_metadata', 'parsekyable_des'],
    ['global_primary_security_composite_metadata', 'primary_security_ticker'],
    ['global_composite_security_metadata', 'parsekyable_des'],
    ['global_composite_security_metadata', 'primary_security_ticker'],
    ['primary_security_ticker'],
    ['parsekyable_des'],
]

def get_correct_ticker(ticker):
    metadata_full = get_security_metadata_full(ticker)
    try:
        metadata_full = metadata_full[ticker]
    except:
        if len(metadata_full) == 1:
            t = list(metadata_full.keys())[0]
            metadata_full = metadata_full[t]
        elif len(metadata_full) == 0:
            return None
    for loc in ticker_locs:
        ticker = get_in(loc, metadata_full)
        if ticker == '':
            continue
        elif ticker is not None:
            return ticker
    return None


def get_index_members_multiday(index, start_date=None, end_date=None, freq='B'):
    """
    Gets index members
    """
    result = get_index_member_weights_multiday(index, start_date=start_date, end_date=end_date, freq=freq)
    result = valfilter(lambda x: x is not None, result)
    result = reduce(lambda x, y: x.union(y.index.tolist()), result.values(), set())
    result = sorted(result)
    result = [x for x in result if x is not None]
    # Now let's get metadata
    pool = ThreadPoolExecutor()
    result = pool.map(get_correct_ticker, result)
    return result

@PORTFOLIO_DB_CACHE.cache_on_arguments()
def get_bloomberg_portfolio_data_for_date(portfolio_id, date):
    bbg = ApexBloomberg()
    date = pd.to_datetime(date).strftime('%Y%m%d')
    res = bbg.portfolio_data(portfolio_id + ' Client', date)
    res.columns = res.columns.str.lower()
    return res.set_index('security')['position']

def p_bloomberg_portfolio_data_for_dates(portfolio_id, dates):
    pool = ThreadPoolExecutor()
    dates = [x.strftime('%Y%m%d') for x in dates]
    pids = [portfolio_id + ' Client' for x in dates]
    holdings = dict(zip(dates, pool.map(get_bloomberg_portfolio_data_for_date, pids, dates)))
    holdings = pd.DataFrame(holdings).T
    holdings.index = pd.to_datetime(holdings.index, format='%Y%m%d')
    return holdings

def get_account_members_on_date(account, day):
    portfolio_id = ACCOUNT_TO_PORTFOLIO_ID[account]
    holdings = get_bloomberg_portfolio_data_for_date(portfolio_id, day)
    holdings = holdings.drop(index='USD Curncy', errors='ignore')
    holdings.index = [get_security(ticker=x) for x in holdings.index]
    return set(holdings.index.tolist())


@INDEX_WEIGHTS_CACHING.cache_on_arguments()
@retry(exceptions=Exception, delay=2, tries=5)
def get_account_holdings_on_date(account, day):
    portfolio_id = ACCOUNT_TO_PORTFOLIO_ID[account]
    holdings = get_bloomberg_portfolio_data_for_date(portfolio_id, day)
    holdings = holdings.drop(index='USD Curncy', errors='ignore')
    return holdings


def get_index_member_weights_multiday(index, start_date=None, end_date=None, freq='B'):
    """
    Gets index member weights weights
    """
    today = pd.Timestamp.now(tz='America/Chicago')
    if start_date is None:
        start_date = pd.to_datetime('1/1/1996').tz_localize('America/Chicago')
    if end_date is None:
        end_date = today
    if index is None:
        raise NotImplementedError("Benchmark cannot be none.")
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).tz_localize('America/Chicago')
    date_range = pd.date_range(start_date, end_date, freq=freq)
    # Now let's split this into 5, and use dask client to do even better...

    result = []
    pool = ThreadPoolExecutor(max_workers=32)
    target = curry(get_index_member_weights_on_day)(index)
    result = dict(zip(date_range, pool.map(target, date_range)))
    return result
