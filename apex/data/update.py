import datetime
import re
import uuid
from pathlib import Path


import funcy
import numpy as np
import pandas as pd
import pendulum
import pyarrow as pa
import pyarrow.parquet as pq
from airflow import DAG
from airflow.contrib.hooks.redis_hook import RedisHook
from airflow.contrib.sensors.redis_key_sensor import RedisKeySensor
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from distributed import fire_and_forget
from toolz import juxt, partition_all

import apex.toolz.filetools
import joblib
from apex.security import ApexSecurity
from apex.toolz.bloomberg import (ApexBloomberg, get_security_fundamental_data,
                                  get_security_historical_data,
                                  get_security_id, get_security_metadata)
from apex.toolz.dask import ApexDaskClient, compute_delayed
from apex.toolz.data import (APEX_DATA_AVAILABLE_COLUMNS,
                             APEX_DATA_INDEX_COLUMNS)
from apex.toolz.databases import ApexDatabaseEngine, ApexDatabaseSession
from apex.toolz.dicttools import keys, values
from apex.toolz.downloader import ApexDataDownloader
from apex.toolz.pandas import sort_stacked_data
from apex.toolz.sampling import sample_indices, sample_values, ssample
from joblib import Parallel, delayed


def equities__store_equity_data(update_data, file_loc):
    update_data = update_data.pivot_table(index=['identifier', 'source', 'adjusted', 'date'], values='value', columns='field')
    table = pa.Table.from_pandas(update_data.reset_index(), preserve_index=False, nthreads=2)
    pq.write_table(table, file_loc)
    return True


def equities__pull_data_and_save(ticker, ds):
    security = ApexSecurity.from_id(ticker)
    sec_id = security.id
    filename = f'{sec_id}.parquet'
    file_loc = Path(f'/apex.data/daily_update/{ds}/raw/security_data/{filename}')
    if file_loc.exists():
        try:
            data = pd.read_parquet(file_loc)
            if len(data) > 0:
                return True
        except:
            pass
    dl = ApexDataDownloader()
    data = dl.equity_data_update(ticker)
    if data is None:
        return False
    if len(data.index) == len(data.columns) == 0:
        return True
    result = equities__store_equity_data(data, file_loc)
    return result

def mastering__master_security_data(filename, master_save_loc):
    data = pq.read_table(filename).to_pandas()
    try:
        identifiers = set(data.identifier)
        securities = {x: ApexSecurity.from_id(x) for x in identifiers}
    except:
        raise ValueError(f"Loading security from id caused exception. {identifiers}")
    for identifier in identifiers:
        identifier_id = securities[identifier].id
        filename = f'{identifier_id}.parquet'
        filename_h5 = f'{identifier_id}.h5'
        security_directory = Path(master_save_loc) / identifier_id
        security_directory.mkdir(exist_ok=True, mode=0o777)
        file_save_loc = Path(master_save_loc) / filename
        file_h5_save_loc = Path(master_save_loc) / filename_h5
        security_data = data[data.identifier == identifier]
        for col in APEX_DATA_AVAILABLE_COLUMNS:
            if col not in security_data.columns:
                security_data[col] = np.nan
        security_data = security_data[APEX_DATA_INDEX_COLUMNS + APEX_DATA_AVAILABLE_COLUMNS]
        if len(security_data.index) > 0:
            security_data.to_parquet(str(file_save_loc.absolute()), engine='fastparquet', compression='gzip')
            security_data.to_hdf(file_h5_save_loc, 'data', mode='w', format='table')
            for column in APEX_DATA_AVAILABLE_COLUMNS:
                col_save_loc = security_directory / (column + '.parquet')
                security_data[APEX_DATA_INDEX_COLUMNS + [column]].to_parquet(
                    security_directory / col_save_loc,
                    compression='gzip', engine='fastparquet')
    return True
