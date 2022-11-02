# from apex.toolz.dask import ApexDaskClient, compute_delayed
# from pathlib import Path
# import pandas as pd
# from apex.security import ApexSecurity
# from joblib import Parallel, delayed, parallel_backend
# from apex.toolz.dicttools import keys, values
# import funcy

# market_data_columns = {'px_last': 'close', 'px_high': 'high',
#                        'px_low': 'low', 'px_open': 'open',
#                        'px_volume': 'volume', 'returns': 'returns'}


# def get_column_data_in_file(file, column, source, adjusted):
#     base_cols = ['identifier', 'source', 'adjusted', 'date']
#     if isinstance(column, str):
#         column = [column]
#     assert len(column) == 1
#     cols = base_cols + column
#     data = pd.read_parquet(file, columns=cols, nthreads=5)
#     identifier = set(data['identifier'].values)
#     original_len = len(data)
#     filtered_data = data[(data.source == source) & (data.adjusted == adjusted)].reset_index(drop=True)
#     if len(filtered_data.index) == 0:
#         if original_len > 0:
#             if source == 'bloomberg':
#                 source = 'tiingo'
#             filtered_data = data[(data.source == source) & (data.adjusted == adjusted)].reset_index(drop=True)

#     data = filtered_data
#     try:
#         assert len(identifier) == 1
#     except AssertionError:
#         raise KeyError(f"Assertion failed for len identifiers. {file}")
#     identifier = identifier.pop()
#     data = data.set_index('date')[column]
#     data.columns = [identifier]
#     return data


# def column_data_futures(columns, securities=None, base_path=Path('/apex.data/security_data/master'), source='bloomberg', adjusted=True):
#     dask = ApexDaskClient()
#     files = list(base_path.glob('*.parquet'))
#     if securities is not None:
#         files = [x for x in files if x.name.split('.')[0] in securities]
#     result = [dask.submit(get_column_data_in_file, str(f.absolute()), columns, source, adjusted) for f in files]
#     return result

# def build_dataframe_in_thread(data):
#     with parallel_backend('threading', n_jobs=4):
#         data = Parallel()(delayed(x.result)() for x in data)
#     return pd.concat(data, axis=1)

# def get_datasets(columns, securities=None):
#     if isinstance(columns, str):
#         columns = [columns]
#     if securities is not None:
#         securities = [ApexSecurity.from_id(x) for x in securities]
#         security_ids = set(x.id for x in securities)
#     else:
#         security_ids = None
#     data = {}
#     for c in columns:
#         cdata = column_data_futures(c, securities=security_ids)
#         data[c] = cdata
#     cols = keys(data)
#     vals = values(data)
#     with parallel_backend('threading', n_jobs=4):
#         data = Parallel()(delayed(build_dataframe_in_thread)(v) for v in vals)
#     return pd.concat(funcy.zipdict(cols, data), axis=1)

# def get_security_market_data(securities):
#     return get_datasets(list(market_data_columns.keys()), securities=securities).rename(
#         columns=market_data_columns
#     )

# def get_security_returns(securities):
#     return get_datasets('returns', securities=securities)


from apex.toolz.dask import ApexDaskClient, compute_delayed
from pathlib import Path
import pandas as pd
from apex.security import ApexSecurity
from joblib import Parallel, delayed, parallel_backend
from apex.toolz.dicttools import keys, values
import funcy
import joblib

import time
import uuid
from apex.toolz.arctic import ArcticApex

market_data_columns = {'px_last': 'close', 'px_high': 'high', 'px_low': 'low', 'px_open': 'open', 'px_volume': 'volume', 'returns': 'returns'}

def cache_job_data(job_id, key, data):
    arctic = ArcticApex()
    library = arctic.get_library(f'apex:data_access:cache:{job_id}')
    library.write(key, data)
    return True

def get_column_data_in_file_cached(job_id, file, columns, adjusted):
    base_cols = ['adjusted', 'date']
    identifier = Path(file).name.split('.')[0]
    if isinstance(columns, str):
        columns = [columns]
    cols = base_cols + columns
    data = pd.read_parquet(file, columns=cols, use_threads=True)
    data = data[(data.adjusted == adjusted)].reset_index(drop=True).drop(columns=['adjusted'])
    data = data.groupby('date').mean()
    cache_job_data(job_id, identifier, data)
    return True

def column_data_futures(columns, securities=None, base_path=Path('/apex.data/security_data/master'), adjusted=True):
    if isinstance(columns, str):
        columns = [columns]
    directories = [x for x in base_path.glob('*') if x.is_dir()]
    files = []
    for security_id in securities:
        files.append(base_path / f'{security_id}.parquet')

    client = ApexDaskClient()
    job_id = uuid.uuid4().hex
    arctic = ArcticApex()
    library = arctic.get_library(f'apex:data_access:cache:{job_id}')

    start = pd.Timestamp.now()
    result = [client.submit(get_column_data_in_file_cached, job_id, str(f.absolute()), columns, adjusted) for f in files]
    result = [x.result() for x in result]
    assert all(result)
    # Now let's load cache
    symbols = library.list_symbols()
    result = pd.concat(funcy.zipdict(symbols, [library.read(x).data for x in symbols]))
    session = arctic.session
    session.delete_library(f'apex:data_access:cache:{job_id}')
    return result

def get_datasets(columns, securities=None):
    if isinstance(columns, str):
        columns = [columns]
    if securities is not None:
        securities = [ApexSecurity.from_id(x) for x in securities]
        security_ids = set(x.id for x in securities)
    else:
        security_ids = None
    data = column_data_futures(columns, securities=security_ids)
    return data.unstack(level=0)

def get_security_market_data(securities):
    return get_datasets(list(market_data_columns.keys()), securities=securities)

def get_security_returns(securities):
    return get_datasets('returns', securities=securities)
