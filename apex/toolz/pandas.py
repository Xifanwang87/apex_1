import io
import math
import time
from itertools import islice
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event

from .databases import ApexDatabaseEngine
from concurrent.futures import ThreadPoolExecutor

def timeseries_df_stacked(df, localize_df=True):
    df = df.copy()
    df.index.rename('date', inplace=True)
    df = df.stack().reset_index(drop=False).rename(columns={'level_1': 'field', 0: 'value'}).dropna()
    df = df.set_index('date')
    if localize_df:
        df = localize(df)
    return df


def localize(df, tz='America/Chicago'):
    try:
        df = df.tz_localize(tz)
    except:
        df = df.tz_convert(tz)
    return df

def sort_stacked_data(data):
    return data.sort_values(by=['adjusted', 'date', 'identifier', 'field']).reset_index(drop=True)


def fast_df_to_database(df, table, index=False):
    conn = ApexDatabaseEngine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=index)
    output.seek(0)
    cur.copy_from(output, table, null="")
    conn.commit()



class ApexDataframeConnector(object):
    """
    A class that works around the memory limitation that can be caused by inserting large pandas dataframes into a SQL storage.

    Example useage:

    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.random.random((100,10)))

    kwargs = {"db_type": 'postgresql',
        "address":'localhost',
        "user": "username",
        "password": "",
       "db_name": "testdb" }


    con = Connector(kwargs)
    con._init_engine() #If using other driver than pyodbc use con._init_engine(SET_FAST_EXECUTEMANY_SWITCH=False)
    con.set_df('test_df', df)
    df = con.get_df('test_df')
    df = con.query_df('SELECT * FROM test_df') #For when running specific queries while using the same connection.


    """
    def __init__(self, engine, logger = None):
        self.logger = logger
        self.engine = engine

    def __write_df(self, table_name, df, **kwargs):
        df.to_sql(table_name, con = self.engine, **kwargs)
        return True

    def __write_split_df(self, table_name, dfs, **kwargs):
        self.__write_df(table_name, dfs[0], **kwargs)
        kwargs.pop('if_exists')
        pool = ThreadPoolExecutor()
        futs = []
        for df in dfs[1:]:
            futs.append(pool.submit(self.__write_df, table_name, df, if_exists='append', **kwargs))
        for fut in futs:
            fut.result()
        return True

    def __split_df(self, df, chunksize):
        chunk_count = int(math.ceil(df.size / chunksize))
        return np.array_split(df, chunk_count)

    def set_df(self, table_name, df, if_exists = 'replace', chunksize = 10**6, **kwargs):
        s = time.time()
        status = False
        if chunksize is not None and df.size > chunksize:
            dfs = self.__split_df(df, chunksize)
            status = self.__write_split_df(table_name, dfs, if_exists = if_exists,  **kwargs)
        else:
            status = self.__write_df(table_name, df, if_exists = 'replace', **kwargs)
        if self.logger:
            self.logger.info('wrote name: {} dataframe shape: {} within: {}s'.format(table_name,
                                                                                     df.shape,
                                                                                     round(time.time()  - s, 4)))
        return status

    def get_df(self, table_name, chunk_count = None, **kwargs):
        s = time.time()
        if 'chunksize' not in kwargs.keys():
            kwargs['chunksize'] = 10**6
        dfs = pd.read_sql_table(table_name, self.engine, **kwargs)

        try:
            df = pd.concat(islice(dfs, chunk_count), axis = 0)
        except ValueError: #No objects to concetenate. dfs is a generator object so has no len() property!
            if self.logger:
                self.logger.warning("No objects to concetenate on table_name: {}".format(table_name))
            return None

        if self.logger:
            self.logger.info('fetched name: {} dataframe shape: {} within: {}'.format(table_name,
                                                                                      df.shape,
                                                                                      round(time.time()  - s, 4)))
        return df

    def get_table_names(self):
        if not hasattr(self, 'engine'):
            self._init_engine()
        df = pd.read_sql("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'", self.engine)
        return df


    def query(self, _query, logging = False):
        if not hasattr(self, 'engine'):
            self._init_engine()
        connection = self.engine.connect()
        result = connection.execute(_query)
        if logging and self.logger:
            if self.logger:
                self.logger.info('ran query: {}'.format(_query))
        connection.close()
        return result

    def df_query(self, _query):
        if not hasattr(self, 'engine'):
            self._init_engine()
        result = pd.read_sql_query(_query, con = self.engine)
        return result
