from minio import Minio
from dataclasses import dataclass, field
import typing
from tempfile import TemporaryDirectory
import pyarrow.parquet as pq
import uuid
import pyarrow as pa
import pendulum
import pandas as pd

MINIO_URL = "10.15.201.153:14870"


@dataclass(frozen=True)
class ApexMinio:
    store: typing.Any = field(
        default_factory=lambda: Minio(MINIO_URL, access_key='apex', secure=False,
        secret_key='apexisawesome'))

    def get(self, bucket, name, prefix='apex'):
        r = self.store.get_object(f'{prefix}.{bucket}', name)
        return r

    def set(self, bucket, name, data, prefix='apex'):
        with TemporaryDirectory() as tmp:
            p = tmp + '/data'
            with open(p, 'wb') as fp:
                fp.write(data)
            self.store.fput_object(f'{prefix}.{bucket}', name, p)


def apex_equity_data_update_store(update_data, date=pendulum.now()):
    if isinstance(date, str):
        date = pendulum.parse(date)
    elif isinstance(date, pd.Timestamp):
        date = pendulum.parse(date.strftime('%Y-%m-%d'))

    update_data = update_data.pivot_table(index=['identifier', 'source', 'adjusted', 'date'], values='value', columns='field')
    tickers = sorted(set(update_data.index.get_level_values(0)))
    filename = uuid.uuid4().hex + '.parquet'
    table = pa.Table.from_pandas(update_data.reset_index(), preserve_index=False, nthreads=2)
    pq.write_table(table, f'/apex.data/daily_update/{date.to_date_string()}/raw/security_data/{filename}')
    return tickers
