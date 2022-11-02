import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.sensors.redis_key_sensor import RedisKeySensor
from airflow.contrib.hooks.redis_hook import RedisHook
from toolz import partition_all

from apex.toolz.data_sources import ApexDataDownloader
from apex.toolz.bloomberg import ApexBloomberg, get_security_fundamental_data, get_security_metadata
from apex.toolz.sampling import sample_values, ssample, sample_indices
from apex.toolz.dicttools import keys, values
from apex.toolz.dask import ApexDaskClient, compute_delayed
from pathlib import Path
import uuid
import re
from apex.pipelines.statistics.covariance import compute_and_save_covariance_matrices
import pandas as pd
import toolz
import funcy
from apex.store import ApexDataStore


default_args = {
    'owner': 'Eduardo Sahione',
    'email': ['esahione@salientpartners.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': datetime.timedelta(minutes=1),
    'catchup': False,
    'pool': 'data_processing'
}

ApexSystemRun = DAG(
    dag_id='apex.daily.system.data.processing',
    description='Apex System Run',
    start_date=datetime.datetime(2018, 10, 12),
    catchup=False,
    schedule_interval=None,
    default_args=default_args,
    concurrency=8,
)

def apex_compute_covariance_matrices_task(days, **context):
    result = compute_and_save_covariance_matrices(days)
    return result


def covariance_matrix_task(dag):
    dummy = DummyOperator(task_id='apex.daily.system.run.covariance_matrix', dag=dag)
    done = DummyOperator(task_id='apex.daily.system.run.covariance_matrix.done', dag=dag)

    store = ApexDataStore()
    library = ArcticApex.library('apex.daily.data_cache')
    store_date = store.date
    returns = store.returns()
    days = returns.index
    batches = list(toolz.partition_all(252, days))
    task_id_template = 'apex.daily.compute.statistics.covariance_matrix.{start}_{end}'
    tasks = [dummy]
    for batch in batches:
        start_date = batch[0]
        end_date = batch[-1]
        task_id = task_id_template.format(
            start=start_date.strftime('%Y%m%d'),
            end=end_date.strftime("%Y%m%d"))
        task = PythonOperator(
            task_id=task_id,
            dag=dag,
            python_callable=apex_compute_covariance_matrices_task,
            op_args=[batch],
            provide_context=True
        )
        task.set_upstream(dummy)
        task.set_downstream(done)
        tasks.append(task)
    tasks.append(done)
    return done, tasks

covariance_matrix_task_done, subtasks = covariance_matrix_task(ApexSystemRun)