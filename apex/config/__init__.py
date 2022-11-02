from .base import ApexBaseConfig
from apex.toolz.storage import ApexMinio
import toml


def ApexServiceSettings():
    minio = ApexMinio()
    settings = minio.get('settings', 'services.toml').read()
    return ApexBaseConfig.from_toml(toml.read(settings))


DEFAULT_APEX_CONFIG = {
    'arctic': {
        'database': 'mongodb://10.15.201.154:27017/apex',
        'pool_size': 4
    },
    'postgresql': {
        'connection_string':
            'postgresql+psycopg2://apexsystem:apexsystem@10.15.201.154:5432/apex'
    },
    'pipelines': {
        'portfolio_construction': {
            'temp_dir': '/mnt/data/experiments/alpha/temp/'
        }
    }
}

ApexConfig = ApexBaseConfig.from_dict(name='default', data=DEFAULT_APEX_CONFIG)