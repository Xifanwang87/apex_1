from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from apex.config import ApexConfig

ApexDatabaseEngine = create_engine(
    ApexConfig.postgresql.connection_string,
    pool_size=64, max_overflow=64)

# For ease
_ApexDatabaseSessionMaker = sessionmaker(bind=ApexDatabaseEngine)


def ApexDatabaseSessionMaker():
    return _ApexDatabaseSessionMaker

def ApexDatabaseSession():
    return ApexDatabaseSessionMaker()
