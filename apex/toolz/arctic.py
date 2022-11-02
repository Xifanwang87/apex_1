import threading
import typing
from collections import OrderedDict, deque
from functools import update_wrapper
from itertools import cycle
from threading import Thread

from dataclasses import dataclass, field
from toolz import curry
from wrapt import CallableObjectProxy

from apex.config import ApexConfig
from apex.toolz.deco import retry
from apex.toolz.singleton import Singleton
from arctic import Arctic
from arctic.exceptions import LibraryNotFoundException
import logging
import pandas as pd


logger = logging.getLogger('arctic')
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.CRITICAL)

def create_cnxn_pool():
    return cycle([Arctic(ApexConfig.arctic.database) for x in range(ApexConfig.arctic.pool_size)])

@dataclass
class _ArcticApexConnectionPool(metaclass=Singleton):
    """
    Problem: I want sessions that destroy themselves after they run out of scope.

    How?
    """
    _internal_session: Arctic = field(default_factory=create_cnxn_pool)

    def __post_init__(self):
        session = self.session
        libraries = session.list_libraries()
        for lib in libraries:
            if 'temporary' in lib.split(':'):
                date = lib['__ttl']
                if pd.Timestamp.now() > date:
                    session.delete_library(lib)

    def clean_temporary(self, force=True):
        session = self.session
        libraries = session.list_libraries()
        for lib in libraries:
            if lib.startswith('apex:temporary'):
                if force:
                    session.delete_library(lib)
                else:
                    date = lib['__ttl']
                    if pd.Timestamp.now() > date:
                        session.delete_library(lib)

    @property
    def session(self):
        return next(self._internal_session)

    @property
    def libraries(self):
        return frozenset(self.session.list_libraries())

    def __contains__(self, name):
        return name in self.libraries

    def __getitem__(self, name):
        return self.get_library(name)

    def __getattr__(self, name):
        return getattr(self.session, name)

    @retry(LibraryNotFoundException, delay=2, logger=ApexConfig.loggers.default)
    def initialize_library(self, library_name):
        """
        a) If library not in arctic, create it.
        b) Get pool
        c) Set pool to library.
        """
        session = self.session
        if library_name not in session.list_libraries():
            session.initialize_library(library_name)

        session.set_quota(library_name, 100 * 1024 * 1024 * 1024)

        return session[library_name]

    def get_library(self, library_name):
        return self.initialize_library(library_name)

    def library(self, library_name):
        return self.get_library(library_name)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and args[0] in self.libraries:
            return self.get_library(args[0])
        return self


_ArcticApexConnectionPool = _ArcticApexConnectionPool()
ArcticApex = update_wrapper(CallableObjectProxy(_ArcticApexConnectionPool), _ArcticApexConnectionPool)
ApexArctic = ArcticApex
