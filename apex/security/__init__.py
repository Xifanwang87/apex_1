from apex.toolz.singleton import SingletonByAttribute
from apex.toolz.bloomberg import (
    get_security_metadata,
)
from dataclasses import dataclass, field
from toolz import memoize
from types import MappingProxyType
import os
import re
import types
import typing
from collections import OrderedDict
from numbers import Number
from pathlib import Path
from typing import Generic, Iterable, Tuple, TypeVar, Any

import attr
import jinja2
import toml
import yaml
from dataclasses import dataclass, field
from toolz import curry
from threading import Thread
import time
import threading
from pathlib import Path
import logging
from addict import Dict as addict
from apex.toolz.singleton import Singleton, SingletonByAttribute
from apex.toolz.bloomberg import BLOOMBERG_METADATA_DEFAULT_KWARGS, ApexBloomberg
import dogpile.cache as dc
from apex.toolz.functools import isnotnone
from apex.toolz.deco import retry
from apex.toolz.caches import METADATA_DB_CACHE


@METADATA_DB_CACHE.cache_on_arguments()
@retry(Exception, tries=3)
def identifier_metadata(identifier):
    METADATA = [
        'cntry_issue_iso',
        'composite_exch_code',
        'composite_id_bb_global',
        'eqy_fund_ticker',
        'eqy_prim_security_comp_exch',
        'eqy_prim_security_crncy',
        'eqy_prim_security_prim_exch',
        'eqy_prim_security_ticker',
        'id_bb_company',
        'id_bb_sec_num',
        'id_bb_global',
        'id_bb_prim_security_flag',
        'id_bb_unique',
        'id_bb',
        'id_cusip',
        'id_exch_symbol',
        'id_full_exchange_symbol',
        'id_isin',
        'id_mic_prim_exch',
        'id_sedol1',
        'id_stock_exchange',
        'id_wertpapier',
        'parsekyable_des_source',
        'parsekyable_des',
        'prim_security_comp_id_bb_global',
        'quoted_crncy',
        'market_status',
        'ticker_and_exch_code',
        'ticker',
        'market_sector_des'
    ]
    bbg = ApexBloomberg()
    res = bbg.reference(identifier, METADATA, kwargs=BLOOMBERG_METADATA_DEFAULT_KWARGS).T.fillna('').to_dict()[identifier]
    return res

ApexParameterType = TypeVar(
    'ApexParameterType',
    typing.AnyStr,
    Number,
    typing.FrozenSet,
    typing.NamedTuple,
    typing.Tuple,
)

@dataclass(frozen=True)
class ApexMetadataItem:
    name: str
    _value: ApexParameterType

    def __repr__(self):
        return f'\t{self.name}: {self.value}'

    def __str__(self):
        return repr(self)

    @property
    def value(self):
        return self._value

    @property
    def tuple(self):
        return (self.name, self.value)

    def to_dict(self):
        return {'name': self.name, 'value': self.value}

@dataclass
class ApexSecurityMetadata:
    identifier: str
    value: typing.Mapping = field(default=None)

    @staticmethod
    def identifier_metadata(identifier):
        return identifier_metadata(identifier)

    @staticmethod
    def parse_security_metadata(data: typing.Mapping):
        result = OrderedDict()

        for k, v in data.items():
            if isinstance(v, typing.Mapping):
                v = ApexSecurityMetadata.from_dict(k, v)
                result[k] = v
            else:
                result[k] = ApexMetadataItem(k, v)
        return result

    @classmethod
    def from_dict(cls, name: str, data: typing.Mapping):
        result = ApexSecurityMetadata.parse_security_metadata(data)
        return cls(identifier=name, value=result)

    @classmethod
    def from_id(cls, identifier):
        data = ApexSecurityMetadata.identifier_metadata(identifier)
        return ApexSecurityMetadata.from_dict(identifier, data)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.value.items():
            if isinstance(v, ApexSecurityMetadata):
                v = v.to_dict()
            elif isinstance(v, ApexMetadataItem):
                v = v.to_dict()
                k = v['name']
                v = v['value']
            elif isinstance(v, typing.Mapping):
                v = dict(**v)
            result[k] = v
        return result

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, name):
        for p in self.value:
            if p == name:
                if isinstance(self.value[p].value, OrderedDict):
                    return self.value[p]
                else:
                    return self.value[p].value
        raise AttributeError(f"Cannot find {name}")

@dataclass
class ApexSecurity:
    security_metadata: ApexSecurityMetadata
    @classmethod
    def from_id(cls, identifier):
        try:
            metadata = ApexSecurityMetadata.from_id(identifier)
            return ApexSecurity.from_metadata(metadata)
        except RecursionError:
            metadata = ApexSecurityMetadata.from_id(identifier)
            result = cls(security_metadata=metadata)
            return result

    def __hash__(self):
        return hash(self.id)

    @property
    def metadata(self):
        return self.security_metadata

    @classmethod
    def from_metadata(cls, metadata: ApexSecurityMetadata):
        result = cls(security_metadata=metadata)
        return result.composite_security()

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.security_metadata.value.items():
            if isinstance(v, ApexSecurityMetadata):
                v = v.to_dict()
            elif isinstance(v, ApexMetadataItem):
                v = v.to_dict()
                k = v['name']
                v = v['value']
            elif isinstance(v, typing.Mapping):
                v = dict(**v)
            result[k] = v
        return result

    def __getattr__(self, name):
        try:
            return getattr(self.security_metadata, name)
        except AttributeError:
            raise
        except:
            raise AttributeError(f"Cannot find {name}")

    def __str__(self):
        return f'ApexSecurity({self.parsekyable_des})'

    def __repr__(self):
        header = f'ApexSecurity({self.parsekyable_des})\n'
        data = yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)
        data = data.split('\n')
        data = ['\t' + x for x in data]
        data = '\n'.join(data)
        return header + data

    @property
    def id(self):
        return self.id_bb_global

    def is_primary(self):
        return self.id_bb_prim_security_flag == 'Y'

    def is_composite(self):
        test_b = self.ticker_and_exch_code.split(' ')[1] == self.composite_exch_code
        return test_b

    def market_data(self, adjusted=True):
        from apex.toolz.bloomberg import apex__adjusted_market_data
        data = apex__adjusted_market_data(self.parsekyable_des, parse=True).swaplevel(axis=1)[self.parsekyable_des]
        return data


    def composite_security(self):
        """
        Tough logic.

        1. If is composite and not empty, return self
        2. Otherwise, if composite id is not empty, return that. It covers acquisitions/bankruptcies/etc.
        3. Otherwise, if ticker . change and
        """
        try:
            assert self.market_sector_des == 'Equity'
        except AssertionError:
            return self

        if self.is_composite() and self.is_primary():
            return self

        if self.composite_id_bb_global != '':
            security = ApexSecurity.from_id(self.composite_id_bb_global)
            return security

        if self.market_status == 'TKCH':
            sec_num = self.id_bb_sec_num
            if self.eqy_fund_ticker != '':
                security = ApexSecurity.from_id(self.eqy_fund_ticker + ' Equity')
                if security.id_bb_sec_num == sec_num:
                    return security
            if self.prim_security_comp_id_bb_global != '':
                security = ApexSecurity.from_id(self.prim_security_comp_id_bb_global)
                if security.id_bb_sec_num == sec_num:
                    return security
        else:
            if self.id_bb_global != '':
                return self

        return self

@memoize
def get_security(ticker=None):
    return ApexSecurity.from_id(ticker)