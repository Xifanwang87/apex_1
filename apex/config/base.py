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

ApexParameterType = TypeVar(
    'ApexParameterType',
    typing.AnyStr,
    Number,
    typing.FrozenSet,
    typing.NamedTuple,
    typing.Tuple,
)

@dataclass(frozen=True)
class ApexConfigParameter:
    name: str
    _value: ApexParameterType

    @property
    def value(self):
        return self._value

    @property
    def tuple(self):
        return (self.name, self.value)

    def to_dict(self):
        return {'name': self.name, 'value': self.value}

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()


class ApexFileWatcher(StoppableThread):
    def __init__(self, path, callback):
        self.callback = callback
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        return super().__init__()

    def run(self):
        """
        Runs method
        """
        last_modified = self.path.lstat().st_mtime
        while not self.stopped:
            if last_modified == self.path.lstat().st_mtime:
                time.sleep(1)
                continue
            last_modified = self.path.lstat().st_mtime
            self.callback(self.path)

@dataclass
class ApexBaseConfig:
    name: str
    value: typing.Mapping
    path: str
    watcher: typing.Any = field(init=False)
    loggers: typing.Mapping = field(init=False)

    def __del__(self):
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher.join()

    def __post_init__(self):
        if self.path is not None:
            if isinstance(self.path, str):
                self.path = Path(self.path)
            self.watcher = ApexFileWatcher(path=self.path, callback=self.update)
            self.watcher.start()
        else:
            self.watcher = None

        self.loggers = addict({'default': logging.getLogger('apex:logging:default')})


    def update(self, path):
        assert self.path == path
        with open(self.path, 'r') as f:
            data = toml.load(f)
            value = ApexBaseConfig.parse_dict(data)
            self.value = value

    @staticmethod
    def parse_dict(data: typing.Mapping):
        result = OrderedDict()

        for k, v in data.items():
            if isinstance(v, typing.Mapping):
                v = ApexBaseConfig.from_dict(k, v, path=None)
                result[k] = v
            else:
                result[k] = ApexConfigParameter(k, v)
        return result

    def __getattr__(self, name):
        for p in self.value:
            if p == name:
                if isinstance(self.value[p].value, OrderedDict):
                    return self.value[p]
                else:
                    return self.value[p].value
        raise AttributeError(f"Cannot find {name}")

    @classmethod
    def from_dict(cls, name: str, data: typing.Mapping, path: typing.Any = None):
        result = ApexBaseConfig.parse_dict(data)
        return cls(name=name, value=result, path=path)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.value.items():
            if isinstance(v, ApexBaseConfig):
                v = v.to_dict()
            elif isinstance(v, ApexConfigParameter):
                v = v.to_dict()
                k = v['name']
                v = v['value']
            elif isinstance(v, typing.Mapping):
                v = dict(**v)
            result[k] = v
        return result

    @classmethod
    def from_toml(cls, name, path):
        with open(path, 'r') as f:
            data = toml.load(f)
            return cls.from_dict(name=name, data=data, path=path)

    @classmethod
    def default(cls):
        raise NotImplementedError("Need to implement defaults in child.")

    def __str__(self):
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)

    def __repr__(self):
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)