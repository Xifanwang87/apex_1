import functools
import threading
import copy
from collections import OrderedDict

def curry_with_lock(fn, lock):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with lock:
            return fn(*args, **kwargs)
    return wrapped


def synchronized(lock):
    """ Synchronization decorator """
    def wrapper(f):
        return curry_with_lock(f, lock)
    return wrapper

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class LockedSingleton(type):
    _instances = {}
    _locks = {}
    _cls_name = {}
    def __new__(meta, name, bases, dct):
        if name not in meta._locks:
            meta._locks[name] = threading.Lock()
        for k, v in dct.items():
            if callable(v):
                dct[k] = curry_with_lock(v, meta._locks[name])
        wrapped_cls = super().__new__(meta, name, bases, dct)
        meta._cls_name[wrapped_cls] = name
        return wrapped_cls

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

from collections import defaultdict

def SingletonByAttribute(*attributes):
    class SingletonByAttribute(type):
        __singleton_instance_attributes = frozenset(copy.deepcopy(attributes))
        __singleton_instances = defaultdict(dict)
        def __call__(cls, *args, **kwargs):
            k = []
            attributes_found = set()
            for attribute in sorted(cls.__singleton_instance_attributes):
                if attribute in kwargs:
                    attributes_found.add(attribute)
                    k.append(f'{attribute}={kwargs[attribute]}')
            k = ':'.join(k)
            if len(attributes_found) == len(cls.__singleton_instance_attributes):
                if k in cls.__singleton_instances[cls]:
                    return cls.__singleton_instances[cls][k]

            k = []
            instance = super().__call__(*args, **kwargs)
            for attribute in sorted(cls.__singleton_instance_attributes):
                k.append(f'{attribute}={getattr(instance, attribute)}')
            k = ':'.join(k)
            if k in cls.__singleton_instances[cls]:
                return cls.__singleton_instances[cls][k]
            else:
                cls.__singleton_instances[cls][k] = instance
                return instance
    return SingletonByAttribute