import concurrent.futures as cf
import typing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import attr
from dask.delayed import Delayed
from dask.distributed import Client, Future
from distributed.actor import ActorFuture
from typing import Union
from collections.abc import Container, Mapping, Sequence
from toolz import valmap
from distributed.diagnostics.plugin import SchedulerPlugin


def ddict():
    return defaultdict(ddict)

def ApexDaskClient():
    return Client()


FutureKind = (Future, cf.Future, ActorFuture)
FutureOrDelayed = (Future, cf.Future, ActorFuture, Delayed)


def compute_future_value(d):
    assert isinstance(d, FutureOrDelayed)
    if isinstance(d, FutureKind):
        if d.done():
            return d.result()
        else:
            return d
    elif isinstance(d, Delayed):
        return d.compute()

def walk_through_container(d: Container):
    result = ddict()
    futures = []
    if isinstance(d, Sequence):
        d = dict(enumerate(d))
    for k, v in d.items():
        if isinstance(v, Container):
            v, futs = walk_through_container(v)
            result[k] = v
            futures += futs
        else:
            v = compute_future_value(v)
            if isinstance(v, FutureKind):
                futures.append(v)
            result[k] = v
    return result, futures

def compute_delayed(d, retry_count=0, exceptions='raise'):
    if isinstance(d, FutureOrDelayed):
        return compute_future_value(d)

    # Now let's walk through dicts and keep track of what we are waiting for.
    while True:
        if isinstance(d, Container):
            result, futures = walk_through_container(d)
            if len(futures) == 0:
                return result
        else:
            raise NotImplementedError("Unknown kind for d.")
