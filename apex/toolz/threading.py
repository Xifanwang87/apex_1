from concurrent.futures import ThreadPoolExecutor
from typing import Container

def prep_inputs_for_threading(inputs):
    result = {}
    length = 0
    for ix, inval in enumerate(inputs):
        if isinstance(inval, Container):
            result[ix] = inval
            length = max(length, len(inval))

    for ix, inval in enumerate(inputs):
        if not isinstance(inval, Container):
            result[ix] = [inval for x in range(length)]

    return [result[x] for x in range(len(inputs))]

def threaded__dictmap(fn, keys, *inputs):
    pool = ThreadPoolExecutor()
    pool_inputs = prep_inputs_for_threading(inputs)
    result = pool.map(fn, *pool_inputs)
    result = [x for x in result]
    return dict(zip(keys, result))