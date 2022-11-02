import numpy as np
from collections import Counter
import toolz
import pandas as pd


def _get_indices_pandas(data, indices):
    return data.iloc[indices].index.tolist()

def _get_indices_identity(data, indices):
    return list(indices)

def _get_values_pandas(data, indices):
    return data.loc[indices]

def _get_values_other(data, indices):
    return toolz.get(indices, data)


def _get_sample(arr, sample_size):
    """Get random sample from arr.

    Parameters
    ----------
    arr: np.array
        array to sample from.
    n_iter: int
        current iteration number.
    sample_size: int
        sample size
    fast: bool
        use sampling optimized for fast consecutive samples
        from the same array.

    Returns
    -------
    sample: np.array
        sample from arr of length n_iter.
    """
    # find the index we last sampled from
    return np.random.choice(arr, sample_size, replace=False)


def sample_indices(
        data,
        sample_size,
        n_samples):
    """
    Collect several samples of indices from data.

    Parameters
    ----------
    arr: np.array
        array to sample from.
    sample_size: int
        sample size.
    n_samples: int
        number of samples to take.
    fast: bool
        use sampling optimized for fast consecutive samples
        from the same array.

    Returns
    -------
    samples: np.ndarray
        sample matrix of shape (n_samples, sample_size)
    """
    arr = np.arange(len(data))
    if isinstance(data, (pd.Series, pd.DataFrame)):
        apply_fn = _get_indices_pandas
    else:
        apply_fn = _get_indices_identity
    result = []
    arr_len = len(arr)
    for sample_n in range(0, n_samples):
        sample = _get_sample(arr,
                            sample_size)
        result.append(apply_fn(data, sample))
    return result

def sample_values(
        data,
        sample_size,
        n_samples):
    """
    Collect several samples of indices from data.

    Parameters
    ----------
    arr: np.array
        array to sample from.
    sample_size: int
        sample size.
    n_samples: int
        number of samples to take.
    fast: bool
        use sampling optimized for fast consecutive samples
        from the same array.

    Returns
    -------
    samples: np.ndarray
        sample matrix of shape (n_samples, sample_size)
    """
    sampled_indices = sample_indices(data, sample_size, n_samples)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        val_fn = toolz.curry(_get_values_pandas)(data)
    else:
        val_fn = toolz.curry(_get_values_other)(data)

    return list(map(val_fn, sampled_indices))


def ssample(
        data,
        sample_size):
    """
    Collect several samples of indices from data.

    Parameters
    ----------
    arr: np.array
        array to sample from.
    sample_size: int
        sample size.
    n_samples: int
        number of samples to take.
    fast: bool
        use sampling optimized for fast consecutive samples
        from the same array.

    Returns
    -------
    samples: np.ndarray
        sample matrix of shape (n_samples, sample_size)
    """
    sampled_indices = sample_indices(data, sample_size, 1)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        val_fn = toolz.curry(_get_values_pandas)(data)
    else:
        val_fn = toolz.curry(_get_values_other)(data)
    return val_fn(sampled_indices[0])



def ivalsample(
        data,
        sample_size,
        n_samples):
    """
    Collect several samples of indices from data.

    Parameters
    ----------
    arr: np.array
        array to sample from.
    sample_size: int
        sample size.
    n_samples: int
        number of samples to take.
    fast: bool
        use sampling optimized for fast consecutive samples
        from the same array.

    Returns
    -------
    samples: np.ndarray
        sample matrix of shape (n_samples, sample_size)
    """
    sample_indices(data, sample_ize, n_samples)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        val_fn = toolz.curry(_get_values_pandas)(data)
    else:
        val_fn = toolz.curry(_get_values_other)(data)

    result = map(val_fn, sampled_indices)
    for sample in result:
        yield sample
