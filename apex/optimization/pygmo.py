from apex.toolz.dask import ApexDaskClient
from threading import Lock as _Lock


def _evolve_func(algo, pop):
    # The evolve function that is actually run from the separate processes
    # in both mp_island (when using the pool) and ipyparallel_island.
    new_pop = algo.evolve(pop)
    return algo, new_pop


class apex_dask_island:
    """
    Dask island.
    """

    def __init__(self, client=ApexDaskClient(), *args, **kwargs):
        self._dask_clt = client
        self._init(*args, **kwargs)

    def _init(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._view_lock = _Lock()

    def __copy__(self):
        return apex_dask_island(*self._args, **self._kwargs)

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        return self._args, self._kwargs

    def __setstate__(self, state):
        self._init(*state[0], **state[1])

    def run_evolve(self, algo, pop):
        """Evolve population.
        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        task is submitted to the ipyparallel cluster via an internal :class:`ipyparallel.LoadBalancedView`
        instance initialised during the construction of the island.
        Args:
            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm
        Returns:
            tuple: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`
        Raises:
            unspecified: any exception thrown during the evolution, or by submitting the evolution task
              to the ipyparallel cluster
        """
        ret = self._dask_clt.submit(_evolve_func, algo, pop)
        return ret.result()

    def get_name(self):
        """Island's name.
        Returns:
            str: ``"Ipyparallel island"``
        """
        return "Dask Island"
