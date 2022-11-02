from dataclasses import dataclass
import typing
import itertools

@dataclass
class ApexTransformParameterFV:
    """
    FV for Fixed Values
    """
    name: str
    values: typing.Sequence

ApexSmoothingParameter = ApexTransformParameterFV(
    name='smoothing',
    values=[
    3, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 150, 200, 250
])
ApexShouldRankParameter = ApexTransformParameterFV(
    name='should_rank',
    values=[True, False]
)


@dataclass
class ApexParameter:
    name: str
    def __iter__(self):
        """
        Iterates over parameters
        """

@dataclass
class ApexTransform:
    """
    Transform(DF) -> DF

    We'll be using it
    """
    name: str
    callable: typing.Callable
    parameters: dict
    def __call__(self, data, **params):
        return self.callable(data, **params)

def rolling_mean(df, period=10):
    return df.rolling(period).mean()

def ewm(df, period=10):
    return df.ewm(span=period).mean()

def rolling_median(df, period=10):
    return df.rolling(period).median()

def rank(df, subtract_mean=True):
    result = df.rank(axis=1, pct=True)
    if subtract_mean:
        result = result.subtract(result.mean(axis=1), axis=0)
    return result

@dataclass
class ApexCompositeTransform:
    """
    The composite transform applies transforms on
    """
    transforms: typing.Sequence
    def __call__(self, data, **kwargs):
        result = data.copy()
        for transform in self.transforms:
            parameters = next(transform.parameters)
            result = transform(result,)

ApexSmoothingMean = ApexTransform(
    name='rolling_mean',
    callable=rolling_mean,
    parameters={'period': ApexSmoothingParameter,}
)
ApexSmoothingMedian = ApexTransform(
    name='rolling_median',
    callable=rolling_median,
    parameters={'period': ApexSmoothingParameter}
)

ApexSmoothingEWM = ApexTransform(
    name='span_ewm',
    callable=ewm,
    parameters={'period': ApexSmoothingParameter}
)

ApexRankingTransform = ApexTransform(
    name='cross_sectional_rank',
    callable=rank,
    prameters={'subtract_mean': ApexShouldRankParameter}
)

