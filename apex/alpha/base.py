from apex.config.base import ApexBaseConfig
from dataclasses import dataclass, field
from apex.data.access import get_security_market_data


@dataclass
class ApexAlphaConfig(ApexBaseConfig):
    pass

ApexDefaultAlphaConfig = ApexAlphaConfig.from_dict(name='ApexDefaultHyperparameters', data={
    'signal_smoothing': 1,
    'weight_transform': {
        'method': 'mean',
        'period': 1
    },
    'score_transform': {
        'method': 'ewm',
        'period': 2
    },
    'portfolio_construction': 'risk_budget',
    'should_rank': False,
    'long_short_deciles': {'long': {9, 8, 7, 6, 5}, 'short': {0, 1, 2, 3, 4}},
    'universe': 'AMNA'
})

@dataclass
class ApexAlpha:
    """
    columns = which columns the alpha needs to work. Used on get_datasets.
    max_lookback is the size of dataset to come as input
    """
    name: str
    columns: frozenset

    config: ApexAlphaHyperparameters = field(default=ApexDefaultHyperparameters)
    max_lookback: int = field(default=256)
    blackboard: dict = field(default_factory=dict)
    callable: typing.Any = field(default=None)

    def __post_init__(self):
        universe = APEX_UNIVERSES[self.config.universe]
        self.blackboard['market_data'] = get_security_market_data(universe)

    def set_blackboard(self, blackboard):
        self.blackboard = blackboard

    def compute(self, dataset):
        """
        Will compute it for a single day.
        """
        score = self.score(dataset)
        smoothed_score = self.smooth_score(score)
        ranked_scores = self.rank(smoothed_score)
        decile_portfolio = self.build_decile_portfolio(ranked_scores)
        portfolio = self.validate_holdings(portfolio, dataset)
        return portfolio

    def score(self, dataset):
        return call_market_data_alpha_fn(self.callable, dataset)

    def smooth_score(self, score):
        fn_name, period = self.hyperparameters.score_transform
        smoothing_fn = SMOOTHING_METHODS[fn_name]
        return smoothing_fn(score, period=period)

    def rank(self, series):
        """
        Applies timeseries ranking to alpha
        """
        if self.hyperparameters.should_rank:
            signal = rank(series)
        return series

    def build_decile_portfolio(self, scores):
        deciles = (scores.rank(axis=1, pct=True) * 9).fillna(-1).astype(int)
        long_scores = self.hyperparameters.get('long_short_deciles').get('long', set())
        short_scores = self.hyperparameters.get('long_short_deciles').get('short', set())
        if long_scores is None and short_scores is None:
            return scores
        long_deciles = deciles.isin(long_scores)
        short_deciles = deciles.isin(short_scores)
        portfolio = long_deciles.astype(int) - short_deciles.astype(int)
        portfolio[deciles == -1] = np.nan
        return portfolio

    def validate_holdings(self, series, dataset):
        assert 'px_last' in dataset
        data = dataset['px_last'].iloc[-1]
        data = data.dropna()
        series = series.reindex(data.index)
        return series
