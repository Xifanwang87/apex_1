import typing

from dataclasses import dataclass, field
from distributed import fire_and_forget

from apex.data.access import get_security_market_data
from apex.toolz.dask import ApexDaskClient
from apex.toolz.experiment import ApexExperiment, ApexExperimentRun
from apex.universe import APEX_UNIVERSES

from .market_alphas import ApexMarketDataAlpha


@dataclass
class ApexMarketAlphaRun(ApexExperimentRun):
    """
    Available hyperparameters:
        - delay -> how much to delay the final signal before budgeting portfolio
        - 
    """
    data: typing.Any
    transforms: list
    def run(self, data):
        result = data
        for transform in self.transforms:
            result = transform(result)
        portfolio = self.create_risk_budget_portfolio(result)
        stats = self.compute_stats(portfolio)
        return stats

    def compute_stats(self, portfolio): pass
    def create_risk_budget_portfolio(self, scores): pass

def create_alpha_run(name, alpha, experiment_id, transforms, parameters):
    run = ApexMarketAlphaRun(name=name,
        alpha=alpha,
        experiment_id=experiment_id,
        transforms=transforms,
        parameters=parameters).run(data)
    return run


@dataclass
class ApexMarketAlphaExperiment(ApexExperiment):
    alpha: ApexMarketDataAlpha
    universe: str
    transforms: dict = field(default_factory=dict)
    @property
    def securities(self):
        return APEX_UNIVERSES[self.universe]

    def load_data(self):
        data = get_security_market_data(self.securities)
        return data

    def run(self):
        data = self.load_data()
        dask = ApexDaskClient()
        scattered = dask.scatter(data)
        alpha_transforms = [self.alpha] + self.transforms
        fire_and_forget(
            dask.submit(
                create_alpha_run,
                self.alpha,
                self.experiment_id,
                self.universe,
                alpha_transforms
            )
        )
