from dataclasses import dataclass, field
import typing
import itertools
from funcy import zipdict
from itertools import product
import pendulum


@dataclass
class ApexExperimentRun:
    experiment_id: int
    parameters: dict = field(default=None)
    def __post_init__(self):
        mlflow.set_tracking_uri("http://10.15.201.160:18001")
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            for param, param_val in self.parameters.items():
                run.log_param(param, param_val)
            results = self.run()
            for metric_name, metric_value in results.items():
                run.log_metric(metric_name, metric_value)

    def record(self):
        raise NotImplementedError

@dataclass
class ApexExperiment:
    name: str
    experiment_id: int
    def __post_init__(self):
        self.experiment_id = mlflow.create_experiment(self.name)
        self.run()

    def run(self):
        raise NotImplementedError