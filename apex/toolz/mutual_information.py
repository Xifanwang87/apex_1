from sklearn.linear_model import HuberRegressor
import sklearn.covariance as sk_cov
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.neural_network import MLPRegressor
import attr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from dataclasses import dataclass, field
import typing

@dataclass
class ApexMutualInformationAnalyzer:
    """
    Automates analysis of correlation/mutual information between two streams of data.
    """
    X: typing.Any
    y: typing.Any
    analysis_window: int = field(default=252*2)

    def __post_init__(self):
        X = self.X
        y = self.y
        X = pd.DataFrame({'target': y, 'factor_returns': X}).dropna()
        y = X['target']
        X = X[['factor_returns']]
        y = y
        self.X = X.iloc[-self.analysis_window:]
        self.y = y.iloc[-self.analysis_window:]

    def linear_beta(self, constant=False):
        X = self.X.copy()
        if constant:
            X = sm.add_constant(X, prepend=False)
        model = HuberRegressor().fit(X, self.y)
        return {'Beta': model.coef_[0], 'R-Squared': model.score(X, self.y)}

    def mutual_information(self):
        return mutual_info_regression(self.X, self.y)[0]

    def correlation(self):
        return np.corrcoef(self.X['factor_returns'], self.y)[0][1]

    def nonlinear_explained_variance(self, runs=25):
        """
        Fitting very small multilayer perceptron, gonna try and see how much it can explain.
        """
        X = self.data
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
        y = X['target']
        X = X[['factor_returns']]
        model = MLPRegressor(hidden_layer_sizes=6, tol=1e-15, max_iter=20, solver='lbfgs')
        results = []
        for run in range(runs):
            results.append(model.fit(X, y).score(X, y))
        return np.mean(results)

    def fit(self):
        """
        Fits all models and return results
        """
        linear_beta_results = self.linear_beta()
        result = {
            'Linear Beta': linear_beta_results['Beta'],
            'Linear R-Squared': linear_beta_results['R-Squared'],
            'Correlation': self.correlation(),
            'Mutual Information': self.mutual_information(),
            'Non-linear R-Squared': self.nonlinear_explained_variance()
        }
        return result

    @property
    def data(self):
        X = self.X.copy()
        X['target'] = self.y
        return X

