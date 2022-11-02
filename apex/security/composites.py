#
# For future if needed
#
import sklearn.covariance as sk_cov
import attr
import pandas as pd
from apex.compute import ApexDaskClient
import distributed
import numpy as np


@attr.s
class EqualWeightedBasket:
    """
    Constituents is list of securities
    """
    constituents = attr.ib()
    name = attr.ib()
    constituent_returns = attr.ib(init=False)
    returns = attr.ib(init=False)

    def __attrs_post_init__(self):
        """
        Just initializing everything..
        """
        self.constituent_returns = pd.DataFrame(dict(zip(self.constituents, map(lambda x: x.unadjusted_data.returns, self.constituents))))
        self.returns = self.constituent_returns.mean(axis=1)

    def __repr__(self):
        return name

    @property
    def adjusted_data(self):
        result = pd.DataFrame({'returns': self.returns}).fillna(0)
        result['open'] = (1 + result['returns']).cumprod()
        result['close'] = result['open']
        result['high'] = result['open']
        result['low'] = result['open']
        result['volume'] = 0
        return result


    @property
    def unadjusted_data(self):
        result = pd.DataFrame({'returns': self.returns}).fillna(0)
        result['open'] = (1 + result['returns']).cumprod()
        result['close'] = result['open']
        result['high'] = result['open']
        result['low'] = result['open']
        result['volume'] = 0
        return result

def get_equal_weighted_basket(name, constituents):
    constituents = [ApexSecurity.from_ticker(x) for x in constituents]
    return EqualWeightedBasket(name=name, constituents=constituents)




@attr.s
class MarketCapWeightedBasket:
    """
    Constituents is list of securities
    """
    constituents = attr.ib()
    name = attr.ib()
    constituent_returns = attr.ib(init=False)
    constituent_shares_outstanding = attr.ib(init=False)
    returns = attr.ib(init=False)

    def __attrs_post_init__(self):
        """
        Just initializing everything..
        """
        self.constituent_returns = pd.DataFrame(dict(zip(self.constituents, map(lambda x: x.unadjusted_data.returns, self.constituents))))
        self.constituent_closes = pd.DataFrame(dict(zip(self.constituents, map(lambda x: x.unadjusted_data.close, self.constituents))))
        self.constituent_shares_outstanding = pd.DataFrame(dict(zip(self.constituents, map(lambda x: x.historical_shares_outstanding(), self.constituents))))
        self.market_caps = self.constituent_closes * self.constituent_shares_outstanding
        self.weights = self.market_caps.divide(self.market_caps.sum(axis=1), axis=0)
        self.returns = (self.constituent_returns * self.weights).sum(axis=1)

    def __repr__(self):
        return self.name

    @property
    def adjusted_data(self):
        result = pd.DataFrame({'returns': self.returns}).fillna(0)
        result['open'] = (1 + result['returns']).cumprod()
        result['close'] = result['open']
        result['high'] = result['open']
        result['low'] = result['open']
        result['volume'] = 0
        return result


    @property
    def unadjusted_data(self):
        result = pd.DataFrame({'returns': self.returns}).fillna(0)
        result['open'] = (1 + result['returns']).cumprod()
        result['close'] = result['open']
        result['high'] = result['open']
        result['low'] = result['open']
        result['volume'] = 0
        return result

def get_cap_weighted_basket(name, constituents):
    constituents = [ApexSecurity.from_ticker(x) for x in constituents]
    return MarketCapWeightedBasket(name=name, constituents=constituents)


@attr.s
class TradingFactorFactory:
    market_cap_weighted = attr.ib(default=False)
    __factors = attr.ib(init=False)
    factor_constituents = attr.ib(init=False)
    __factor_basket_futures = attr.ib(init=False)


    def __attrs_post_init__(self):
        """
        Initialization of object
        """
        EQUITY_BASE_FACTORS = {
            'Value': {'SGEPVBU Index'},
            'Quality': {'SGEPQBU Index'},
            'Profitability Factor': {'SGEPPBU Index'},
            'Momentum': {'SGEPMBU Index'},
            'Size': {'SGEPSBU Index'},
            'Low Volatility': {'SGEPLBU Index'},
            'Value + Quality': {'SGEPVQBU Index'},
            'Value Ex-Junk': {'SGEPVXBU Index'},
            'Quality Ex-Overvalued': {'SGEPQXBU Index'}
        }

        INTERNAL_FACTORS = pd.read_excel('/flatty/datasets/external/benchmarks/custom_mlp_indices.xlsx', 'Indices')
        self.__factors = {}
        factors = {}
        for factor in INTERNAL_FACTORS:
            factor_metadata = get_security_metadata(*INTERNAL_FACTORS[factor].dropna().tolist())
            factors[factor] = [x.parsekyable_des for x in factor_metadata.values()]

        for factor in EQUITY_BASE_FACTORS:
            factor_metadata = get_security_metadata(*EQUITY_BASE_FACTORS[factor])
            factors[factor] = [x.parsekyable_des for x in factor_metadata.values()]
        self.factor_constituents = factors
        self.__factor_basket_futures = {}

        dask_client = ApexDaskClient()
        for factor in factors:
            if self.market_cap_weighted:
                self.__factor_basket_futures[factor] = dask_client.submit(get_cap_weighted_basket, factor, factors[factor])
            else:
                self.__factor_basket_futures[factor] = dask_client.submit(get_equal_weighted_basket, factor, factors[factor])

    @property
    def available_factors(self):
        return list(self.factor_constituents.keys())

    def get(self, factor):
        if factor in self.available_factors:
            if factor in self.__factors:
                return self.__factors[factor]
            else:
                self.__factors[factor] = self.__factor_basket_futures[factor].result()
                return self.__factors[factor]

    def new(self, name, constituents):
        """
        Constituents is ticker list
        """
        assert name not in self.factors
        factor_metadata = get_security_metadata(*constituents)
        self.__factors[name] = EqualWeightedBasket(name=name,
                                                 constituents=[ApexSecurity.from_ticker(x.parsekyable_des) for x in factor_metadata.values()])
        return self.__factors[name]

    def economic_factors(self):
        """
        Most important indicators for the economy (forward looking) that we can have on a daily basis that affects MLPS
        - VIX
        - S&P 500 Momentum
        - Crude Oil
        - Natural Gas Prices
        - EIA Storage Numbers
        - Interest Rates
        - Crude Oil Volatility
        - AMZ/AMEI/SPMLP Volatility
        - Russell 2000 Volatility
        - Russell 2000 Momentum
        - Etc

        Now... since I want to forecast *changes* in volatility, i can't have any information that might tell the NN
        where we are in the sample in terms of time.

        So... need to diff everything so that there's no levels.
        """
        indices = [
            'SPX Index',
            'VIX Index',
            'RTY Index',
            'SPMLP Index',
            'NG1 Comdty',
            'CL1 Comdty',
            'USYC2Y10 Index', # 2-10s Spread
            'LPGSMBPE Index', #Ethane
            'LPGSMBPP Index', # Propane
            'LPGSMBNB Index', # Normal Butane Gasoline
            'LPGSMBIB Index', # Butane
            'LPGSMBNG Index', # Natural Gasoline
            'CRKS321C Index' # Crack Spread
        ]
        economic_data = pd.concat(
            {x.parsekyable_des: x.adjusted_data.close for x in
                [ApexSecurity.from_ticker(x) for x in indices]
            },
        axis=1)

        economic_data['S&P500 Realized Variance'] = economic_data['SPX Index'].pct_change().pow(2) * 100 * 260
        economic_data['S&P500 20D Volatility Premium'] = economic_data['VIX Index'] - economic_data['SPX Index'].pct_change().pow(2).rolling(20).mean().pow(0.5) * 100 * np.sqrt(260)
        economic_data['CL1 Realized Variance'] = economic_data['CL1 Comdty'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['NG1 Realized Variance'] = economic_data['NG1 Comdty'].pct_change().pow(2).rolling(3).mean() * 100 * 260

        economic_data['Butane Realized Variance'] = economic_data['LPGSMBIB Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['Ethane Realized Variance'] = economic_data['LPGSMBPE Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['Propane Realized Variance'] = economic_data['LPGSMBPP Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['Natural Gasoline Realized Variance'] = economic_data['LPGSMBNG Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['Normal Butane Realized Variance'] = economic_data['LPGSMBNB Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['Crack Spread Realized Variance'] = economic_data['CRKS321C Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260

        economic_data['Crack Spread RV 20D Diff'] = (economic_data['CRKS321C Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)
        economic_data['Ethane RV 20D Diff'] = (economic_data['LPGSMBPE Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)
        economic_data['Natural Gasoline RV 20D Diff'] = (economic_data['LPGSMBNG Index'].pct_change().pow(2).rolling(20).mean() * 100 * 260).diff(20)


        economic_data['SPMLP Realized Variance'] = economic_data['SPMLP Index'].pct_change().pow(2).rolling(3).mean() * 100 * 260
        economic_data['SPMLP RV vs S&P 500 RV'] = economic_data['SPMLP Realized Variance'] - economic_data['S&P500 Realized Variance']


        ### All series that need to be diffed
        SERIES_TO_DIFF = [
            'SPX Index',
            'VIX Index',
            'RTY Index',
            'SPMLP Index',
            'NG1 Comdty',
            'CL1 Comdty',
            'USYC2Y10 Index', # 2-10s Spread
            'LPGSMBPE Index', #Ethane
            'LPGSMBPP Index', # Propane
            'LPGSMBNB Index', # Normal Butane Gasoline
            'LPGSMBIB Index', # Butane
            'LPGSMBNG Index', # Natural Gasoline
            'CRKS321C Index' # Crack Spread
        ]

        economic_data[SERIES_TO_DIFF] = economic_data[SERIES_TO_DIFF].diff()

        economic_data = economic_data.rename(columns={
            'SPX Index': 'S&P 500 1d Difference',
            'VIX Index': 'VIX 1d Difference',
            'RTY Index': 'Russell 2000 1d Difference',
            'SPMLP Index': 'S&P MLP Index 1d Difference',
            'CL1 Comdty': 'Crude Oil 1st 1d Difference',
            'NG1 Comdty': 'Natural Gas 1st 1d Difference',
            'USYC2Y10 Index': 'US 2yr-10yr Spread 1d Difference',
            'LPGSMBPE Index': 'Ethane 1d Difference', #Ethane
            'LPGSMBPP Index': 'Propane 1d Difference', # Propane
            'LPGSMBNB Index': 'Normal Butane 1d Difference', # Normal Butane Gasoline
            'LPGSMBIB Index': 'Butane 1d Difference', # Butane
            'LPGSMBNG Index': 'Natural Gasoline 1d Difference', # Natural Gasoline
            'CRKS321C Index': 'Crack Spread 1d Difference' # Crack Spread
        })
        data = economic_data.dropna()
        return data


    def get_volatility_target_series(store, benchmark='AMZ Index', return_tensor=False):
        series = store.market_data(benchmark)
        series = series.close.pct_change().pow(2).rolling(3).mean() * 100.0 * 261
        series = pd.DataFrame({'volatility_target': series.fillna(0).diff(3).shift(-3).dropna()})
        if return_tensor:
            return torch.from_numpy(series.values)
        return series