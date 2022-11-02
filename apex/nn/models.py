from pathlib import Path, PurePath
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GroupShuffleSplit, train_test_split
import datetime
import logging
import numpy as np
import pandas as pd
import pickle
import sklearn as sk
import sys
import re
import scipy as sc

# AI
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
from torch.utils.data.sampler import *
from torchvision.transforms import Compose

# Local
from apex.nn.layers import Swish, Swish1, NoisyLinear


class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_features, n_hidden, n_gaussians, noise_level=None):
        super().__init__()
        self.z_h = nn.Sequential(
            nn.Linear(in_features, n_hidden * 4),
            Swish1(),
            nn.Linear(n_hidden * 4, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softsign(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_dof = nn.Sequential(
            nn.Linear(n_hidden, n_gaussians),
            nn.Softplus(),
        )

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        dof = self.z_dof(z_h)
        return pi, sigma, mu, dof


class Encoder(nn.Module):
    def __init__(self, in_features, n_encoded, n_hidden):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_features, n_hidden * 4),
            Swish1(),
            nn.Linear(n_hidden * 4, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            NoisyLinear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softsign(),
            nn.Linear(n_hidden, n_encoded),
        )

    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, n_encoded, n_hidden, out_features):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_encoded, n_hidden * 4),
            Swish1(),
            nn.Linear(n_hidden * 4, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            Swish1(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softsign(),
            nn.Linear(n_hidden, out_features),
        )

    def forward(self, x):
        return self.decode(x)


class ReducedNoiseEncoder(nn.Module):
    def __init__(self, in_features, n_encoded, n_hidden):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_features, n_hidden * 4),
            Swish(),
            nn.Linear(n_hidden * 4, n_hidden),
            NoisyLinear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            nn.Softsign(),
            nn.Linear(n_hidden, n_encoded),
        )

    def forward(self, x):
        return self.encode(x)

class ReducedNoiseDecoder(nn.Module):
    def __init__(self, n_encoded, n_hidden, out_features):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_encoded, n_hidden * 4),
            Swish1(),
            nn.Linear(n_hidden * 4, n_hidden),
            NoisyLinear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, n_hidden),
            Swish(),
            nn.Linear(n_hidden, out_features),
        )

    def forward(self, x):
        return self.decode(x)

class Autoencoder(nn.Module):
    """
    An autoencoder starts with input, and reduces it to n_hidden.
    """
    def __init__(self, in_features, n_encoded, n_hidden):
        super().__init__()
        self.encode = Encoder(in_features, n_encoded, n_hidden)
        self.decode = Decoder(n_encoded, n_hidden, in_features)

    def forward(self, x):
        x = self.encode(x)
        return x, self.decode(x)


class ReducedNoiseAutoencoder(nn.Module):
    """
    An autoencoder starts with input, and reduces it to n_hidden.
    """
    def __init__(self, in_features, n_encoded, n_hidden):
        super().__init__()
        self.encode = ReducedNoiseEncoder(in_features, n_encoded, n_hidden)
        self.decode = ReducedNoiseDecoder(n_encoded, n_hidden, in_features)

    def forward(self, x):
        x = self.encode(x)
        return x, self.decode(x)


class AutoencodedMixtureDensityNetwork(nn.Module):
    """
    A) Autoencoder
    The risk forecaster will first reduce all the information down to a small number of risk-sources.

    I'm going to have N=3 real risk-sources.

    B) MDN will be used to forecast the joint conditional distributions of target variables
    """
    def __init__(self, n_features=13, n_risk_factors=4, n_encoded=3, n_hidden_autoencoder=64, n_hidden_mdn=32, n_hidden_forecaster=8, n_output_forecaster=1):
        super().__init__()
        self.autoencoder = Autoencoder(n_features, n_encoded, n_hidden_autoencoder).double()
        self.mixed_density_network = MixtureDensityNetwork(n_encoded, n_hidden_mdn, n_risk_factors).double()
        self.forecaster = nn.Sequential(
            nn.Linear(n_risk_factors * 4 + n_encoded, n_hidden_forecaster * 2),
            Swish1(),
            nn.Linear(n_hidden_forecaster * 2, n_hidden_forecaster),
            Swish1(),
            nn.Linear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            NoisyLinear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            nn.Linear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            NoisyLinear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            nn.Linear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            NoisyLinear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            nn.Linear(n_hidden_forecaster, n_hidden_forecaster),
            Swish1(),
            nn.Linear(n_hidden_forecaster, n_hidden_forecaster),
            nn.Softsign(),
            nn.Linear(n_hidden_forecaster, n_output_forecaster),
        )

        self.n_risk_factors = n_risk_factors
        self.n_features = n_features
        self.n_hidden_mdn = n_hidden_mdn
        self.n_hidden_autoencoder = n_hidden_autoencoder
        self.n_encoded = n_encoded

    def forward(self, x):
        encoded_x, decoded_x = self.autoencoder(x)
        encoded_x = encoded_x.reshape(x.shape[0], 1, self.n_encoded)
        decoded_x = decoded_x.reshape(x.shape[0], 1, self.n_features)

        pi, sigma, mu, dof = self.mixed_density_network(encoded_x)

        # Now let's create a point estimate.
        forecaster_in = torch.cat([encoded_x, pi, mu, sigma, dof], -1)
        point_estimate = self.forecaster(forecaster_in)

        return pi, sigma, mu, dof, encoded_x, decoded_x, point_estimate[0]



oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians
def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    x = torch.div((y - mu), sigma)
    x = -torch.pow(x, 2) * 0.5
    return torch.div(torch.exp(x), sigma) * oneDivSqrtTwoPI

def students_t_distribution(y, mu, sigma, dof):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    x = torch.div((y - mu), sigma)
    result = sc.stats.t.pdf(x.detach().numpy(), dof.data.numpy())
    return torch.div(torch.from_numpy(result), sigma).double()

def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


def mdn_t_loss_fn(pi, sigma, mu, dof, y):
    result = students_t_distribution(y, mu, sigma, dof) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


def autoencoded_mdn_loss_fn(pi, sigma, mu, dof, encoded_x, decoded_x, forecasted_y, inputs, y):
    first_part = mdn_t_loss_fn(pi, sigma, mu, dof, y).double()
    second_part = torch.mean(torch.abs(inputs - decoded_x)).double()
    third_part = torch.mean(torch.abs(forecasted_y - y)).double()
    # Problem: if first_part < 0, which it can be, since log likelihood, if first_part < 0 then second_part will want to increase.
    # Solution:
    return first_part + torch.log(0.15+second_part) + torch.log(0.15+third_part)