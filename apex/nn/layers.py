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

# AI
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
from torch.utils.data.sampler import *
from torchvision.transforms import Compose
import math


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()
        self._beta = torch.Tensor([1])
        self._beta_normalization = nn.ReLU()
        self._beta.requires_grad_()

    def forward(self, input):
        self._beta = self._beta.to(input.device)
        return (self.sigmoid(self._beta_normalization(self._beta) * input) * input)

class Swish1(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return (self.sigmoid(input) * input)

class NoisyLinear(nn.Module):
    """Applies a noisy linear transformation to the incoming data: :math:`y = (mu_w + sigma_w \cdot epsilon_w)x + mu_b + sigma_b \cdot epsilon_b`
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,
            defaults to 0.017 for independent and 0.4 for factorised. Default: None
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """
    def __init__(self, in_features, out_features, bias=True, factorised=True, std_init=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.cutoff = nn.Parameter(torch.Tensor([0.2]))
        self.cutoff.requires_grad = False

        self._t_eps_in = torch.Tensor(self.in_features)
        self._t_eps_out = torch.Tensor(self.out_features)
        self._t_bias_eps = torch.Tensor(self.out_features)
        self._t_weight_epsilon = torch.Tensor((self.out_features, self.in_features))
        self._t_beta_epsilon = torch.Tensor(self.out_features)
        self.n_decay_factor = 1
        self.decay_factor = torch.Tensor([5e-4])
        if not std_init:
            if self.factorised:
                self.std_init = 0.2/260/2
            else:
                self.std_init = 0.015/260/2
        else:
            self.std_init = std_init
        self.reset_parameters(bias)


    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(0.5 / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, x, device):
        x = x.to(device).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        # get device
        device = input.device
        decay = self.decay_factor.to(device).pow(self.n_decay_factor)
        self.n_decay_factor += 1
        if self.factorised:
            epsilon_in = self.scale_noise(self._t_eps_in, device)
            epsilon_out = self.scale_noise(self._t_eps_out, device)
            weight_epsilon = epsilon_out.ger(epsilon_in)
            bias_epsilon = self.scale_noise(self._t_bias_eps, device)
        else:
            weight_epsilon = self._t_weight_epsilon.to(device).normal_()
            bias_epsilon = self._t_beta_epsilon.to(device).normal_()
        A1 = self.weight_sigma.mul(weight_epsilon)
        B1 = self.bias_sigma.mul(bias_epsilon)
        A = self.weight_mu + A1
        B = self.bias_mu + B1
        decay = torch.max(decay, self.cutoff)
        return F.linear(input, A, B) * decay + input * (1 - decay)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
