from typing import Optional

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from torch import abs, det, log, Tensor

from .normalizing_flow.distribution import GraphPrior
from .normalizing_flow import ParamMultiEnvCausalDistribution
from .normalizing_flow.distribution import NaiveMultiEnvCausalDistribution
from .normalizing_flow.nonparametric_distribution import (
    NonparamMultiEnvCausalDistribution,
)
from .normalizing_flow.utils import *


class CauCAEncoder(nn.Module):
    """
    CauCA encoder for multi-environment data.

    The encoder maps from the observed data x to the latent space v_hat. The latent space is
    assumed to have causal structure. The encoder is trained to maximize the likelihood of
    the data under the causal model. x and v_hat are assumed to have the same dimension.

    The encoder has two main components:
        1. A causal base distribution q0 over the latent space. This encodes the latent
        causal structure.
        2. An unmixing function mapping from the observations to the latent space.

    Attributes
    ----------
    latent_dim: int
        Dimension of the latent and observed variables.
    adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        Adjacency matrix of the latent causal graph.
    intervention_targets_per_env: Tensor, shape (no_envs, latent_dim)
        Which variables are intervened on in each environment.
    fix_mechanisms: bool
        Whether to fix some fixable mechanisms in the causal model. (See documentation of the
        ParamMultiEnvCausalDistribution for details.) Default: False.
    fix_all_intervention_targets: bool
        Whether to fix all intervention targets in the causal model. (See documentation of the
        ParamMultiEnvCausalDistribution for details.) Default: False.
    nonparametric_base_distr: bool
        Whether to use a nonparametric base distribution. If False, a parametric base distribution
        assuming linear causal mechanisms is used. Default: False.
    flows: Optional[list[nf.flows.Flow]]
        List of normalizing flows to use for the unmixing function. Default: None.
    q0: Optional[nf.distributions.BaseDistribution]
        Base distribution over the latent space. Default: None.
    K_cbn: int
        Number of normalizing flows to use for the nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn: int
        Hidden dimension of the neural network used in the nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn: int
        Number of hidden layers in the neural network used in the nonparametric base distribution. Default: 3.

    Methods
    -------
    multi_env_log_prob(x, e, intervention_targets) -> Tensor
        Computes log probability of x in environment e.
    forward(x) -> Tensor
        Maps from the observed data x to the latent space v_hat.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Optional[Tensor] = None,
        fix_mechanisms: bool = False,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        flows: Optional[list[nf.flows.Flow]] = None,
        q0: Optional[nf.distributions.BaseDistribution] = None,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.adjacency_matrix = adjacency_matrix
        self.intervention_targets_per_env = intervention_targets_per_env
        self.fix_mechanisms = fix_mechanisms
        self.fix_all_intervention_targets = fix_all_intervention_targets
        self.nonparametric_base_distr = nonparametric_base_distr
        self.K_cbn = K_cbn
        self.net_hidden_dim_cbn = net_hidden_dim_cbn
        self.net_hidden_layers_cbn = net_hidden_layers_cbn


        self.flows = flows
        if q0 is None:
            if self.nonparametric_base_distr:
                self.q0 = NonparamMultiEnvCausalDistribution(
                    adjacency_matrix=adjacency_matrix,
                    K=K_cbn,
                    net_hidden_dim=net_hidden_dim_cbn,
                    net_hidden_layers=net_hidden_layers_cbn,
                )
            else:
                assert (
                    intervention_targets_per_env is not None
                ), "intervention_targets_per_env must be provided for parametric base distribution"

                self.q0 = GraphPrior(self.adjacency_matrix, len(adjacency_matrix))

    def multi_env_log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class NonlinearCauCAEncoder(CauCAEncoder):
    """
    Nonlinear CauCA encoder for multi-environment data.

    Here the unmixing function is a normalizing flow.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent and observed variables.
    adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        Adjacency matrix of the latent causal graph.
    K: int
        Number of normalizing flows to use for the unmixing function. Default: 1.
    intervention_targets_per_env: Tensor, shape (no_envs, latent_dim)
        Which variables are intervened on in each environment.
    net_hidden_dim: int
        Hidden dimension of the neural network used in the normalizing flows. Default: 128.
    net_hidden_layers: int
        Number of hidden layers in the neural network used in the normalizing flows. Default: 3.
    q0: Optional[nf.distributions.BaseDistribution]
        Base distribution over the latent space. Default: None.
    K_cbn: int
        Number of normalizing flows to use for the nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn: int
        Hidden dimension of the neural network used in the nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn: int
        Number of hidden layers in the neural network used in the nonparametric base distribution. Default: 3.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        K: int = 8,
        intervention_targets_per_env: Optional[Tensor] = None,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
        fix_mechanisms: bool = True,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        q0: Optional[nf.distributions.BaseDistribution] = None,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3,
        mode: str = "synthetic"
    ) -> None:
        self.K = K
        self.intervention_targets_per_env = intervention_targets_per_env
        self.net_hidden_dim = net_hidden_dim
        self.net_hidden_layers = net_hidden_layers
        # flows = construct_net_dsp(coupling_block='gin', init_identity=False)
        # flows = make_spline_flows(K, latent_dim, net_hidden_dim, net_hidden_layers)
        if mode == "synthetic":
            flows = construct_net(latent_dim, coupling_block='gin', init_identity=False, n_flows = K)
        else:
            flows = construct_net_image(coupling_block='gin')

        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            intervention_targets_per_env=intervention_targets_per_env,
            fix_mechanisms=fix_mechanisms,
            fix_all_intervention_targets=fix_all_intervention_targets,
            nonparametric_base_distr=nonparametric_base_distr,
            flows=flows,
            q0=q0,
            K_cbn=K_cbn,
            net_hidden_dim_cbn=net_hidden_dim_cbn,
            net_hidden_layers_cbn=net_hidden_layers_cbn
        )

    def forward(self, x: Tensor, rev=False) -> Tensor:
        return self.flows(x, rev=rev)
