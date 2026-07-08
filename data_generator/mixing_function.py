from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from .utils import leaky_tanh, sample_invertible_matrix
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np


class MixingFunction(ABC):
    """
    Base class for mixing functions.

    The mixing function is the function that maps from the latent space to the observation space.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

    def __call__(self, v: Tensor) -> Tensor:
        """
        Apply the mixing function to the latent variables.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        x: Tensor, shape (num_samples, observation_dim)
            Observed variables.
        """
        raise NotImplementedError()

    def save_coeffs(self, path: Path) -> None:
        """
        Save the coefficients of the mixing function to disk.

        Parameters
        ----------
        path: Path
            Path to save the coefficients to.
        """
        raise NotImplementedError()

    def unmixing_jacobian(self, v: Tensor) -> Tensor:
        """
        Compute the jacobian of the inverse mixing function using autograd and the inverse function theorem.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        unmixing_jacobian: Tensor, shape (num_samples, observation_dim, latent_dim)
            Jacobian of the inverse mixing function.

        References
        ----------
        https://en.wikipedia.org/wiki/Inverse_function_theorem
        https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/7
        """
        func = self.__call__
        inputs = v

        mixing_jacobian = torch.vmap(torch.func.jacrev(func))(inputs)
        unmixing_jacobian = torch.inverse(mixing_jacobian)

        return unmixing_jacobian


def subnet_fc(latent_dim, c_in, c_out, init_identity):
    subnet = nn.Sequential(
        nn.Linear(c_in, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, c_out),
    )
    if init_identity:
        subnet[-1].weight.data.fill_(0.0)
        subnet[-1].bias.data.fill_(0.0)
    return subnet


def construct_net(latent_dim, coupling_block, init_identity=True, n_flows=8):
    if coupling_block == "gin":
        block = Fm.GINCouplingBlock
    elif coupling_block == "glow":
        block = Fm.GLOWCouplingBlock
    else:
        raise ValueError(f"Unknown coupling_block {coupling_block!r}")
    nodes = [Ff.InputNode(latent_dim, name="input")]

    for k in range(n_flows):
        nodes.append(
            Ff.Node(
                nodes[-1],
                block,
                {
                    "subnet_constructor": lambda c_in, c_out: subnet_fc(
                        latent_dim, c_in, c_out, init_identity
                    ),
                    "clamp": 2.0,
                },
                name=f"coupling_{k}",
            )
        )
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.PermuteRandom,
                {"seed": np.random.randint(2**31)},
                name=f"permute_{k+1}",
            )
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return Ff.ReversibleGraphNet(nodes)


class VolumePreservingMixing(MixingFunction):
    def __init__(self, latent_dim, observation_dim, n_flows=8):
        super().__init__(latent_dim, observation_dim)
        self.generate_model = construct_net(
            latent_dim, "gin", init_identity=False, n_flows=n_flows
        )

    def __call__(self, v: Tensor) -> Tensor:
        x = self.generate_model(v)[0].detach()
        return x


class NonVolumePreservingMixing(MixingFunction):
    """Non-VP coupling-flow mixing function (GLOW-style affine coupling).

    Architectural twin of ``VolumePreservingMixing`` — same depth, same
    subnet, same permutations — except the per-block log|det J| is no
    longer constrained to zero. Used as a controlled VP-vs-nonVP ablation:
    swapping this in keeps every other knob fixed so any change in
    identifiability metrics is attributable to the loss of the VP property
    in the data-generating mixing g.
    """

    def __init__(self, latent_dim, observation_dim, n_flows=8):
        super().__init__(latent_dim, observation_dim)
        self.generate_model = construct_net(
            latent_dim, "glow", init_identity=False, n_flows=n_flows
        )

    def __call__(self, v: Tensor) -> Tensor:
        x = self.generate_model(v)[0].detach()
        return x


class LinearMixing(MixingFunction):
    """
    Linear mixing function. The coefficients are sampled from a uniform distribution.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        super().__init__(latent_dim, observation_dim)
        self.coeffs = torch.rand((latent_dim, observation_dim))

    def __call__(self, v: Tensor) -> Tensor:
        return torch.matmul(v, self.coeffs.to(v.device))

    def save_coeffs(self, path: Path) -> None:
        # save matrix coefficients
        torch.save(self.coeffs, path / "matrix.pt")
        matrix_np = self.coeffs.numpy()  # convert to Numpy array
        df = pd.DataFrame(matrix_np)  # convert to a dataframe
        df.to_csv(path / "matrix.csv", index=False)  # save as csv


class NonlinearMixing(MixingFunction):
    """
    Nonlinear mixing function.

    The function is composed of a number of invertible matrices and leaky-tanh nonlinearities. I.e. we
    apply a random neural network to the latent variables.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    n_nonlinearities: int
        Number of layers (i.e. invertible maps and nonlinearities) in the mixing function. Default: 1.
    """

    def __init__(
        self, latent_dim: int, observation_dim: int, n_nonlinearities: int = 1
    ) -> None:
        super().__init__(latent_dim, observation_dim)
        assert latent_dim == observation_dim
        self.coefs = torch.rand((latent_dim, observation_dim))
        self.n_nonlinearities = n_nonlinearities

        matrices = []
        for i in range(n_nonlinearities):
            matrices.append(sample_invertible_matrix(observation_dim))
        self.matrices = matrices

        nonlinearities = []
        for i in range(n_nonlinearities):
            nonlinearities.append(leaky_tanh)
        self.nonlinearities = nonlinearities

    def __call__(self, v: Tensor) -> Tensor:
        x = v
        for i in range(self.n_nonlinearities):
            mat = self.matrices[i].to(v.device)
            nonlinearity = self.nonlinearities[i]
            x = nonlinearity(torch.matmul(x, mat))
        return x

    def save_coeffs(self, path: Path) -> None:
        # save matrix coefficients
        for i in range(self.n_nonlinearities):
            torch.save(self.matrices[i], path / f"matrix_{i}.pt")
            matrix_np = self.matrices[i].numpy()  # convert to Numpy array
            df = pd.DataFrame(matrix_np)  # convert to a dataframe
            df.to_csv(path / f"matrix_{i}.csv", index=False)  # save as csv

        # save matrix determinants in one csv
        matrix_determinants = []
        for i in range(self.n_nonlinearities):
            matrix_determinants.append(torch.det(self.matrices[i]))
        matrix_determinants_np = torch.stack(matrix_determinants).numpy()
        df = pd.DataFrame(matrix_determinants_np)
        df.to_csv(path / "matrix_determinants.csv")
