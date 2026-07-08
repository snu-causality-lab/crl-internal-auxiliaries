from __future__ import annotations
from model import mcc
from abc import ABC
from itertools import product
from typing import Optional, List, Union
from torch.distributions import MultivariateNormal
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split 
import csv
from .net import *
from .dci import *
from model.result_io import save_mcc_result
import matplotlib.pyplot as plt
from PIL import Image

class GIN(pl.LightningModule, ABC):
    """
    GIN (General Incompressible-flow Network) baseline model.
    It implements the training loop and the evaluation metrics for the GIN baseline.

    Attributes
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.
    encoder : ReversibleGraphNet
        The GIN encoder (volume-preserving normalizing flow).

    Methods
    -------
    training_step(batch, batch_idx) -> Tensor
        Training step.
    validation_step(batch, batch_idx) -> dict[str, Tensor]
        Validation step.
    validation_epoch_end(outputs) -> None
        Computes validation metrics across all validation data.
    test_step(batch, batch_idx) -> dict[str, Tensor]
        Test step.
    test_epoch_end(outputs) -> None
        Computes test metrics across all test data.
    configure_optimizers() -> dict | torch.optim.Optimizer
        Configures the optimizer and learning rate scheduler.
    forward(x) -> torch.Tensor
        Computes the latent variables from the observed data.
    """

    def __init__(
        self,
        latent_dim: int,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        dgp_name: str = "a",
        observed_idx = [],
        selected_idx = [4],
        seed: int = 42,
        result_dir = "./result_GIN",
        mode: str = 'synthetic'
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        if mode == "synthetic":
            self.encoder = construct_net(latent_dim, 'gin')  # needs to be set in subclasses
        else:
            self.encoder = construct_net_image('gin')
        self.seed = seed
        self.selected_idx = selected_idx
        self.mu = nn.Linear(len(selected_idx), latent_dim)
        self.log_sig = nn.Linear(len(selected_idx), latent_dim)
        self.result_dir = result_dir
        self.dgp_name = dgp_name
        self.observed_idx = observed_idx

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:

        x, v, z_o_s, z_o_ns = batch
         
        z, logdet_J = self.encoder(x)  # latent space variable

        m = self.mu(z_o_s).to(x.device)
        ls = self.log_sig(z_o_s).to(x.device).clamp(0.01,10000)
        # negative log-likelihood for gaussian in latent space
        loss = torch.mean(0.5*(z-m)**2 * torch.exp(-2*ls).clamp(0.01,10000) + ls, 1) + 0.5*np.log(2*np.pi)
        loss -= logdet_J / self.latent_dim

        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> dict[str, Tensor]:
        pass

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        pass

    def test_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> Union[None, dict[str, Tensor]]:
        x, v, z_o_s, z_o_ns = batch
        z, logdet_J = self.encoder(x)
        return {"logdet_J": logdet_J, "z": z, "v": v}

    def test_epoch_end(self, outputs: List[dict]) -> None:
        logdet_J = torch.cat([o["logdet_J"] for o in outputs])
        z_g = torch.cat([o["z"] for o in outputs])
        v_g = torch.cat([o["v"] for o in outputs])
        z = z_g.cpu().detach().numpy()
        v = v_g.cpu().detach().numpy()
        logdet_J = logdet_J.mean()
        all_idx = list(range(v.shape[1]))
        idx = list(set(all_idx)-set(self.selected_idx)-set(self.observed_idx))

        score, table, match = mcc.mean_corr_coef(
            v[:, idx], z
        )
        print("det : ", logdet_J)
        print("score :", score)
        print("idx :", match)
        print(table)
        result_dgp_dir = os.path.join(self.result_dir, self.dgp_name)
        matched_mcc = table[:, match[1]]
        save_mcc_result(result_dgp_dir, self.seed, matched_mcc, score, match)
        #train_indices, test_indices = train_test_split(range(z.shape[0]), test_size=0.2, random_state=42)

        dci_metric = {}
        importance_matrix = matched_mcc.T
        # factor_types = ["continuous"]*z[train_indices][:,match[1]].shape[1]

        # importance_matrix, train_err, test_err = compute_importance_gbt(
        #     v[train_indices][:, idx].T, z[train_indices][:,match[1]].T, v[test_indices][:, idx].T, z[test_indices][:, match[1]].T, factor_types
        # )

        # assert importance_matrix.shape[0] == v[train_indices][:, idx].T.shape[0]
        # assert importance_matrix.shape[1] == z[train_indices][:, match[1]].T.shape[0]
        # dci_metric["informativeness_train"] = train_err
        # dci_metric["informativeness_test"] = test_err
        dci_metric["disentanglement"] = disentanglement(importance_matrix)
        dci_metric["completeness"] = completeness(importance_matrix)

        print(dci_metric)
        with open(os.path.join(self.result_dir, f"{self.dgp_name}/{self.seed}_dci.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dci_metric.keys())
            writer.writeheader()
            writer.writerow(dci_metric)
        # if  z.shape[1] > 5:
        #     z_sample = z_g[1,:]
        #     for i, dim_idx in enumerate(match[1]):
        #         mini = z_g[:,dim_idx].min().item()
        #         maxi = z_g[:,dim_idx].max().item()
        #         self.latent_traversal_and_save(z_sample, dim_idx, traversal_range=(mini,maxi), save_dir=f"{self.result_dir}/{self.dgp_name}/", seed=self.seed)
    def latent_traversal_and_save(self, v_sample, dim_idx, traversal_range=(-3, 3), steps=10, save_dir=None, seed=0):
            """
            Performs latent traversal and saves the reconstructed outputs as images.

            Args:
                v_sample (torch.Tensor): A single latent vector to manipulate.
                dim_idx (int): Index of the latent dimension to traverse.
                traversal_range (tuple): Range of traversal for the dimension.
                steps (int): Number of steps in the traversal.
                save_dir (str): Directory to save the traversal images.

            Returns:
                None
            """
            os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
            traversal_values = torch.linspace(traversal_range[0], traversal_range[1], steps)
            self.encoder.to('cpu')
            self.encoder.eval()
            for step, value in enumerate(traversal_values):
                v_sample_mod = v_sample.clone()  # Clone to avoid modifying the original
                v_sample_mod[dim_idx] = value  # Manipulate specific dimension
                v_sample_mod = v_sample_mod.cpu().detach()
                x_hat, _ = self.encoder(v_sample_mod.unsqueeze(0), rev=True)  # Decode the latent vector

                x_hat = x_hat.squeeze().cpu().detach().clamp(0, 1).permute(1, 2, 0).numpy()


                image = Image.fromarray((x_hat * 255).astype(np.uint8), mode='RGBA')  # Normalize to [0, 255]
                image_path = os.path.join(save_dir, f"dim{dim_idx}_step{step}_z{value:.2f}_{seed}.png")

                image.save(image_path)


            print(f"Saved latent traversal images in: {save_dir}")
        
    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        config_dict = {}
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        config_dict["optimizer"] = optimizer

        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
            config_dict["lr_scheduler"] = lr_scheduler_config
        elif self.lr_scheduler is None:
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        return config_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_hat, _ = self.encoder(x)
        return v_hat

