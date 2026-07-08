from __future__ import annotations
from model import mcc
from abc import ABC
from itertools import product
from typing import Optional, List, Union
from torch.distributions import MultivariateNormal, Independent, Normal
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
# from torch.optim import Optimizer
import torch.nn as nn
import csv
from sklearn.model_selection import train_test_split 
import os
# import matplotlib.pyplot as plt
# from PIL import Image
from torch.cuda.amp import autocast


from .encoder import NonlinearCauCAEncoder
from .dci import *
from .image_compress import *
from .result_io import save_mcc_result
#from .utils import vae_flow_loss

class CauCAModel(pl.LightningModule, ABC):
    """
    Base class for Causal Component Analysis (CauCA) models. It implements the
    training loop and the evaluation metrics.

    Attributes
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    adjacency_matrix : np.ndarray, shape (num_nodes, num_nodes)
        Adjacency matrix of the causal graph assumed by the model. This is not necessarily
        the true adjacency matrix of the data generating process (see below).
    adjacency_matrix_gt : np.ndarray, shape (num_nodes, num_nodes)
        Ground truth adjacency matrix of the causal graph. This is the adjacency matrix
        of the data generating process.
    adjacency_misspecified : bool
        Whether the adjacency matrix is misspecified. If True, the model assumes a wrong
        adjacency matrix.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.
    encoder : CauCAEncoder
        The CauCA encoder. Needs to be set in subclasses.

    Methods
    -------
    training_step(batch, batch_idx) -> Tensor
        Training step.
    validation_step(batch, batch_idx) -> dict[str, Tensor]
        Validation step: basically passes data to validation_epoch_end.
    validation_epoch_end(outputs) -> None
        Computes validation metrics across all validation data.
    test_step(batch, batch_idx) -> dict[str, Tensor]
        Test step: basically passes data to test_epoch_end.
    test_epoch_end(outputs) -> None
        Computes test metrics across all test data.
    configure_optimizers() -> dict | torch.optim.Optimizer
        Configures the optimizer and learning rate scheduler.
    forward(x) -> torch.Tensor
        Computes the latent variables from the observed data.
    on_before_optimizer_step(optimizer, optimizer_idx) -> None
        Callback that is called before each optimizer step. It ensures that some gradients
        are set to zero to fix some causal mechanisms. See documentation of ParamMultiEnvCausalDistribution
        for more details.
    set_adjacency(adjacency_matrix, adjacency_misspecified) -> np.ndarray
        Sets the adjacency matrix and possibly changes it if it is misspecified.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
        selected_idx: list = [4],
        observed_idx: list = [4],
        dgp_name: str = "a",
        seed: int = 42,
        result_dir: str = "./result",
        transfer: bool = True
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.adjacency_matrix = self.set_adjacency(
            adjacency_matrix, adjacency_misspecified
        )
        self.adjacency_matrix_gt = adjacency_matrix
        self.adjacency_misspecified = adjacency_misspecified
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.encoder = None  # needs to be set in subclasses
        self.save_hyperparameters()
        self.selected_idx = selected_idx
        self.observed_idx = observed_idx
        self.dgp_name = dgp_name
        self.seed = seed
        self.result_dir = result_dir
        self.transfer = transfer

    @staticmethod
    def set_adjacency(
        adjacency_matrix: np.ndarray, adjacency_misspecified: bool
    ) -> np.ndarray:
        if not adjacency_misspecified:
            return adjacency_matrix

        if adjacency_matrix.shape[0] == 2 and np.sum(adjacency_matrix) == 0:
            # for 2 variables, if adjacency matrix is [[0, 0], [0, 0]], then
            # replace with [[0, 1], [0, 0]]

            adjacency_matrix_out = np.zeros_like(adjacency_matrix)
            adjacency_matrix_out[0, 1] = 1
            return adjacency_matrix_out
        elif adjacency_matrix.shape[0] > 2:
            raise ValueError(
                "Adjacency misspecification not supported for empty adjacency matrix for >2 variables"
            )
        else:
            return adjacency_matrix.T

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        #recloss = nn.MSELoss(reduction='none')
        x, v, z_o_s, z_o_ns = batch
        z, logdet_J = self.encoder(x)

        # p(z_o_s)
        if z_o_s.shape[1] == 1:
            sigma_s = torch.cov(z_o_s.squeeze(-1))
            sigma_s = sigma_s.reshape(1,1)
        else:
            sigma_s = torch.cov(z_o_s.T)

        dist_s = MultivariateNormal(z_o_s, sigma_s)
        loss = -dist_s.log_prob(z[:, self.selected_idx]).unsqueeze(-1)


        #p(z_o_s^- | z_o_s)
        #m = self.mu(z_o_s).to(x.device)
        m = torch.zeros(self.n_blocks).to(x.device)
        ls = self.log_sig(z_o_s).reshape(-1, self.n_blocks).exp()
        all_idx = list(range(z.shape[1]))
        idx = sorted(list(set(all_idx)-set(self.selected_idx)))

        # dstn = MultivariateNormal(m, torch.diag_embed(ls))
        # loss += -dstn.log_prob(z[:,idx]).unsqueeze(-1)
        dstn = Independent(Normal(m, ls), 1)
        loss = loss + (-dstn.log_prob(z[:, idx].clone()).unsqueeze(-1))

        # # graphical constraint
        # if len(self.observed_idx) != 0:
        #     if z_o_ns.shape[1] == 1:
        #         sigma_ns = torch.cov(z_o_ns.squeeze(-1))
        #         sigma_ns = sigma_s.reshape(1,1)
        #     else:
        #         sigma_ns = torch.cov(z_o_ns)
        #     z_o_ns_hat = self.encoder.q0(z, self.observed_idx)
            #loss += -MultivariateNormal(z_o_ns.mean(axis=0), sigma_ns).log_prob(z_o_ns_hat).unsqueeze(-1) * 0.1
        if len(self.observed_idx) != 0:
            z_o_ns_hat = self.encoder.q0(z, self.observed_idx)
            loss = loss + (
                -MultivariateNormal(
                    z_o_ns,
                    torch.diag_embed(torch.ones_like(z_o_ns[0, :]))
                ).log_prob(z_o_ns_hat).unsqueeze(-1)
            )

        loss = loss - (logdet_J.unsqueeze(-1) / self.latent_dim)
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
        # log_prob, res = self.encoder.multi_env_log_prob(x, e, int_target, self.observed_idx)
        # mu, log_var = self.vae.encoder(x)
        # compressed = self.vae.reparameterize(mu, log_var)
        # compressed = self.image_encoder(x)
        z, logdet_J = self.encoder.flows(x)  # latent space variable
        # restored = self.vae.decoder(compressed)
        return {"logdet_J": logdet_J, "z": z, "v": v}

    def test_epoch_end(self, outputs: List[dict]) -> None:
        logdet_J = torch.cat([o["logdet_J"] for o in outputs])
        z_g = torch.cat([o["z"] for o in outputs])
        v_g = torch.cat([o["v"] for o in outputs])
        #imgs = torch.cat([o["imgs"] for o in outputs])
        z = z_g.cpu().detach().numpy()
        v = v_g.cpu().detach().numpy()
        logdet_J = logdet_J.mean()
        all_idx = list(range(v.shape[1]))
        all_idx2 = list(range(z.shape[1]))
        
        idx = sorted(set(all_idx) - set(self.selected_idx) - set(self.observed_idx))
        idx2 = sorted(set(all_idx2) - set(self.selected_idx) - set(self.observed_idx))
        score, table, match = mcc.mean_corr_coef(
            v[:, idx], z[:, idx2]
        )
        print("det : ", logdet_J)
        print("score :", score)
        print("idx :", match)
        print(table[:,match[1]])

        matched_mcc = table[:, match[1]]
        importance_matrix = matched_mcc.T
        dci_metric = {}
        dci_metric["disentanglement"] = disentanglement(importance_matrix)
        dci_metric["completeness"] = completeness(importance_matrix)
        print(dci_metric)

        result_dgp_dir = os.path.join(self.result_dir, self.dgp_name)
        os.makedirs(result_dgp_dir, exist_ok=True)

        save_mcc_result(result_dgp_dir, self.seed, matched_mcc, score, match)
        with open(os.path.join(result_dgp_dir, f"{self.seed}_dci.csv"), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dci_metric.keys())
            writer.writeheader()
            writer.writerow(dci_metric)

        # ---- Optional nonlinear-metric validation for appendix experiments ----
        # Triggered by env var COMPUTE_NONLINEAR_METRICS=1 to avoid
        # impacting baseline runs. Saves a separate CSV.
        if os.environ.get("COMPUTE_NONLINEAR_METRICS", "0") == "1":
            nonlinear_metrics = {}
            z_matched = z[:, idx2][:, match[1]]
            v_selected = v[:, idx]
            train_idx, test_idx = train_test_split(
                range(z.shape[0]), test_size=0.2, random_state=42
            )
            # GBR-based DCI (Eastwood-Williams 2018 standard variant)
            try:
                factor_types = ["continuous"] * v_selected.shape[1]
                imp_gbr, _, _ = compute_importance_gbt(
                    z_matched[train_idx].T, v_selected[train_idx].T,
                    z_matched[test_idx].T, v_selected[test_idx].T,
                    factor_types,
                )
                nonlinear_metrics["disentanglement_gbr"] = disentanglement(imp_gbr)
                nonlinear_metrics["completeness_gbr"] = completeness(imp_gbr)
            except Exception as e:
                print(f"[nonlinear-metric] GBR-DCI failed: {e}")
                nonlinear_metrics["disentanglement_gbr"] = float("nan")
                nonlinear_metrics["completeness_gbr"] = float("nan")
            # R² kernel-ridge: fit z_matched -> v_selected per factor, report mean R²
            try:
                from sklearn.kernel_ridge import KernelRidge
                from sklearn.metrics import r2_score
                r2_list = []
                for i in range(v_selected.shape[1]):
                    kr = KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0)
                    kr.fit(z_matched[train_idx], v_selected[train_idx, i])
                    pred = kr.predict(z_matched[test_idx])
                    r2_list.append(r2_score(v_selected[test_idx, i], pred))
                nonlinear_metrics["r2_kr_mean"] = float(np.mean(r2_list))
                nonlinear_metrics["r2_kr_per_factor"] = str([round(r, 3) for r in r2_list])
            except Exception as e:
                print(f"[nonlinear-metric] R² kernel-ridge failed: {e}")
                nonlinear_metrics["r2_kr_mean"] = float("nan")
                nonlinear_metrics["r2_kr_per_factor"] = "nan"
            print("[nonlinear-metric]", nonlinear_metrics)
            with open(
                os.path.join(result_dgp_dir, f"{self.seed}_nonlinear_metrics.csv"),
                "w", newline="",
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=nonlinear_metrics.keys())
                writer.writeheader()
                writer.writerow(nonlinear_metrics)

        # if self.mode != 'synthetic':
        #             imgs = imgs.detach().cpu()
        #             imgs = imgs.clamp(0, 1)
        #             for i in range(10):
        #                 image_array = imgs[i].permute(1, 2, 0).numpy()
        #                 image_array = (image_array * 255).astype(np.uint8)

        #                 # Create a PIL Image and save it
        #                 image = Image.fromarray(image_array, mode='RGBA')
        #                 image.save(os.path.join(self.result_dir, f'test{i}.png'))                
        # if self.mode != "synthetic":
        #     n_step = 9
        #     max_std = 2
        #     z_sample = z_g[:1]
        #     selected =  v_g[:1, self.selected_idx]
        #     observed = v_g[:1, self.observed_idx]
        #     all_idx2 = list(range(z.shape[1]))
        #     idx2 = list(set(all_idx2)-set(self.selected_idx))
        #     for i, dim in enumerate(idx):
        #         style = torch.zeros(n_step, 64*64)
        #         style[:,self.selected_idx] = selected
        #         style[:,dim] = torch.linspace(-max_std, max_std, n_step)
        #         style = style.to(z_g.device)
        #         ls = self.log_sig(selected).exp()

        #         style[:,idx2] = style[:,idx2]*ls.unsqueeze(1) 
        #         # style[:, self.observed_idx] = self.encoder.q0(style, self.observed_idx)
        #         style[:,self.observed_idx] = observed
        #         x_hat, _ = self.encoder(style, rev=True)
        #         x_hat = x_hat.squeeze().cpu().detach().numpy()  # batch, channel, height, width

        #         for j, image_array in enumerate(x_hat):
        #             image = Image.fromarray((image_array * 255).astype(np.uint8))  # channel-first to channel-last
        #             image_path = f"{self.result_dir}/{self.dgp_name}/test_{i}_{j}_{z_sample[:, self.selected_idx]}.png"
        #             image.save(image_path)

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
        elif self.lr_scheduler == "reduce":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif self.lr_scheduler is None:
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        return config_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_hat = self.encoder(x)
        return v_hat

    # def latent_traversal_and_save(self, v_sample, dim_idx, traversal_range=(-3, 3), steps=10, save_dir=None, seed=0):
    #     """
    #     Performs latent traversal and saves the reconstructed outputs as images.

    #     Args:
    #         v_sample (torch.Tensor): A single latent vector to manipulate.
    #         dim_idx (int): Index of the latent dimension to traverse.
    #         traversal_range (tuple): Range of traversal for the dimension.
    #         steps (int): Number of steps in the traversal.
    #         save_dir (str): Directory to save the traversal images.

    #     Returns:
    #         None
    #     """
    #     os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    #     traversal_values = torch.linspace(traversal_range[0], traversal_range[1], steps)
    #     self.encoder.to('cpu')
    #     self.encoder.eval()
    #     for step, value in enumerate(traversal_values):
    #         v_sample_mod = v_sample.clone()  # Clone to avoid modifying the original
    #         v_sample_mod[:, dim_idx] = value  # Manipulate specific dimension
    #         v_sample_mod = v_sample_mod.cpu().detach()
    #         x_hat, _ = self.encoder(v_sample_mod, rev=True)  # Decode the latent vector

    #         x_hat = x_hat.squeeze().cpu().detach().numpy()


    #         image = Image.fromarray((x_hat * 255).astype(np.uint8), mode='L')  # Normalize to [0, 255]
    #         image_path = os.path.join(save_dir, f"dim{dim_idx}_step{step}_z{value:.2f}_{seed}.png")

    #         image.save(image_path)


    #     print(f"Saved latent traversal images in: {save_dir}")



class NonlinearCauCAModel(CauCAModel):
    """
    CauCA model with nonlinear unmixing function.

    Additional attributes
    ---------------------
    k_flows : int
        Number of flows to use in the nonlinear unmixing function. Default: 1.
    net_hidden_dim : int
        Hidden dimension of the neural network used in the nonlinear unmixing function. Default: 128.
    net_hidden_layers : int
        Number of hidden layers of the neural network used in the nonlinear unmixing function. Default: 3.
    fix_mechanisms : bool
        Some mechanisms can be fixed to a simple gaussian distribution without loss of generality.
        This has only an effect for the parametric base distribution. If True, these mechanisms are fixed.
        Default: True.
    fix_all_intervention_targets : bool
        When fixable mechanisms are fixed, this parameter determines whether all intervention targets
        are fixed (option 1) or all intervention targets which are non-root nodes together with all
        non-intervened root nodes (option 2). See documentation of ParamMultiEnvCausalDistribution
        for more details. Default: False.
    nonparametric_base_distr : bool
        Whether to use a nonparametric base distribution for the flows. If false, a parametric linear
        gaussian causal base distribution is used. Default: False.
    K_cbn : int
        Number of flows to use in the nonlinear nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn : int
        Hidden dimension of the neural network used in the nonlinear nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn : int
        Number of hidden layers of the neural network used in the nonlinear nonparametric base distribution. Default: 3.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Tensor,
        obs_dim=None,  # 1
        n_blocks=None,  # 4
        block_dims=None,  # [1,1,1,1]
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
        k_flows: int = 1,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
        fix_mechanisms: bool = True,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3,
        selected_idx: list = [4],
        observed_idx: list = [4],
        dgp_name: str = "a",
        seed: int = 42,
        result_dir: str = "./result",
        mode: str = "synthetic"
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            adjacency_misspecified=adjacency_misspecified,
            selected_idx= selected_idx,
            observed_idx= observed_idx,
            dgp_name=dgp_name,
            seed=seed,
            result_dir = result_dir
        )
        self.encoder = NonlinearCauCAEncoder(
            latent_dim,
            self.adjacency_matrix,  # this is the misspecified adjacency matrix if adjacency_misspecified=True
            K=k_flows,
            intervention_targets_per_env=intervention_targets_per_env,
            net_hidden_dim=net_hidden_dim,
            net_hidden_layers=net_hidden_layers,
            fix_mechanisms=fix_mechanisms,
            fix_all_intervention_targets=fix_all_intervention_targets,
            nonparametric_base_distr=nonparametric_base_distr,
            K_cbn=K_cbn,
            net_hidden_dim_cbn=net_hidden_dim_cbn,
            net_hidden_layers_cbn=net_hidden_layers_cbn,
            mode= mode
        )
        self.mode = mode
        self.n_blocks = n_blocks
        self.block_dims = block_dims
        self.obs_dim = obs_dim
        self.selected_dim = len(selected_idx)

        # if mode == "flow":
        #     self.vae = VAE()
        # self.image_encoder = ImageEncoder()
        # self.image_decoder = ImageDecoder()
        # self.log_sig_s = nn.Linear(self.selected_dim, self.selected_dim*self.selected_dim)

        #self.log_sig_ns = nn.Linear(self.selected_dim, self.obs_dim)

        #self.mu = nn.Linear(self.selected_dim, n_blocks)
        self.log_sig = nn.Linear(self.selected_dim, n_blocks)
        # hidden_dim = 256
        # self.log_sig = nn.Sequential(
        #         nn.Linear(self.selected_dim, hidden_dim),  
        #         nn.ReLU(),                                
        #         nn.Linear(hidden_dim, hidden_dim),        
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, n_blocks)          
        #     )
        #self.mu = {}
        #self.log_sig = {}

        #for i, block_dim in enumerate(block_dims):
            #self.mu[i] = nn.Linear(self.selected_dim, block_dim)
            #self.log_sig[i] = nn.Linear(self.selected_dim, block_dim * block_dim)

        self.save_hyperparameters()
