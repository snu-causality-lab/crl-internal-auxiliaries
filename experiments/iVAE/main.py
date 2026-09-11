import argparse
import os
from pathlib import Path
import pytorch_lightning as pl

from config_iVAE import DGP
from data_generator import MultiEnvDataModule, make_multi_env_dgp
from .wrappers import *
from data_flows.data_module import ImageDataModule

import torch
import numpy as np
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiment for Causal Component Analysis (CauCA)."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator to use for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--checkpoint-root-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory.",
    )
    parser.add_argument(
        "--noise-shift-type",
        type=str,
        default="mean",
        choices=["mean", "std"],
        help="Property of noise distribution that is shifted between environments.",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=10,
        help="Check validation loss every n epochs.",
    )
    parser.add_argument(
        "--dgp",
        type=str,
        default="b",
        help="Data generation process to use.",
    )
    parser.add_argument(
        "--k-flows",
        type=int,
        default=8,
        help="Number of flows to use in nonlinear ICA model.",
    )
    parser.add_argument(
        "--k-flows-cbn",
        type=int,
        default=3,
        help="Number of flows to use in nonlinear latent CBN model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nonlinear",
        help="Type of encoder to use.",
        choices=["linear", "nonlinear", "naive"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mixing",
        type=str,
        default="volumepreserving",
        help="Type of mixing function to use.",
        choices=["linear", "nonlinear", "volumepreserving"],
    )
    parser.add_argument(
        "--scm",
        type=str,
        default="linear",
        help="Type of SCM to use.",
        choices=["linear", "location-scale"],
    )
    parser.add_argument(
        "--n-nonlinearities",
        type=int,
        default=2,
        help="Number of nonlinearities to use in nonlinear mixing function.",
    )
    parser.add_argument(
        "--learn-scm-params",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to learn SCM parameters.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        help="Learning rate scheduler.",
        choices=[None, "cosine"],
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine learning rate scheduler.",
    )
    parser.add_argument(
        "--scm-coeffs-low",
        type=float,
        default=0.5,
        help="Lower bound for SCM coefficients.",
    )
    parser.add_argument(
        "--scm-coeffs-high",
        type=float,
        default=1,
        help="Upper bound for SCM coefficients.",
    )
    parser.add_argument(
        "--scm-coeffs-min-abs-value",
        type=float,
        default=0.5,
        help="Minimum absolute value for SCM coefficients.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=1.0,
        help="Signal-to-noise ratio in latent SCM.",
    )
    parser.add_argument(
        "--adjacency-misspec",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Misspecify adjacency matrix - assume ICA.",
    )
    parser.add_argument(
        "--function-misspec",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Misspecify function class - assume linear.",
    )
    parser.add_argument(
        "--net-hidden-layers",
        type=int,
        default=3,
        help="Number of hidden layers in nonlinear encoder.",
    )
    parser.add_argument(
        "--net-hidden-layers-cbn",
        type=int,
        default=3,
        help="Number of hidden layers in latent CBN model.",
    )
    parser.add_argument(
        "--net-hidden-dim",
        type=int,
        default=128,
        help="Number of hidden dimensions in nonlinear encoder.",
    )
    parser.add_argument(
        "--net-hidden-dim-cbn",
        type=int,
        default=128,
        help="Number of hidden dimensions in latent CBN model.",
    )
    parser.add_argument(
        "--fix-mechanisms",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fix fixable mechanisms in latents.",
    )
    parser.add_argument(
        "--fix-all-intervention-targets",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fix all intervention targets.",
    )
    parser.add_argument(
        "--nonparametric-base-distr",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use nonparametric base distribution for flows.",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to log to weights and biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cauca",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./result",
        help="folder name.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index when accelerator=gpu.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='synthetic',
        choices=["synthetic", "flow", "pendulum"],
        help="data",
    )


    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #
    # args.k_flows = 12
    # args.dgp == "graph-7-random-1"
    # args.n_nonlinearities = 3
    # args.lr_scheduler = "cosine"
    #

    if args.function_misspec:
        assert (
            args.mixing == "nonlinear" and args.model == "linear"
        ), "Function not misspecified."

    checkpoint_dir = Path(args.checkpoint_root_dir) / "default"
    logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )
    if args.mode == 'synthetic':
        multi_env_dgp = make_multi_env_dgp(
            latent_dim=DGP[args.dgp]["num_causal_variables"],
            observation_dim=DGP[args.dgp]["observation_dim"],
            adjacency_matrix=DGP[args.dgp]["adj_matrix"],
            intervention_targets_per_env=DGP[args.dgp]["int_targets"],
            ################################
            num_blocks=0,
            ##################################
            noise_shift_type=args.noise_shift_type,
            mixing=args.mixing,  # nonlinear
            scm=args.scm,  # linear
            n_nonlinearities=args.n_nonlinearities,  # 3
            scm_coeffs_low=args.scm_coeffs_low,
            scm_coeffs_high=args.scm_coeffs_high,
            coeffs_min_abs_value=args.scm_coeffs_min_abs_value,
            edge_prob=DGP[args.dgp].get("edge_prob", None),
            snr=args.snr,  # 1
            selected_idx= DGP[args.dgp]["selected_idx"],
            observed_idx = DGP[args.dgp]["observed_idx"]
        )
        data_module = MultiEnvDataModule(
            multi_env_dgp=multi_env_dgp,
            num_samples_per_env=DGP[args.dgp]["num_samples_per_env"],
            batch_size=args.batch_size,
            num_workers=os.cpu_count(),
            intervention_targets_per_env=DGP[args.dgp]["int_targets"],
            log_dir=checkpoint_dir / "data_stats",
            selected_idx= DGP[args.dgp]["selected_idx"],
            observed_idx= DGP[args.dgp]["observed_idx"]
        )
        model_graph = data_module.medgp.adjacency_matrix
        data_module.setup()
        latent_dim = DGP[args.dgp]["num_causal_variables"]
        data_dim = latent_dim
        selected_idx =  DGP[args.dgp]["selected_idx"]
        observed_idx = DGP[args.dgp]["observed_idx"]
    elif args.mode == "pendulum":
        data_module = ImageDataModule(
            dataset=args.mode,
            batch_size=args.batch_size,
            num_workers=os.cpu_count(),
            log_dir=checkpoint_dir / "data_stats",
            selected_idx= [0,1], # angle, light
            observed_idx= [] #water flow
        )
        data_module.setup()
        model_graph = np.array([[0, 0, 1, 1],  # pendulum angle (0)
                        [0, 0, 1, 1],  # light pos (1)
                        [0, 0, 0, 0],  # shadow len (2)
                        [0, 0, 0, 0]])  # shadow pos (3)
        latent_dim = 4
        data_dim = 4*96*96
        selected_idx = data_module.selected_idx
        observed_idx = data_module.observed_idx
        n_blocks = latent_dim - len(selected_idx)
        block_dims = [1,1]
    elif args.mode == "flow":
        data_module = ImageDataModule(
            dataset = args.mode,
            batch_size=args.batch_size,
            num_workers=os.cpu_count(),
            log_dir=checkpoint_dir / "data_stats",
            selected_idx= [0], #ball size
            observed_idx= [2] #water flow
        )
        data_module.setup()
        model_graph = np.array([[0, 1, 0, 1],  # ball size (0)
                        [0, 0, 1, 0],  # water height (1)
                        [0, 0, 0, 0],  # water flow (2)
                        [0, 0, 1, 0]])  # hole (3)
        data_dim = 4*96*96
        latent_dim = 4
        selected_idx = data_module.selected_idx
        observed_idx = data_module.observed_idx
        n_blocks = latent_dim - len(selected_idx)
        block_dims = [1,1]
    pl.seed_everything(args.training_seed, workers=True)
    intervention_targets_per_env = DGP[args.dgp]["int_targets"]

    # Resolve the actual device before constructing iVAE's distribution tensors.
    if args.accelerator == "gpu":
        _devices = [args.device]
    else:
        _devices = 1
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=None,
        callbacks=[checkpoint_callback] if args.wandb else [],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        devices=_devices,
    )
    _tensor_device = trainer.strategy.root_device

    # Model Initialization
    if args.model == "nonlinear":
        model = iVAEWrapper(
            data_dim=data_dim,
            latent_dim= latent_dim,
            lr=args.lr,
            lr_scheduler=args.lr_scheduler,
            lr_min=args.lr_min,
            selected_idx= selected_idx,
            observed_idx= observed_idx,
            dgp_name = args.dgp,
            seed = args.training_seed,
            result_dir = args.result_dir,
            device=_tensor_device,
            mode = args.mode
        )
    else:
        raise ValueError(f"Unknown model type {args.model}")

    trainer.fit(
        model,
        datamodule=data_module,
    )
    print(f"Checkpoint dir: {checkpoint_dir}")

    trainer.test(model=model, datamodule=data_module, ckpt_path=None)
