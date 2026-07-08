import numpy as np
import torch

DGP = {
    "a": {
        "num_causal_variables": 5,  # N
        "adj_matrix": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
            ]
        ),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 5,  # D
        "num_blocks": 4,
        "block_dims": [1, 1, 1, 1],
        "selected_idx":[4],
        "observed_idx":[],
    },
    "b_aug": {
        "num_causal_variables": 5,  # N
        "adj_matrix": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
            ]
        ),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 5,  # D
        "num_blocks": 4,
        "block_dims": [1, 1, 1, 1],
        "selected_idx":[4],
        "observed_idx":[],
    },
    "b": {
        "num_causal_variables": 5,  # N
        "adj_matrix": np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,1,1],[0,0,0,0,0],[1,1,0,1,0]]),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 5,  # D
        "num_blocks": 4,
        "block_dims": [1, 1, 1, 1],
        "selected_idx":[4],
        "observed_idx":[],
    },
    "c_aug": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array([[0,0,1,0],[0,0,1,0],[0,0,0,0],[1,1,1,0]]),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 4,  # D
        "num_blocks": 3, # cardinality of z_u^-
        "block_dims": [1, 1],
        "selected_idx":[3],
        "observed_idx":[2],
    },
    "c": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array([[0,0,1,0],[0,0,1,0],[0,0,0,0],[1,1,0,0]]),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 4,  # D
        "num_blocks": 3,
        "block_dims": [1, 1],
        "selected_idx":[3],
        "observed_idx":[2],
    },
    "c_real": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array([[0,0,1,1],[0,0,1,0],[0,0,0,0],[0,1,0,0]]),
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0]
            ]
        ),
        "num_samples_per_env": 10_000,
        "observation_dim": 4,  # D
        "num_blocks": 3,
        "block_dims": [1, 1],
        "selected_idx":[3],
        "observed_idx":[2],
    },
    # "c_modi": {
    #     "num_causal_variables": 5,  # N
    #     "adj_matrix": np.array([[],[0,0,1,1,0],[0,0,1,0,0],[0,0,0,0,0],[0,1,0,0,0]]),
    #     "edge_prob": 0.5,
    #     "int_targets": torch.tensor(
    #         [
    #             [0, 0, 0, 0]
    #         ]
    #     ),
    #     "num_samples_per_env": 10_000,
    #     "observation_dim": 5,  # D
    #     "num_blocks": 3,
    #     "block_dims": [1, 1],
    #     "selected_idx":[2],
    #     "observed_idx":[3],
    # }
}
