"""Extra synthetic DGPs for the graph robustness sweep.

These graphs generalise ``c_real`` from ``config.DGP`` (Fig. 1d in the
paper): 4-6 nodes, multiple observables, denser parent sets. They are
exported as ``EXTRA_DGPS``, a dict with the same schema as ``config.DGP``
entries, and are injected at runtime (never written back to ``config.py``).

Keys follow the ``config.DGP`` entry schema:
    num_causal_variables, adj_matrix, edge_prob, int_targets,
    num_samples_per_env, observation_dim, num_blocks, block_dims,
    selected_idx, observed_idx

Every new graph specifies two selected variables. These are fixed auxiliary
sets, not outputs of an automatic graph-selection step.
"""

from __future__ import annotations

import numpy as np
import torch


def _no_intervention(n: int) -> torch.Tensor:
    """Single observational environment tensor ``int_targets``."""
    return torch.tensor([[0] * n])


EXTRA_DGPS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # c_dense: 4 nodes, same variables as c_real with two extra edges.
    # Nodes 1 and 2 have multiple parents; node 3 has parent 0.
    # ------------------------------------------------------------------
    "c_dense": {
        "num_causal_variables": 4,
        "adj_matrix": np.array([
            [0, 1, 1, 1],  # 0 -> {1,2,3}
            [0, 0, 1, 0],  # 1 -> 2
            [0, 0, 0, 0],
            [0, 1, 1, 0],  # 3 -> {1,2}
        ]),
        "edge_prob": 0.5,
        "int_targets": _no_intervention(4),
        "num_samples_per_env": 10_000,
        "observation_dim": 4,
        "num_blocks": 2,
        "block_dims": [1, 1],
        "selected_idx": [3, 2],
        "observed_idx": [],
    },
    # ------------------------------------------------------------------
    # c_deep: 5-node DAG with a dependency chain and shortcut edges.
    # ------------------------------------------------------------------
    "c_deep": {
        "num_causal_variables": 5,
        "adj_matrix": np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]),
        "edge_prob": 0.5,
        "int_targets": _no_intervention(5),
        "num_samples_per_env": 10_000,
        "observation_dim": 5,
        "num_blocks": 3,
        "block_dims": [1, 1, 1],
        "selected_idx": [4, 3],
        "observed_idx": [],
    },
    # ------------------------------------------------------------------
    # c_chain: 5-node pure cascade 0 -> 1 -> 2 -> 3 -> 4 (no shortcuts).
    # The previous "c_wide" entry was a 1-hub graph identical to the
    # paper's DGP ``a`` (config.DGP["a"]: row 4 = [1,1,1,1,0]); replacing
    # it with a pure chain gives a genuinely new topology where the two
    # deepest variables (3, 4) serve as selection targets.
    # Adjacency: each row i has a single edge i -> i+1 (i < 4).
    # ------------------------------------------------------------------
    "c_chain": {
        "num_causal_variables": 5,
        "adj_matrix": np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]),
        "edge_prob": 0.5,
        "int_targets": _no_intervention(5),
        "num_samples_per_env": 10_000,
        "observation_dim": 5,
        # num_blocks = N - |selected_idx| = 5 - 2 = 3.
        "num_blocks": 3,
        "block_dims": [1, 1, 1],
        "selected_idx": [4, 3],
        "observed_idx": [],
    },
    # ------------------------------------------------------------------
    # c_obs_chain: the chain 0 -> 1 -> 2 feeds nodes 4 and 5.
    # Node 5 feeds observed-but-unselected node 3; nodes 4 and 5 are selected.
    # ------------------------------------------------------------------
    "c_obs_chain": {
        "num_causal_variables": 6,
        "adj_matrix": np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]),
        "edge_prob": 0.5,
        "int_targets": _no_intervention(6),
        "num_samples_per_env": 10_000,
        "observation_dim": 6,
        # num_blocks = N - len(selected_idx), matching the convention in
        # config.DGP (see e.g. c_aug: num_blocks = 4 - 1 = 3, even with
        # an observed_idx). Here 6 - 2 = 4.
        "num_blocks": 4,
        "block_dims": [1, 1, 1, 1],
        "selected_idx": [4, 5],
        "observed_idx": [3],
    },
    # ------------------------------------------------------------------
    # c_hub6: 6-node, mildly dense graph with selected nodes 2 and 5,
    # including the non-leaf node 2.
    # ------------------------------------------------------------------
    "c_hub6": {
        "num_causal_variables": 6,
        "adj_matrix": np.array([
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]),
        "edge_prob": 0.5,
        "int_targets": _no_intervention(6),
        "num_samples_per_env": 10_000,
        "observation_dim": 6,
        # num_blocks = N - |selected_idx| = 6 - 2 = 4.
        "num_blocks": 4,
        "block_dims": [1, 1, 1, 1],
        "selected_idx": [2, 5],
        "observed_idx": [],
    },
}


def register_extra_dgps() -> list[str]:
    """Inject ``EXTRA_DGPS`` into ``config.DGP`` at runtime.

    Returns the list of newly added keys. Safe to call multiple times:
    existing keys are not overwritten.
    """
    import config  # noqa: WPS433  -- intentional runtime import
    added: list[str] = []
    for name, spec in EXTRA_DGPS.items():
        if name not in config.DGP:
            config.DGP[name] = spec
            added.append(name)
    return added


if __name__ == "__main__":
    added = register_extra_dgps()
    print("Registered extra DGPs:", added)
    for name, spec in EXTRA_DGPS.items():
        n = spec["num_causal_variables"]
        sel = spec["selected_idx"]
        obs = spec["observed_idx"]
        edges = int(spec["adj_matrix"].sum())
        print(f"  {name}: N={n}, edges={edges}, "
              f"selected={sel}, observed={obs}")
