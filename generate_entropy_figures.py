import math
from typing import List

import torch
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from ltm_core import LTMCascadeNetwork, get_default_device
from ltm_boolean_mc import (
    run_truth_table_for_network,
)

# ============================================================
# Helpers: entropy
# ============================================================

def compute_shannon_entropy_binary(prob_one: float) -> float:
    '''
    Compute Shannon entropy of a Bernoulli variable.

    Parameters
    ----------
    prob_one : float
        Probability P(X = 1).

    Returns
    -------
    float
        Entropy H(X) in bits.
    '''
    p1 = max(0.0, min(1.0, float(prob_one)))
    p0 = 1.0 - p1
    if p0 <= 0.0 or p1 <= 0.0:
        return 0.0
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))


def run_output_entropy_z_scan(
    num_nodes: int = 100,
    phi_value: float = 0.1,
    k: int = 2,
    z_min: float = 0.0,
    z_max: float = 2.0,
    z_step: float = 0.5,
    num_trials_per_z: int = 10,
    output_node_index: int | None = None,
    run_timestamp: str = "00000000_000000",
) -> pd.DataFrame:
    '''
    Run a z-scan and compute entropy of a single output node's final state.

    For each z, this computes mean and std of the output entropy across trials.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N in each network.
    phi_value : float
        Homogeneous threshold phi.
    k : int
        Number of inputs (nodes [0, ..., k-1]).
    z_min, z_max : float
        Range of mean degree z.
    z_step : float
        Step in z.
    num_trials_per_z : int
        Number of networks per z.
    output_node_index : int or None
        Index of the output node to measure. If None, uses node k.
    run_timestamp : str
        Timestamp string used for naming the output CSV file.

    Returns
    -------
    pandas.DataFrame
        Columns: run_timestamp, z, p, mean_output_entropy, std_output_entropy,
                 num_nodes, num_trials, k, output_node
    '''
    device = get_default_device()

    if output_node_index is None:
        output_node_index = k  # first non-input node by default

    # Build z grid
    z_values: List[float] = []
    z = z_min
    while z <= z_max + 1e-9:
        z_values.append(z)
        z += z_step

    rows = []

    print("Entropy z values:", z_values)

    for z_val in z_values:
        # Convert z -> p
        if num_nodes <= 1:
            p = 0.0
        else:
            p = z_val / float(num_nodes - 1)

        entropies: List[float] = []

        for trial_idx in range(num_trials_per_z):
            if trial_idx % max(1, num_trials_per_z // 10) == 0:
                print(f"[Entropy] z={z_val:.3f}, trial={trial_idx}")

            # Create ER network at this p
            net = LTMCascadeNetwork(
                num_nodes=num_nodes,
                directed=False,
                device=device,
            )
            net.generate_erdos_renyi(p)
            net.set_thresholds(phi_value=phi_value)

            input_indices = list(range(k))

            # Truth table: (2^k, N)
            truth_table_states = run_truth_table_for_network(
                network=net,
                input_indices=input_indices,
                max_steps=0,
            )

            # Extract final states of chosen output node
            output_states = truth_table_states[:, output_node_index]
            # Estimate P(output = 1) over input patterns
            prob_one = float(output_states.to(torch.float32).mean().item())
            H = compute_shannon_entropy_binary(prob_one)
            entropies.append(H)

        if entropies:
            entropy_tensor = torch.tensor(entropies, dtype=torch.float32)
            mean_H = float(entropy_tensor.mean().item())
            std_H = float(entropy_tensor.std(unbiased=False).item())
        else:
            mean_H = 0.0
            std_H = 0.0

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "z": float(z_val),
                "p": float(p),
                "mean_output_entropy": float(mean_H),
                "std_output_entropy": float(std_H),
                "num_nodes": int(num_nodes),
                "num_trials": int(num_trials_per_z),
                "k": int(k),
                "output_node": int(output_node_index),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs("./data", exist_ok=True)
    csv_path = f"./data/entropy_z_scan_{run_timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved entropy stats to {csv_path}")
    return df


# ============================================================
# Helpers: transfer entropy
# ============================================================

def compute_te_from_counts(counts: torch.Tensor) -> float:
    '''
    Compute TE from 2x2x2 counts for binary source and target.

    Parameters
    ----------
    counts : torch.Tensor
        Tensor of shape (2, 2, 2) with integer counts:
        counts[x_next, y, x] where:
            x_next = target(t+1)
            y      = target(t)
            x      = source(t)

    Returns
    -------
    float
        Transfer entropy TE(source -> target) in bits.
    '''

    if counts.device.type == "mps":
        counts = counts.to(torch.float32)
    else:
        counts = counts.to(torch.float64)

    total = counts.sum().item()

    if total <= 0:
        return 0.0

    te = 0.0

    # Marginals
    # p(y, x) = sum_{x_next} counts[x_next, y, x]
    p_yx = counts.sum(dim=0)  # shape (2, 2)
    # p(y)   = sum_{x_next, x} counts[x_next, y, x]
    p_y = counts.sum(dim=(0, 2))  # shape (2,)

    for x_next in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                c_xyz = counts[x_next, y, x].item()
                if c_xyz <= 0:
                    continue

                p_xyz = c_xyz / total

                denom_yx = p_yx[y, x].item()
                denom_y = p_y[y].item()

                if denom_yx <= 0 or denom_y <= 0:
                    continue

                # p(x_next | y, x)
                p_xnext_given_yx = c_xyz / denom_yx

                # p(x_next | y) = sum_x' counts[x_next, y, x'] / p(y)
                num_xnext_y = counts[x_next, y, :].sum().item()
                p_xnext_given_y = num_xnext_y / denom_y

                if p_xnext_given_y <= 0 or p_xnext_given_yx <= 0:
                    continue

                te += p_xyz * math.log2(p_xnext_given_yx / p_xnext_given_y)

    return float(te)


def compute_te_for_network(
    net: LTMCascadeNetwork,
    input_indices: List[int],
    output_index: int,
    max_steps: int = 0,
) -> float:
    '''
    Compute total TE (sum over inputs) from input nodes to a single output node
    for a given LTM network, by aggregating transitions over all input patterns.

    Parameters
    ----------
    net : LTMCascadeNetwork
        Initialized network (adjacency, degrees, thresholds already set).
    input_indices : list of int
        Indices of input nodes [0, ..., k-1].
    output_index : int
        Index of the output node.
    max_steps : int
        Maximum cascade steps; 0 = until convergence.

    Returns
    -------
    float
        Total TE from all inputs to the chosen output node (bits).
    '''
    device = net.device
    k = len(input_indices)
    if k <= 0:
        return 0.0

    # For each input node we maintain a 2x2x2 count tensor
    counts_per_input = [
        torch.zeros((2, 2, 2), device=device, dtype=torch.float32)
        for _ in range(k)
    ]

    # We will explore all 2^k input patterns for TE (as for truth tables)
    num_patterns = 2 ** k
    for pattern_idx in range(num_patterns):
        # Decode pattern_idx into bits for inputs
        bits = []
        for j in range(k):
            shift = k - 1 - j
            bit = (pattern_idx >> shift) & 1
            bits.append(bit)

        # Initial state: inputs set to bits, others 0
        s_init = torch.zeros((net.num_nodes,), device=device, dtype=torch.float32)
        for idx, bit in zip(input_indices, bits):
            s_init[idx] = float(bit)

        # Run cascade and get trajectory (T, N)
        final_state, trajectory = net.run_cascade(
            initial_state=s_init,
            input_indices=input_indices,
            max_steps=max_steps,
            return_trajectory=True,
        )

        # trajectory[t, node]
        # we want transitions t -> t+1
        T = trajectory.shape[0]
        if T < 2:
            continue

        # Binarize as int
        traj_int = (trajectory > 0.5).to(torch.int64)

        # For each input node, accumulate counts
        for j_input, node_idx in enumerate(input_indices):
            source_series = traj_int[:, node_idx]      # shape (T,)
            target_series = traj_int[:, output_index]  # shape (T,)

            # Collect transitions
            for t in range(T - 1):
                x_next = int(target_series[t + 1].item())  # target at t+1
                y = int(target_series[t].item())           # target at t
                x = int(source_series[t].item())           # source at t
                counts_per_input[j_input][x_next, y, x] += 1.0

    # Compute TE per input and sum
    total_te = 0.0
    for counts in counts_per_input:
        total_te += compute_te_from_counts(counts)

    return total_te


def run_te_z_scan(
    num_nodes: int = 100,
    phi_value: float = 0.1,
    k: int = 2,
    z_min: float = 0.0,
    z_max: float = 2.0,
    z_step: float = 0.5,
    num_trials_per_z: int = 10,
    output_node_index: int | None = None,
    run_timestamp: str = "00000000_000000",
) -> pd.DataFrame:
    '''
    Run a z-scan and compute TE from input nodes to a single output node.

    For each z, this computes mean and std of TE across trials.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N.
    phi_value : float
        Homogeneous threshold.
    k : int
        Number of input nodes [0, ..., k-1].
    z_min, z_max : float
        Range of mean degree z.
    z_step : float
        Step in z.
    num_trials_per_z : int
        Number of networks per z.
    output_node_index : int or None
        Output node index; if None, uses node k.
    run_timestamp : str
        Timestamp string used for naming the output CSV file.

    Returns
    -------
    pandas.DataFrame
        Columns: run_timestamp, z, p, mean_te, std_te,
                 num_nodes, num_trials, k, output_node
    '''
    device = get_default_device()

    if output_node_index is None:
        output_node_index = k

    # Build z grid
    z_values: List[float] = []
    z = z_min
    while z <= z_max + 1e-9:
        z_values.append(z)
        z += z_step

    rows = []

    print("TE z values:", z_values)

    for z_val in z_values:
        # Convert z -> p
        if num_nodes <= 1:
            p = 0.0
        else:
            p = z_val / float(num_nodes - 1)

        te_values: List[float] = []

        for trial_idx in range(num_trials_per_z):
            if trial_idx % max(1, num_trials_per_z // 10) == 0:
                print(f"[TE] z={z_val:.3f}, trial={trial_idx}")

            net = LTMCascadeNetwork(
                num_nodes=num_nodes,
                directed=False,
                device=device,
            )
            net.generate_erdos_renyi(p)
            net.set_thresholds(phi_value=phi_value)

            input_indices = list(range(k))

            te_val = compute_te_for_network(
                net=net,
                input_indices=input_indices,
                output_index=output_node_index,
                max_steps=0,
            )
            te_values.append(te_val)

        if te_values:
            te_tensor = torch.tensor(te_values, dtype=torch.float32)
            mean_te = float(te_tensor.mean().item())
            std_te = float(te_tensor.std(unbiased=False).item())
        else:
            mean_te = 0.0
            std_te = 0.0

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "z": float(z_val),
                "p": float(p),
                "mean_te": float(mean_te),
                "std_te": float(std_te),
                "num_nodes": int(num_nodes),
                "num_trials": int(num_trials_per_z),
                "k": int(k),
                "output_node": int(output_node_index),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs("./data", exist_ok=True)
    csv_path = f"./data/te_z_scan_{run_timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved TE stats to {csv_path}")
    return df


def main():
    
    # ============================================================
    # Run scans (set these to match your DTC run)
    # ============================================================
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Adjust these to exactly match your DTC scan parameters
    num_nodes = 10000
    
    phi_value = 0.1
    k = 5
    z_min = 0.0
    z_max = 4.0
    z_step = 0.1
    
    # You can choose fewer trials for TE if needed
    num_trials_entropy = 500
    num_trials_te = 200
    
    entropy_df = run_output_entropy_z_scan(
        num_nodes=num_nodes,
        phi_value=phi_value,
        k=k,
        z_min=z_min,
        z_max=z_max,
        z_step=z_step,
        num_trials_per_z=num_trials_entropy,
        output_node_index=None,
        run_timestamp=run_timestamp,
    )
    
    te_df = run_te_z_scan(
        num_nodes=num_nodes,
        phi_value=phi_value,
        k=k,
        z_min=z_min,
        z_max=z_max,
        z_step=z_step,
        num_trials_per_z=num_trials_te,
        output_node_index=None,
        run_timestamp=run_timestamp,
    )
    
    # ============================================================
    # Plot results: mean ± std vs z, with shaded bands
    # ============================================================
    
    os.makedirs("./data/figures", exist_ok=True)
    
    # Entropy plot
    z_vals_entropy = entropy_df["z"].values
    mean_H = entropy_df["mean_output_entropy"].values
    std_H = entropy_df["std_output_entropy"].values
    
    plt.figure()
    plt.plot(z_vals_entropy, mean_H, marker="o")
    plt.fill_between(
        z_vals_entropy,
        mean_H - std_H,
        mean_H + std_H,
        alpha=0.2,
    )
    plt.xlabel("z (mean degree)")
    plt.ylabel("Mean output entropy (bits)")
    plt.title("Output entropy vs z")
    plt.grid(False)
    plt.savefig(f"./data/figures/output_entropy_vs_z_{run_timestamp}.pdf")
    
    # TE plot
    z_vals_te = te_df["z"].values
    mean_te = te_df["mean_te"].values
    std_te = te_df["std_te"].values
    
    plt.figure()
    plt.plot(z_vals_te, mean_te, marker="o")
    plt.fill_between(
        z_vals_te,
        mean_te - std_te,
        mean_te + std_te,
        alpha=0.2,
    )
    plt.xlabel("z (mean degree)")
    plt.ylabel("Mean TE (bits)")
    plt.title("Transfer entropy vs z")
    plt.grid(False)
    plt.savefig(f"./data/figures/te_vs_z_{run_timestamp}.pdf")
    
    return(entropy_df, te_df)


if __name__ == "__main__":
    main()
    
