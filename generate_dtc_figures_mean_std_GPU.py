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
    compute_boolean_function_ids,
    compute_variable_dependence_dtc,
)

# We never need gradients here
torch.set_grad_enabled(False)


def compute_gcc_size_from_adjacency(adjacency: torch.Tensor) -> int:
    '''
    Compute the size of the largest connected component (GCC) in an
    undirected graph given its adjacency matrix.

    Parameters
    ----------
    adjacency : torch.Tensor
        Adjacency matrix of shape (N, N) with 0/1 entries. This function
        keeps the tensor on its current device and only transfers small
        neighbor lists to CPU when needed.

    Returns
    -------
    int
        Size (number of nodes) in the largest connected component.
    '''
    # Binarize on the current device
    A = (adjacency > 0.5)
    N = A.shape[0]

    if N == 0:
        return 0
    if N == 1:
        # Count the node itself as a component of size 1
        return 1

    visited = torch.zeros(N, dtype=torch.bool, device=A.device)
    max_size = 0

    for start in range(N):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component_size = 0

        while stack:
            v = stack.pop()
            component_size += 1

            # Neighbors as a small 1D tensor on device
            neighbors = torch.nonzero(A[v], as_tuple=False).view(-1)

            # Iterate in Python over neighbor indices
            for u in neighbors.tolist():
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)

        if component_size > max_size:
            max_size = component_size

    return int(max_size)


def run_boolean_z_scan(
    num_nodes: int = 100,
    phi_value: float = 0.1,
    k: int = 2,
    z_min: float = 0.0,
    z_max: float = 2.0,
    z_step: float = 0.5,
    num_trials_per_z: int = 10,
    run_timestamp: str = "00000000_000000",
) -> pd.DataFrame:
    '''
    Run Monte Carlo Boolean cascades over a range of mean degrees z.

    For each z, we compute:
    - mean_num_unique_functions, std_num_unique_functions
      (unique functions per network)
    - mean_dtc, std_dtc
      (mean DTC per network, excluding input nodes)
    - mean_gcc_size, std_gcc_size
      (GCC size per network)
    - total_unique_functions
      (union of all functions seen across all trials at that z)

    The results are saved to a timestamped CSV and returned as a DataFrame.
    '''
    device = get_default_device()

    # Build z grid
    z_values: List[float] = []
    z = z_min
    while z <= z_max + 1e-9:
        z_values.append(z)
        z += z_step

    rows = []

    print("z values:", z_values)

    for z in z_values:
        print()
        print("z:", z)

        if num_nodes <= 1:
            p = 0.0
        else:
            p = z / float(num_nodes - 1)

        # Per-z accumulators
        all_function_ids_union = set()

        num_unique_per_trial: List[int] = []
        mean_dtc_per_trial: List[float] = []
        gcc_size_per_trial: List[float] = []

        for trial_idx in range(num_trials_per_z):
            if trial_idx % max(1, num_trials_per_z // 10) == 0:
                print("  trial:", trial_idx)

            # Create network for this trial (on GPU if available)
            net = LTMCascadeNetwork(
                num_nodes=num_nodes,
                directed=False,
                device=device,
            )
            net.generate_erdos_renyi(p)
            net.set_thresholds(phi_value=phi_value)

            # GCC size for this network (stays mostly on device)
            gcc_size = compute_gcc_size_from_adjacency(net.adjacency)
            gcc_size_per_trial.append(float(gcc_size))

            input_indices = list(range(k))

            # Truth table for all nodes: shape (2^k, N), on device
            truth_table_states = run_truth_table_for_network(
                network=net,
                input_indices=input_indices,
                max_steps=0,
            )

            # Function IDs and DTC per node (tensors on device)
            function_ids = compute_boolean_function_ids(truth_table_states)
            dtc_tensor = compute_variable_dependence_dtc(truth_table_states, k)

            # --- Unique functions in THIS network (GPU -> CPU only for IDs) ---
            fids_cpu = function_ids.detach().cpu()
            unique_this_trial = torch.unique(fids_cpu)
            num_unique_per_trial.append(int(unique_this_trial.numel()))

            # Update global union as Python ints
            all_function_ids_union.update(int(fid) for fid in unique_this_trial.tolist())

            # --- Mean DTC per network (exclude input nodes if k > 0) ---
            dtc_tensor_net = dtc_tensor
            if k > 0 and dtc_tensor_net.shape[0] > k:
                dtc_tensor_net = dtc_tensor_net[k:]

            # Mean stays on device, then scalar back to Python
            mean_dtc_trial = float(dtc_tensor_net.to(torch.float32).mean().item())
            mean_dtc_per_trial.append(mean_dtc_trial)

        # Aggregate stats for this z (small CPU tensors)
        if num_unique_per_trial:
            num_unique_tensor = torch.tensor(num_unique_per_trial, dtype=torch.float32)
            mean_num_unique = float(num_unique_tensor.mean().item())
            std_num_unique = float(num_unique_tensor.std(unbiased=False).item())
        else:
            mean_num_unique = 0.0
            std_num_unique = 0.0

        if mean_dtc_per_trial:
            dtc_tensor_all = torch.tensor(mean_dtc_per_trial, dtype=torch.float32)
            mean_dtc = float(dtc_tensor_all.mean().item())
            std_dtc = float(dtc_tensor_all.std(unbiased=False).item())
        else:
            mean_dtc = 0.0
            std_dtc = 0.0

        if gcc_size_per_trial:
            gcc_tensor = torch.tensor(gcc_size_per_trial, dtype=torch.float32)
            mean_gcc_size = float(gcc_tensor.mean().item())
            std_gcc_size = float(gcc_tensor.std(unbiased=False).item())
        else:
            mean_gcc_size = 0.0
            std_gcc_size = 0.0

        rows.append(
            {
                "run_timestamp": run_timestamp,
                "z": float(z),
                "p": float(p),
                "total_unique_functions": int(len(all_function_ids_union)),
                "mean_num_unique_functions": float(mean_num_unique),
                "std_num_unique_functions": float(std_num_unique),
                "mean_dtc": float(mean_dtc),
                "std_dtc": float(std_dtc),
                "mean_gcc_size": float(mean_gcc_size),
                "std_gcc_size": float(std_gcc_size),
                "num_nodes": int(num_nodes),
                "num_trials": int(num_trials_per_z),
                "k": int(k),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs("./data", exist_ok=True)
    csv_path = f"./data/boolean_z_scan_stats_{run_timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")
    return df



def main():
    # ---------------------------------------------------------
    # Run the scan (example settings)
    # ---------------------------------------------------------

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    stats_df = run_boolean_z_scan(
        num_nodes=10000,
        phi_value=0.1,
        k=5,
        z_min=0.0,
        z_max=4.0,
        z_step=0.1,
        num_trials_per_z=300,
        run_timestamp=run_timestamp,
    )

    # ---------------------------------------------------------
    # Plot results: mean ± std vs z, with shaded bands
    # ---------------------------------------------------------

    os.makedirs("./data/figures", exist_ok=True)

    z_vals = stats_df["z"].values

    # 1) Number of unique functions vs z
    mean_unique = stats_df["mean_num_unique_functions"].values
    std_unique = stats_df["std_num_unique_functions"].values

    plt.figure()
    plt.plot(z_vals, mean_unique, marker="o")
    plt.fill_between(
        z_vals,
        mean_unique - std_unique,
        mean_unique + std_unique,
        alpha=0.2,
    )
    plt.xlabel("z (mean degree)")
    plt.ylabel("Unique Boolean functions per network")
    plt.title("Unique Boolean functions vs z")
    plt.grid(False)
    plt.savefig(f"./data/figures/unique_functions_vs_z_{run_timestamp}.pdf")

    # 2) Mean DTC vs z
    mean_dtc = stats_df["mean_dtc"].values
    std_dtc = stats_df["std_dtc"].values

    plt.figure()
    plt.plot(z_vals, mean_dtc, marker="o")
    plt.fill_between(
        z_vals,
        mean_dtc - std_dtc,
        mean_dtc + std_dtc,
        alpha=0.2,
    )
    plt.xlabel("z (mean degree)")
    plt.ylabel("Mean DTC (per network, excl. inputs)")
    plt.title("Mean decision-tree complexity vs z")
    plt.grid(False)
    plt.savefig(f"./data/figures/mean_dtc_vs_z_{run_timestamp}.pdf")

    # 3) GCC size vs z
    mean_gcc = stats_df["mean_gcc_size"].values
    std_gcc = stats_df["std_gcc_size"].values

    plt.figure()
    plt.plot(z_vals, mean_gcc, marker="o")
    plt.fill_between(
        z_vals,
        mean_gcc - std_gcc,
        mean_gcc + std_gcc,
        alpha=0.2,
    )
    plt.xlabel("z (mean degree)")
    plt.ylabel("Mean GCC size")
    plt.title("Giant component size vs z")
    plt.grid(False)
    plt.savefig(f"./data/figures/mean_gcc_vs_z_{run_timestamp}.pdf")

    return stats_df


if __name__ == "__main__":
    main()


