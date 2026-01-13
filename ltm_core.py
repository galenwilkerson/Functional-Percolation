#!/usr/bin/env python

# ============================
# file: ltm_core.py
# ============================

"""
Core utilities for Linear Threshold Model (LTM) cascades on Erdős–Rényi networks.

This module provides:

- get_default_device()
- LTMCascadeNetwork
    * ER graph generation (directed / undirected)
    * GPU/CPU vectorized LTM cascades

- LTMCascadeMonteCarlo
    * Single-cascade runs for a given edge probability p (or mean degree z via conversion)
    * Sampling cascade sizes for a fixed p
    * Monte Carlo cascade size statistics over a range of p (or z via conversion)
"""

from typing import List, Optional, Dict
import argparse
import json
import torch
import os 
import csv
import datetime

def default_output_path():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./data", exist_ok=True)
    return f"./data/results_{ts}.csv"

def write_csv_row(csv_path: str, header: list, row: list):
    """
    Utility: append a row to a CSV file, creating header if missing.
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def save_result_as_csv(csv_path: str, result: object) -> None:
    """
    Save the top-level result object as a CSV table.

    - If `result` is a dict whose values are dicts, we treat it as
      a table of rows (outer key) and columns (inner keys).
    - Otherwise we fall back to a simple key/value or single-value table.
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Case 1: dict of dicts -> table
        if isinstance(result, dict):
            # Check if all values are dict-like
            if all(isinstance(v, dict) for v in result.values()):
                # Collect all inner keys
                inner_keys = sorted({k for d in result.values() for k in d.keys()})
                header = ["key"] + inner_keys
                writer.writerow(header)

                for outer_key, inner_dict in result.items():
                    row = [outer_key]
                    for k in inner_keys:
                        row.append(inner_dict.get(k, ""))
                    writer.writerow(row)
                return

            # Case 2: simple dict -> key/value rows
            writer.writerow(["key", "value"])
            for k, v in result.items():
                # Use JSON encoding for arbitrary Python objects
                writer.writerow([k, json.dumps(v)])
            return

        # Case 3: anything else -> single value
        writer.writerow(["value"])
        writer.writerow([json.dumps(result)])





def get_default_device() -> torch.device:
    '''
    Get the default computation device.

    Order of preference:
    1. CUDA GPU if available
    2. Apple Metal (MPS) if available
    3. CPU otherwise

    Returns
    -------
    torch.device
        Selected device object.
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        # If MPS check fails for any reason, fall back to CPU.
        pass

    return torch.device("cpu")


class LTMCascadeNetwork:
    '''
    Linear Threshold Model (LTM) cascades on an Erdős–Rényi network.

    Encapsulates:
    - ER graph generation (directed / undirected)
    - Adjacency, degrees, thresholds, device
    - Monotone LTM cascades until convergence

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the network (N).
    directed : bool
        Whether the graph is directed.
    device : torch.device
        Device on which all tensors live.
    dtype : torch.dtype
        Data type for adjacency / states.
    adjacency : torch.Tensor or None
        Adjacency matrix A of shape (N, N), entries in {0.0, 1.0}.
        A[i, j] = 1.0 means node j can influence node i.
    degrees : torch.Tensor or None
        Degree vector of shape (N,).
    thresholds : torch.Tensor or None
        Threshold vector of shape (N,).
    '''

    def __init__(
        self,
        num_nodes: int,
        directed: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        '''
        Initialize an empty LTMCascadeNetwork.

        Parameters
        ----------
        num_nodes : int
            Number of nodes N.
        directed : bool, optional
            If True, treat edges as directed. Default is False.
        device : torch.device or None, optional
            Computation device. If None, choose via get_default_device().
        dtype : torch.dtype, optional
            Data type for adjacency and states. Default is torch.float32.
        '''
        self.num_nodes = int(num_nodes)
        self.directed = bool(directed)
        self.device = device if device is not None else get_default_device()
        self.dtype = dtype

        self.adjacency: Optional[torch.Tensor] = None
        self.degrees: Optional[torch.Tensor] = None
        self.thresholds: Optional[torch.Tensor] = None

    def generate_erdos_renyi(self, p: float) -> None:
        '''
        Generate an Erdős–Rényi random graph G(N, p) and store adjacency and degrees.

        Parameters
        ----------
        p : float
            Connection probability.

        Returns
        -------
        None
        '''
        N = self.num_nodes
        p = float(p)

        if self.directed:
            rand_mat = torch.rand((N, N), device=self.device)
            adj = (rand_mat < p).to(self.dtype)
            adj.fill_diagonal_(0.0)
        else:
            rand_mat = torch.rand((N, N), device=self.device)
            upper = torch.triu((rand_mat < p).to(self.dtype), diagonal=1)
            adj = upper + upper.T

        self.adjacency = adj
        self.degrees = self.adjacency.sum(dim=1)

    def set_thresholds(
        self,
        phi_value: Optional[float] = None,
        phi_vector: Optional[torch.Tensor] = None,
    ) -> None:
        '''
        Set node thresholds.

        Parameters
        ----------
        phi_value : float, optional
            Homogeneous threshold in [0, 1].
        phi_vector : torch.Tensor, optional
            Heterogeneous thresholds of shape (N,).

        Returns
        -------
        None
        '''
        if phi_value is None and phi_vector is None:
            raise ValueError("Either phi_value or phi_vector must be provided.")

        if phi_vector is not None:
            if phi_vector.shape[0] != self.num_nodes:
                raise ValueError(
                    f"phi_vector must have shape ({self.num_nodes},), "
                    f"got {phi_vector.shape}"
                )
            self.thresholds = phi_vector.to(device=self.device, dtype=self.dtype)
        else:
            self.thresholds = torch.full(
                (self.num_nodes,),
                float(phi_value),
                device=self.device,
                dtype=self.dtype,
            )

    def _check_ready(self) -> None:
        '''
        Ensure adjacency / degrees / thresholds are initialized.

        Raises
        ------
        RuntimeError
            If adjacency or thresholds are missing.
        '''
        if self.adjacency is None or self.degrees is None:
            raise RuntimeError(
                "Adjacency/degrees not initialized. Call generate_erdos_renyi(p) first."
            )
        if self.thresholds is None:
            raise RuntimeError(
                "Thresholds not initialized. Call set_thresholds(...) first."
            )

    def run_cascade(
        self,
        initial_state: torch.Tensor,
        input_indices: Optional[List[int]] = None,
        max_steps: int = 0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''
        Run a monotone LTM cascade until convergence (or optional step cap).

        Parameters
        ----------
        initial_state : torch.Tensor
            Initial node states (N,), values in {0, 1}.
        input_indices : list of int, optional
            Node indices whose states remain clamped to their initial values.
        max_steps : int, optional
            Maximum iterations. If 0, run until convergence.
        return_trajectory : bool, optional
            If True, also return the full trajectory as a tensor of shape
            (T, N), where T is the number of time steps including the
            initial state. Default is False.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            If return_trajectory is False:
                Final node states (N,) on the network device.
            If return_trajectory is True:
                (final_state, trajectory) where:
                    final_state : (N,)
                    trajectory  : (T, N) stacked over time.
        '''
        self._check_ready()

        s = initial_state.to(device=self.device, dtype=self.dtype).clone()
        if s.shape[0] != self.num_nodes:
            raise ValueError(
                f"initial_state must have shape ({self.num_nodes},), got {s.shape}"
            )

        input_indices_tensor = None
        fixed_inputs = None
        if input_indices is not None and len(input_indices) > 0:
            input_indices_tensor = torch.as_tensor(
                input_indices, device=self.device, dtype=torch.long
            )
            fixed_inputs = s[input_indices_tensor].clone()

        # Optional trajectory logging
        trajectory_states: list[torch.Tensor] = []
        if return_trajectory:
            trajectory_states.append(s.clone())

        step = 0
        while True:
            # Aggregate active neighbors.
            active_neighbors = torch.matmul(self.adjacency, s)

            # Compute fraction of active neighbors for nodes with degree > 0.
            frac_active = torch.zeros_like(s)
            deg_positive = self.degrees > 0
            frac_active[deg_positive] = (
                active_neighbors[deg_positive] / self.degrees[deg_positive]
            )

            # Nodes activate if fraction >= threshold; dynamics is monotone (no deactivation).
            activated = (frac_active >= self.thresholds).to(self.dtype)
            new_s = torch.maximum(s, activated)

            # Clamp input nodes back to their initial values, if any.
            if input_indices_tensor is not None:
                new_s[input_indices_tensor] = fixed_inputs

            # Check for convergence.
            if torch.equal(new_s, s):
                if return_trajectory:
                    trajectory_states.append(new_s.clone())
                s = new_s
                break

            s = new_s
            step += 1

            if return_trajectory:
                trajectory_states.append(s.clone())

            if max_steps > 0 and step >= max_steps:
                break

        if not return_trajectory:
            return s

        trajectory_tensor = torch.stack(trajectory_states, dim=0)  # (T, N)
        return s, trajectory_tensor



class LTMCascadeMonteCarlo:
    '''
    Monte Carlo experiments for LTM cascades on ER networks (non-Boolean).

    All core logic is implemented in terms of the ER edge probability p.
    Mean degree z is only used via thin conversion wrappers where needed.

    Attributes
    ----------
    num_nodes : int
        Number of nodes in each network.
    phi_value : float
        Homogeneous threshold.
    directed : bool
        Whether to use directed graphs.
    device : torch.device
        Device for all network computations.
    '''

    def __init__(
        self,
        num_nodes: int,
        phi_value: float = 0.1,
        directed: bool = False,
        device: Optional[torch.device] = None,
    ):
        '''
        Initialize the Monte Carlo manager.

        Parameters
        ----------
        num_nodes : int
            Number of nodes N.
        phi_value : float, optional
            Homogeneous threshold for all nodes. Default 0.1.
        directed : bool, optional
            If True, use directed ER graphs. Default False.
        device : torch.device or None, optional
            Computation device. If None, choose default.
        '''
        self.num_nodes = int(num_nodes)
        self.phi_value = float(phi_value)
        self.directed = bool(directed)
        self.device = device if device is not None else get_default_device()

    # ----- p <-> z conversion helpers (used only for convenience) -----

    def _z_to_p(self, z: float) -> float:
        '''
        Convert mean degree z to ER edge probability p = z / (N - 1).
        '''
        if self.num_nodes <= 1:
            return 0.0
        return float(z) / float(self.num_nodes - 1)

    def _p_to_z(self, p: float) -> float:
        '''
        Convert edge probability p to mean degree z = p * (N - 1).
        '''
        if self.num_nodes <= 1:
            return 0.0
        return float(p) * float(self.num_nodes - 1)

    # ----- Core p-based helpers -----

    def _create_network_for_p(self, p: float) -> LTMCascadeNetwork:
        '''
        Create and initialize an LTM network with G(N, p).

        Parameters
        ----------
        p : float
            Connection probability.

        Returns
        -------
        LTMCascadeNetwork
            Initialized network.
        '''
        net = LTMCascadeNetwork(
            num_nodes=self.num_nodes,
            directed=self.directed,
            device=self.device,
        )
        net.generate_erdos_renyi(p)
        net.set_thresholds(phi_value=self.phi_value)
        return net

    def _run_cascade_for_p(
        self,
        p: float,
        input_indices: List[int],
        input_pattern: List[int],
        max_steps: int = 0,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''
        Internal helper: run a single cascade for G(N, p) and given inputs.

        Parameters
        ----------
        p : float
            Connection probability.
        input_indices : list of int
            Indices of input nodes.
        input_pattern : list of int
            Binary input pattern aligned with input_indices.
        max_steps : int, optional
            Max iterations; 0 means until convergence.
        return_trajectory : bool, optional
            If True, also return the full trajectory (T, N).

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            Final state, or (final_state, trajectory) if return_trajectory is True.
        '''
        if len(input_indices) != len(input_pattern):
            raise ValueError(
                "input_indices and input_pattern must have the same length."
            )

        net = self._create_network_for_p(p)
        N = self.num_nodes

        s_init = torch.zeros((N,), device=self.device, dtype=torch.float32)
        for idx, bit in zip(input_indices, input_pattern):
            s_init[idx] = float(bit)

        return net.run_cascade(
            s_init,
            input_indices=input_indices,
            max_steps=max_steps,
            return_trajectory=return_trajectory,
        )

    def sample_cascade_sizes_for_p(
        self,
        p: float,
        num_realizations: int,
        k: int,
        max_steps: int = 0,
    ) -> List[int]:
        '''
        Sample cascade sizes for a single edge probability p.

        Parameters
        ----------
        p : float
            Connection probability for G(N, p).
        num_realizations : int
            Number of independent networks / cascades to run.
        k : int
            Number of input nodes, taken as [0, 1, ..., k-1].
        max_steps : int, optional
            Max iterations per cascade; 0 means until convergence.

        Returns
        -------
        list of int
            List of cascade sizes (one per realization).
        '''
        input_indices = list(range(k))
        cascade_sizes: List[int] = []

        for _ in range(num_realizations):
            pattern = torch.randint(
                low=0,
                high=2,
                size=(k,),
                device=self.device,
                dtype=torch.int64,
            )
            s_final = self._run_cascade_for_p(
                p=p,
                input_indices=input_indices,
                input_pattern=pattern.tolist(),
                max_steps=max_steps,
            )
            cascade_size = int((s_final > 0.5).sum().item())
            cascade_sizes.append(cascade_size)

        return cascade_sizes

    def monte_carlo_cascades_over_p(
        self,
        p_list: List[float],
        num_realizations: int,
        k: int,
        max_steps: int = 0,
        log_trajectories: bool = False,
        log_dir: Optional[str] = None,
    ) -> Dict[float, Dict[str, float]]:
        '''
        Monte Carlo cascades vs. edge probability p, returning mean/std cascade size per p.

        Optionally logs full cascade trajectories and network parameters
        for later transfer entropy / entropy analysis, and writes per-z CSVs.

        Parameters
        ----------
        p_list : list of float
            Connection probabilities p to test.
        num_realizations : int
            Number of independent networks per p.
        k : int
            Number of input nodes, taken as [0, 1, ..., k-1].
        max_steps : int, optional
            Max iterations per cascade; 0 means until convergence.
        log_trajectories : bool, optional
            If True, save adjacency, thresholds, trajectory, and metadata
            for each realization to disk. Default is False.
        log_dir : str or None, optional
            Root directory for logging when log_trajectories is True.

        Returns
        -------
        dict
            Mapping p -> {"mean_cascade_size", "std_cascade_size", "z"}.
        '''
        if log_trajectories and log_dir is None:
            raise ValueError("log_dir must be provided when log_trajectories=True.")

        results: Dict[float, Dict[str, float]] = {}
        input_indices = list(range(k))

        for p in p_list:
            cascade_sizes: List[int] = []
            z = self._p_to_z(p)

            # Directory for this z
            p_root_dir = None
            summary_csv = None
            if log_trajectories:
                z_str = f"z_{z:.3f}"
                p_root_dir = os.path.join(log_dir, z_str)
                os.makedirs(p_root_dir, exist_ok=True)
                summary_csv = os.path.join(p_root_dir, "summary.csv")

            # Run realizations
            for run_idx in range(num_realizations):
                net = self._create_network_for_p(p)

                pattern = torch.randint(
                    low=0,
                    high=2,
                    size=(k,),
                    device=self.device,
                    dtype=torch.int64,
                )

                s_init = torch.zeros(
                    (self.num_nodes,),
                    device=self.device,
                    dtype=torch.float32,
                )
                s_init[input_indices] = pattern.to(torch.float32)

                if log_trajectories:
                    final_state, trajectory = net.run_cascade(
                        s_init,
                        input_indices=input_indices,
                        max_steps=max_steps,
                        return_trajectory=True,
                    )
                else:
                    final_state = net.run_cascade(
                        s_init,
                        input_indices=input_indices,
                        max_steps=max_steps,
                        return_trajectory=False,
                    )
                    trajectory = None

                cascade_size = int((final_state > 0.5).sum().item())
                cascade_sizes.append(cascade_size)

                if log_trajectories:
                    run_dir = os.path.join(p_root_dir, f"run_{run_idx:04d}")
                    os.makedirs(run_dir, exist_ok=True)

                    torch.save(net.adjacency, os.path.join(run_dir, "adjacency.pt"))
                    torch.save(net.thresholds, os.path.join(run_dir, "thresholds.pt"))
                    torch.save(trajectory, os.path.join(run_dir, "trajectory.pt"))

                    # Per-run CSV row
                    write_csv_row(
                        summary_csv,
                        header=[
                            "run_idx", "p", "z",
                            "cascade_size", "pattern", "num_steps",
                        ],
                        row=[
                            run_idx,
                            float(p),
                            float(z),
                            cascade_size,
                            pattern.tolist(),
                            int(trajectory.shape[0]) if trajectory is not None else None,
                        ],
                    )

            # Summary statistics
            sizes_tensor = torch.tensor(cascade_sizes, dtype=torch.float32)
            mean_size = float(sizes_tensor.mean().item())
            std_size = float(sizes_tensor.std(unbiased=False).item())

            results[float(p)] = {
                "mean_cascade_size": mean_size,
                "std_cascade_size": std_size,
                "z": float(z),
            }

            # One SUMMARY row per p
            if log_trajectories:
                write_csv_row(
                    summary_csv,
                    header=[
                        "run_idx", "p", "z",
                        "cascade_size", "pattern", "num_steps",
                    ],
                    row=[
                        "SUMMARY",
                        float(p),
                        float(z),
                        mean_size,
                        None,
                        None,
                    ],
                )

        return results

    def run_single_cascade_p(
        self,
        p: float,
        input_indices: List[int],
        input_pattern: List[int],
        max_steps: int = 0,
    ) -> Dict[str, object]:
        '''
        Run a single cascade for a given edge probability p.

        Parameters
        ----------
        p : float
            Connection probability p.
        input_indices : list of int
            Indices of input nodes.
        input_pattern : list of int
            Binary input pattern aligned with input_indices.
        max_steps : int, optional
            Max iterations; 0 means until convergence.

        Returns
        -------
        dict
            Summary with final state and cascade size.
        '''
        s_final = self._run_cascade_for_p(
            p=p,
            input_indices=input_indices,
            input_pattern=input_pattern,
            max_steps=max_steps,
        )
        cascade_size = int((s_final > 0.5).sum().item())
        z = self._p_to_z(p)

        return {
            "p": float(p),
            "z": float(z),
            "final_state": s_final.detach().cpu().to(torch.int64).tolist(),
            "cascade_size": cascade_size,
            "input_indices": list(map(int, input_indices)),
            "input_pattern": list(map(int, input_pattern)),
        }

    # ----- Thin z-based wrappers (all implemented via p) -----

    def run_single_cascade_z(
        self,
        z: float,
        input_indices: List[int],
        input_pattern: List[int],
        max_steps: int = 0,
    ) -> Dict[str, object]:
        p = self._z_to_p(z)
        result = self.run_single_cascade_p(
            p=p,
            input_indices=input_indices,
            input_pattern=input_pattern,
            max_steps=max_steps,
        )
        result["z"] = float(z)
        return result

    def sample_cascade_sizes_for_z(
        self,
        z: float,
        num_realizations: int,
        k: int,
        max_steps: int = 0,
    ) -> List[int]:
        p = self._z_to_p(z)
        return self.sample_cascade_sizes_for_p(
            p=p,
            num_realizations=num_realizations,
            k=k,
            max_steps=max_steps,
        )

    def monte_carlo_cascades_over_z(
        self,
        z_list: List[float],
        num_realizations: int,
        k: int,
        max_steps: int = 0,
    ) -> Dict[float, Dict[str, float]]:
        results: Dict[float, Dict[str, float]] = {}

        for z in z_list:
            p = self._z_to_p(z)
            cascade_sizes = self.sample_cascade_sizes_for_p(
                p=p,
                num_realizations=num_realizations,
                k=k,
                max_steps=max_steps,
            )

            sizes_tensor = torch.tensor(cascade_sizes, dtype=torch.float32)
            mean_size = float(sizes_tensor.mean().item())
            std_size = float(sizes_tensor.std(unbiased=False).item())

            results[float(z)] = {
                "mean_cascade_size": mean_size,
                "std_cascade_size": std_size,
                "p": float(p),
            }

        return results

    def te_z_scan_with_logging(
        self,
        z_min: float = 0.1,
        z_max: float = 2.0,
        z_step: float = 0.1,
        num_realizations: int = 100,
        k: int = 2,
        max_steps: int = 0,
        log_dir: str = "ltm_logs_zscan",
    ) -> Dict[float, Dict[str, float]]:
        '''
        Convenience wrapper: TE/entropy-ready z-scan with trajectory logging.

        Parameters
        ----------
        z_min, z_max : float
            Range of z values (inclusive of z_min, best-effort z_max).
        z_step : float
            Step size in z.
        num_realizations : int
            Number of cascades per z.
        k : int
            Number of inputs [0, ..., k-1].
        max_steps : int
            Max cascade steps; 0 = until convergence.
        log_dir : str
            Root directory to store z_* folders and summary.csv files.

        Returns
        -------
        dict
            Same structure as monte_carlo_cascades_over_p, keyed by p.
        '''
        os.makedirs(log_dir, exist_ok=True)

        # Build z list
        z_values: List[float] = []
        z = z_min
        while z <= z_max + 1e-12:  # small epsilon to avoid float issues
            z_values.append(z)
            z += z_step

        # Convert to p list
        p_list = [self._z_to_p(z_val) for z_val in z_values]

        # Delegate to the core MC routine with logging enabled
        return self.monte_carlo_cascades_over_p(
            p_list=p_list,
            num_realizations=num_realizations,
            k=k,
            max_steps=max_steps,
            log_trajectories=True,
            log_dir=log_dir,
        )




# ----- CLI utilities -----

def _parse_float_list(list_str: str) -> List[float]:
    '''
    Parse a comma-separated list of floats.

    Parameters
    ----------
    list_str : str
        String such as "0.5,1.0,1.5".

    Returns
    -------
    list of float
        Parsed float values.
    '''
    return [float(x.strip()) for x in list_str.split(",") if x.strip()]


def _parse_input_pattern(pattern_str: str) -> List[int]:
    '''
    Parse a comma-separated binary pattern, e.g. "0,1,0,1".

    Parameters
    ----------
    pattern_str : str
        Comma-separated bits.

    Returns
    -------
    list of int
        Parsed bits in {0, 1}.
    '''
    bits = [int(x.strip()) for x in pattern_str.split(",") if x.strip()]
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"Invalid bit in input pattern: {b}")
    return bits


def main():
    '''
    CLI for running LTM cascade experiments (non-Boolean).

    Modes
    -----
    - Single cascade mode:  --single
        * Defaults:
            p = 0.1   (if no z or p given)
            k = 2
            input_pattern = "1,1"

    - Monte Carlo mode (default if --single not given)
        * Defaults:
            p_list = "0.01,0.05,0.1"
            k = 2
            num_realizations = 10
    '''
    parser = argparse.ArgumentParser(
        description="Monte Carlo LTM cascades on ER networks (non-Boolean)."
    )

    # Global configuration
    parser.add_argument("--N", type=int, default=100,
                        help="Number of nodes N (default: 100).")
    parser.add_argument("--phi", type=float, default=0.1,
                        help="Homogeneous threshold phi (default: 0.1).")
    parser.add_argument("--directed", action="store_true",
                        help="Use directed ER graphs (default: undirected).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Computation device (default: auto).")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max cascade steps; 0 = until convergence (default: 0).")

    # Random seed (for reproducibility)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for torch. "
                             "If not provided, derived from current time.")

    # Mode selection
    parser.add_argument("--single", action="store_true",
                        help="Run a single cascade instead of Monte Carlo.")

    # Single-cascade args (z / p and input pattern)
    parser.add_argument("--z", type=float,
                        help="Mean degree z for single run.")
    parser.add_argument("--p", type=float,
                        help="Connection probability p for single run "
                             "(alternative to z).")
    parser.add_argument("--k", type=int,
                        help="Number of inputs (k). Default depends on mode.")
    parser.add_argument("--input-pattern", type=str,
                        help="Comma-separated input pattern, e.g. '0,1,0' "
                             "(default in single mode: '1,1').")

    # Monte Carlo specific (also optional; defaults provided if missing)
    parser.add_argument("--z-list", type=str,
                        help="Comma-separated z values for MC, e.g. '0.5,1.0,1.5'.")
    parser.add_argument("--p-list", type=str,
                        help="Comma-separated p values for MC, e.g. '0.01,0.05,0.1' "
                             "(default: '0.01,0.05,0.1').")
    parser.add_argument("--num-realizations", type=int, default=10,
                        help="Number of realizations per z/p in MC (default: 10).")


    parser.add_argument("--output", type=str, default=None,
                    help="Output file (default: ./data/results_<timestamp>.csv).")

    
    parser.add_argument("--output-format", type=str, default="csv",
                        choices=["json", "csv"],
                        help="Format for saving results when --output is given "
                             "(default: csv).")

    parser.add_argument("--print", action="store_true",
                    help="Print results to stdout (default: False).")


    # TE / entropy z-scan mode (trajectory logging)
    parser.add_argument("--te-zscan", action="store_true",
                        help="Run TE/entropy-ready z-scan with trajectory logging.")
    parser.add_argument("--z-min", type=float, default=0.1,
                        help="Minimum z for --te-zscan (default: 0.1).")
    parser.add_argument("--z-max", type=float, default=2.0,
                        help="Maximum z for --te-zscan (default: 2.0).")
    parser.add_argument("--z-step", type=float, default=0.1,
                        help="Step in z for --te-zscan (default: 0.1).")
    parser.add_argument("--log-dir", type=str, default="ltm_logs_zscan",
                        help="Root directory for trajectory logs when using --te-zscan "
                             "(default: ltm_logs_zscan).")
    
    args = parser.parse_args()

    # ---------- Random seed ----------
    import time
    if args.seed is None:
        # use highest-resolution time available, then clamp into 32-bit range
        seed = time.time_ns() & 0xFFFFFFFF
    else:
        seed = int(args.seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Resolve device
    if args.device == "auto":
        device = get_default_device()
    else:
        device = torch.device(args.device)

    mc = LTMCascadeMonteCarlo(
        num_nodes=args.N,
        phi_value=args.phi,
        directed=args.directed,
        device=device,
    )

    # ---------- Single-cascade mode ----------
    if args.single:
        # Determine p (prefer explicit p, otherwise convert from z, otherwise default).
        if args.p is not None:
            p = args.p
        elif args.z is not None:
            if args.N <= 1:
                p = 0.0
            else:
                p = float(args.z) / float(args.N - 1)
        else:
            p = 0.1  # default connection probability

        k = args.k if args.k is not None else 2
        pattern_str = args.input_pattern if args.input_pattern is not None else "1,1"

        input_indices = list(range(k))
        pattern = _parse_input_pattern(pattern_str)
        if len(pattern) != k:
            raise ValueError(
                f"Input pattern length ({len(pattern)}) must match k={k}."
            )

        result = mc.run_single_cascade_p(
            p=p,
            input_indices=input_indices,
            input_pattern=pattern,
            max_steps=args.max_steps,
        )


    # ---------- TE z-scan mode (with logging) ----------
    elif args.te_zscan:
        k = args.k if args.k is not None else 2

        result = mc.te_z_scan_with_logging(
            z_min=args.z_min,
            z_max=args.z_max,
            z_step=args.z_step,
            num_realizations=args.num_realizations,
            k=k,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
        
    # ---------- Monte Carlo mode (default) ----------
    else:
        k = args.k if args.k is not None else 2

        # Determine list of p values to use.
        if args.p_list is not None:
            p_list = _parse_float_list(args.p_list)
        elif args.z_list is not None:
            z_list = _parse_float_list(args.z_list)
            if args.N <= 1:
                p_list = [0.0 for _ in z_list]
            else:
                p_list = [float(z) / float(args.N - 1) for z in z_list]
        else:
            default_p_list_str = "0.01,0.05,0.1"
            p_list = _parse_float_list(default_p_list_str)

        result = mc.monte_carlo_cascades_over_p(
            p_list=p_list,
            num_realizations=args.num_realizations,
            k=k,
            max_steps=args.max_steps,
        )
    
    # ---------- Output ----------
    # Choose output path (default to ./data/results_<timestamp>.csv)
    output_path = args.output if args.output is not None else default_output_path()

    # Optional pretty-print to stdout
    if args.print:
        print(json.dumps(result, indent=2))

    # Write to file in requested format
    if args.output_format == "json":
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    else:  # "csv"
        save_result_as_csv(output_path, result)


if __name__ == "__main__":
    main()
