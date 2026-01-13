#!/usr/bin/env python

# ============================
# file: ltm_boolean_mc.py
# ============================

"""
Boolean function / functional-percolation experiments for LTM cascades.

- Uses LTMCascadeNetwork from ltm_core.py
- All dynamics and Monte Carlo are implemented in terms of edge probability p.
- Mean degree z is handled only by converting z <-> p when needed (e.g. in the CLI).
"""

import argparse
import json
from typing import Dict, List, Optional

import torch

from ltm_core import LTMCascadeNetwork, get_default_device, save_result_as_csv


# ============================================================
# Boolean utilities (only here, not in ltm_core)
# ============================================================

def run_truth_table_for_network(
    network: LTMCascadeNetwork,
    input_indices: List[int],
    max_steps: int = 0,
) -> torch.Tensor:
    '''
    Run cascades for all input patterns and collect final states.

    Parameters
    ----------
    network : LTMCascadeNetwork
        Initialized LTM network with adjacency, degrees, and thresholds set.
    input_indices : list of int
        Indices of input nodes; let k = len(input_indices).
    max_steps : int, optional
        Maximum iterations per cascade; 0 means until convergence.

    Returns
    -------
    torch.Tensor
        Final states of shape (2^k, N) on the network device.
    '''
    N = network.num_nodes
    k = len(input_indices)
    if k <= 0:
        raise ValueError("At least one input index must be provided.")

    device = network.device
    dtype = torch.float32

    input_indices_tensor = torch.as_tensor(
        input_indices,
        device=device,
        dtype=torch.long,
    )

    num_patterns = 2 ** k
    final_states = torch.zeros(
        (num_patterns, N),
        device=device,
        dtype=dtype,
    )

    for pattern_idx in range(num_patterns):
        bits = []
        for j in range(k):
            shift = k - 1 - j
            bit = (pattern_idx >> shift) & 1
            bits.append(bit)

        bits_tensor = torch.tensor(bits, device=device, dtype=dtype)

        s_init = torch.zeros((N,), device=device, dtype=dtype)
        s_init[input_indices_tensor] = bits_tensor

        s_final = network.run_cascade(
            initial_state=s_init,
            input_indices=input_indices,
            max_steps=max_steps,
        )
        final_states[pattern_idx] = s_final

    return final_states
    

def compute_boolean_function_ids(truth_table_states: torch.Tensor) -> torch.Tensor:
    '''
    Compute integer Boolean function IDs from truth table states.

    Parameters
    ----------
    truth_table_states : torch.Tensor
        Tensor of shape (num_patterns, num_nodes) with 0/1 entries.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_nodes,) with integer function IDs in
        [0, 2**num_patterns - 1].
    '''
    num_patterns, num_nodes = truth_table_states.shape
    device = truth_table_states.device

    powers = 2 ** torch.arange(
        num_patterns - 1,
        -1,
        -1,
        device=device,
        dtype=torch.float32,
    )

    states_float = truth_table_states.to(dtype=torch.float32)
    function_ids_float = torch.matmul(states_float.T, powers)
    function_ids = function_ids_float.to(dtype=torch.long)

    return function_ids



def compute_variable_dependence_dtc(truth_table_states: torch.Tensor, k: int) -> torch.Tensor:
    '''
    Compute deterministic decision-tree complexity as the number of
    input variables each node's Boolean function actually depends on.

    Parameters
    ----------
    truth_table_states : torch.Tensor
        Tensor of shape (num_patterns, num_nodes) with 0/1 entries, where
        num_patterns = 2**k and the rows are ordered as in
        run_truth_table_for_network (patterns correspond to integers
        0, 1, ..., 2**k - 1 in binary over k bits, most significant bit
        = input 0).
    k : int
        Number of input variables.

    Returns
    -------
    torch.Tensor
        1D tensor of shape (num_nodes,) with integer DTC values, equal
        to the number of input variables that actually influence each node.
    '''
    num_patterns, num_nodes = truth_table_states.shape
    if num_patterns != 2 ** k:
        raise ValueError(
            f"Expected 2**k = {2**k} patterns, got {num_patterns}."
        )

    device = truth_table_states.device
    # Indices 0 .. 2**k - 1
    idx = torch.arange(num_patterns, device=device)

    # relevance[j, i] = True if node i depends on variable j
    relevance_flags = []

    for j in range(k):
        # Bit position used for input j in run_truth_table_for_network:
        # shift = k - 1 - j, so we flip that bit.
        bitmask = 1 << (k - 1 - j)

        # For each pattern index, get the index with bit j flipped.
        paired_idx = idx ^ bitmask  # shape: (num_patterns,)

        # Compare outputs for x and x with bit j flipped.
        # diff has shape (num_patterns, num_nodes)
        diff = truth_table_states != truth_table_states[paired_idx]

        # Variable j is relevant for node i if any pair differs.
        relevant_j = diff.any(dim=0)  # shape: (num_nodes,)
        relevance_flags.append(relevant_j)

    # Stack to shape (k, num_nodes), then sum over variables.
    relevance_tensor = torch.stack(relevance_flags, dim=0)
    dtc = relevance_tensor.sum(dim=0).to(torch.int64)

    return dtc
    
    
# ============================================================
# Boolean MC explorer (internally p-only)
# ============================================================

class LTMBooleanExplorer:
    '''
    Boolean function mapping and Monte Carlo experiments for LTM cascades.

    All internal logic is implemented in terms of edge probability p.
    Mean degree z is handled only via z <-> p conversion when requested.
    '''

    def __init__(
        self,
        num_nodes: int,
        phi_value: float = 0.1,
        directed: bool = False,
        device: Optional[torch.device] = None,
    ):
        '''
        Initialize the Boolean explorer.

        Parameters
        ----------
        num_nodes : int
            Number of nodes N.
        phi_value : float, optional
            Homogeneous threshold. Default 0.1.
        directed : bool, optional
            If True, use directed ER graphs. Default False.
        device : torch.device or None, optional
            Computation device. If None, choose default.
        '''
        self.num_nodes = int(num_nodes)
        self.phi_value = float(phi_value)
        self.directed = bool(directed)
        self.device = device if device is not None else get_default_device()

    # ----- z <-> p helpers (for callers / CLI) -----

    def z_to_p(self, z: float) -> float:
        '''
        Convert mean degree z to ER edge probability p = z / (N - 1).
        '''
        if self.num_nodes <= 1:
            return 0.0
        return float(z) / float(self.num_nodes - 1)

    def p_to_z(self, p: float) -> float:
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

    # ----- Single-graph Boolean mapping (p-only) -----

    def map_functions_single_graph(
        self,
        p: float,
        input_indices: List[int],
        max_steps: int = 0,
    ) -> Dict[str, object]:
        '''
        Map Boolean functions realized by cascades on a single ER graph (given p).

        Parameters
        ----------
        p : float
            Connection probability.
        input_indices : list of int
            Indices of input nodes.
        max_steps : int, optional
            Max iterations per cascade; 0 means until convergence.

        Returns
        -------
        dict
            Contains truth-table states and function IDs for each node.
        '''
        net = self._create_network_for_p(p)

        final_states = run_truth_table_for_network(
            network=net,
            input_indices=input_indices,
            max_steps=max_steps,
        )

        function_ids = compute_boolean_function_ids(
            truth_table_states=final_states,
        )

        z = self.p_to_z(p)

        return {
            "p": float(p),
            "z": float(z),
            "input_indices": list(map(int, input_indices)),
            "truth_table_states": final_states.detach().cpu().to(torch.int64).tolist(),
            "function_ids": function_ids.detach().cpu().tolist(),
        }

    # ----- Monte Carlo over p (only implementation) -----

    def monte_carlo_functions_over_p(
        self,
        p_list: List[float],
        num_realizations: int,
        k: int,
        max_steps: int = 0,
    ) -> Dict[float, Dict[str, object]]:
        '''
        Monte Carlo Boolean function mapping vs. p.

        Parameters
        ----------
        p_list : list of float
            p values to test.
        num_realizations : int
            Number of networks per p.
        k : int
            Number of input nodes; inputs = [0, ..., k-1].
        max_steps : int, optional
            Max iterations per cascade; 0 means until convergence.

        Returns
        -------
        dict
            Mapping p -> summary stats:
            - "z" (mean degree corresponding to p)
            - "max_function_id"
            - "num_distinct_functions"
            - "function_id_histogram"
        '''
        results: Dict[float, Dict[str, object]] = {}
        input_indices = list(range(k))

        for p in p_list:
            all_ids = []

            for _ in range(num_realizations):
                net = self._create_network_for_p(p)
                final_states = run_truth_table_for_network(
                    network=net,
                    input_indices=input_indices,
                    max_steps=max_steps,
                )
                function_ids = compute_boolean_function_ids(
                    truth_table_states=final_states,
                )
                all_ids.append(function_ids.detach().cpu())

            all_ids_tensor = torch.cat(all_ids, dim=0)
            unique_ids, counts = torch.unique(all_ids_tensor, return_counts=True)

            if unique_ids.numel() > 0:
                max_function_id = int(unique_ids.max().item())
            else:
                max_function_id = 0

            num_distinct = int(unique_ids.numel())
            histogram = {
                int(fid): int(cnt)
                for fid, cnt in zip(unique_ids.tolist(), counts.tolist())
            }

            z = self.p_to_z(p)

            results[float(p)] = {
                "z": float(z),
                "max_function_id": max_function_id,
                "num_distinct_functions": num_distinct,
                "function_id_histogram": histogram,
            }

        return results


# ============================================================
# CLI helpers
# ============================================================

def _parse_float_list(list_str: str) -> List[float]:
    '''
    Parse a comma-separated list of floats.
    '''
    return [float(x.strip()) for x in list_str.split(",") if x.strip()]


def main():
    '''
    CLI for Boolean / functional-percolation experiments.

    Modes
    -----
    - Single-graph mode:  --single
        * Requires either p or z, and k.
    - Monte Carlo mode (default if --single not given):
        * If --p-list and --k are provided: MC over p, output keyed by p.
        * If --z-list and --k are provided: convert z -> p, MC over p, output keyed by z.
        * Otherwise:
            - N = 1000 (or user override)
            - z in [0.1, 2.0] step 0.1
            - p(z) = z / (N - 1)
            - k = 2
            - num_realizations = 100
            - output keyed by z.
    '''
    parser = argparse.ArgumentParser(
        description="Boolean-function mapping for LTM cascades on ER networks."
    )

    # Global configuration
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of nodes N (default: 1000).")
    parser.add_argument("--phi", type=float, default=0.1,
                        help="Homogeneous threshold phi (default: 0.1).")
    parser.add_argument("--directed", action="store_true",
                        help="Use directed ER graphs (default: undirected).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Computation device (default: auto).")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max cascade steps; 0 = until convergence (default: 0).")

    parser.add_argument("--single", action="store_true",
                        help="Run Boolean mapping on a single graph instead of MC.")

    # Single-graph args
    parser.add_argument("--p", type=float,
                        help="Connection probability p for single run.")
    parser.add_argument("--z", type=float,
                        help="Mean degree z for single run (alternative to p).")
    parser.add_argument("--k", type=int,
                        help="Number of inputs (k).")

    # MC args
    parser.add_argument("--p-list", type=str,
                        help="Comma-separated p values for MC, e.g. '0.01,0.05,0.1'.")
    parser.add_argument("--z-list", type=str,
                        help="Comma-separated z values for MC, e.g. '0.5,1.0,1.5'.")
    parser.add_argument("--num-realizations", type=int, default=100,
                        help="Number of networks per p/z for MC (default: 100).")

    parser.add_argument("--output", type=str, default="",
                        help="Optional path to save results.")

    parser.add_argument("--output-format", type=str, default="csv",
                        choices=["json", "csv"],
                        help="Format for saving results when --output is given "
                             "(default: csv).")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = get_default_device()
    else:
        device = torch.device(args.device)

    explorer = LTMBooleanExplorer(
        num_nodes=args.N,
        phi_value=args.phi,
        directed=args.directed,
        device=device,
    )

    # -------- Single-graph mode --------
    if args.single:
        if args.k is None:
            raise ValueError("--k is required for --single.")

        input_indices = list(range(args.k))

        # Determine p (possibly from z).
        if args.p is not None:
            p = args.p
        elif args.z is not None:
            p = explorer.z_to_p(args.z)
        else:
            raise ValueError("Either --p or --z is required for --single.")

        result = explorer.map_functions_single_graph(
            p=p,
            input_indices=input_indices,
            max_steps=args.max_steps,
        )

    # -------- Monte Carlo mode (default) --------
    else:
        # Case 1: explicit p-list -> MC over p, output keyed by p.
        if args.p_list is not None:
            if args.k is None:
                raise ValueError("--k is required when using --p-list.")
            p_list = _parse_float_list(args.p_list)
            k = args.k
            result = explorer.monte_carlo_functions_over_p(
                p_list=p_list,
                num_realizations=args.num_realizations,
                k=k,
                max_steps=args.max_steps,
            )

        # Case 2: explicit z-list -> convert z -> p, MC over p, re-key by z.
        elif args.z_list is not None:
            if args.k is None:
                raise ValueError("--k is required when using --z-list.")
            z_list = _parse_float_list(args.z_list)
            p_list = [explorer.z_to_p(z) for z in z_list]
            k = args.k

            stats_by_p = explorer.monte_carlo_functions_over_p(
                p_list=p_list,
                num_realizations=args.num_realizations,
                k=k,
                max_steps=args.max_steps,
            )

            # Re-key by z for output
            result: Dict[float, Dict[str, object]] = {}
            for z, p in zip(z_list, p_list):
                stats = stats_by_p[float(p)]
                result[float(z)] = {
                    "p": float(p),
                    "max_function_id": stats["max_function_id"],
                    "num_distinct_functions": stats["num_distinct_functions"],
                    "function_id_histogram": stats["function_id_histogram"],
                }

        # Case 3: default z-range -> build z_list, convert to p, MC over p, re-key by z.
        else:
            N = args.N
            z_min, z_max, z_step = 0.1, 2.0, 0.1
            z_list: List[float] = []
            z = z_min
            while z <= z_max + 1e-9:
                z_list.append(z)
                z += z_step

            p_list = [z / (N - 1) if N > 1 else 0.0 for z in z_list]
            k = 2  # default k for functional-percolation scans

            stats_by_p = explorer.monte_carlo_functions_over_p(
                p_list=p_list,
                num_realizations=args.num_realizations,
                k=k,
                max_steps=args.max_steps,
            )

            result = {}
            for z, p in zip(z_list, p_list):
                stats = stats_by_p[float(p)]
                result[float(z)] = {
                    "p": float(p),
                    "max_function_id": stats["max_function_id"],
                    "num_distinct_functions": stats["num_distinct_functions"],
                    "function_id_histogram": stats["function_id_histogram"],
                }

    # print(json.dumps(result, indent=2))

    if args.output:
        if args.output_format == "json":
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        else:  # csv
            save_result_as_csv(args.output, result)

if __name__ == "__main__":
    main()
