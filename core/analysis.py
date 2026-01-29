"""
====================================================================================================
Existence of phase-cohesive solutions of the lossless power flow equations - Core Analysis Routines
====================================================================================================
Analytical bounds for region of trust around linear solution and phase cohesiveness.

This module implements all quantities derived from the model, including:
  - computation of error bounds (kappa, chi) defining the region of trust around the linear solution,
  - evaluation of phase-cohesiveness,
  - finding these quantities when there is voltage uncertainty.

The functions here use solutions produced by the model routine and encode
the theoretical results used for validation and comparison.
"""

import math
import numpy as np
import networkx as nx
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, BR_STATUS
from typing import List
from collections import Counter

from .model import sample_balanced_p, get_orth_basis_hyperplane, get_convergent_init_conds, get_convergent_sol_given_p

# --- Functions for detecting bridges ---

def find_bridges(N: int, f: np.ndarray, t: np.ndarray):
    """
    Constructs an undirected graph from edge lists and computes its bridges.
    Args:
        N (int): Number of nodes (used if constructing graph).
        f (np.ndarray): List of "from" nodes for edges, length Ne.
        t (np.ndarray): List of "to" nodes for edges, length Ne.

    Returns:
        bridge_edges (list[tuple[int, int]]): List of bridge edges as (u, v) node pairs.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(zip(f, t))

    if not nx.is_connected(graph):
        print("Warning: Graph is not connected.")

    bridges = nx.bridges(graph) 
    return list(bridges)


def find_bridge_indices_from_ft(N: int, f: np.ndarray, t: np.ndarray):
    """
    Bridge edge indices for graphs with possible parallel edges.

    An edge (u,v) is a bridge if:
      (i)  (u,v) is a bridge in the underlying simple graph, and
      (ii) there is exactly one edge between u and v (multiplicity==1).

    Args:
        N (int): Number of nodes.
        f, t (np.ndarray): From/to arrays (length Ne).

    Returns:
        list[int]: Edge indices that are bridges.
    """
    f = np.asarray(f, dtype=int)
    t = np.asarray(t, dtype=int)

    # multiplicities of unordered pairs
    pairs = [tuple(sorted((uu, vv))) for uu, vv in zip(f, t)]
    mult = Counter(pairs)

    # simple graph for bridge structure
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(pairs)  # duplicates irrelevant in simple graph

    bridge_pairs = {tuple(sorted(e)) for e in nx.bridges(G)}

    bridge_indices = [
        idx for idx, p in enumerate(pairs)
        if (p in bridge_pairs) and (mult[p] == 1)
    ]
    return bridge_indices


# ===== Figure 3 =====

def gradient(psi):
    """Calculate the gradient of the objective function for bounding the error (Eq 31).
    
    Args:
        psi: Vector of phase differences in the linearized model.

    Returns:
        list[float]: Gradient components for each element of psi."""
    return [(np.arcsin(psi[a] - 2*math.floor((1+psi[a])/2)) + np.pi*math.floor((1+psi[a])/2)) for a in range(len(psi))]

def compute_kappa_chi(K_vec, E, psi_lin):
    """
    Calculate individual line errors for line loads.
    According to Eq 59: - chi[e] <= xi[e] - kappa[e] <= + chi[e]
    where xi[e] = f_dc[e] - f_ac[e], and chi[e] and xi[e] are defined in Eq 58.
    For the final result, we divide with K[e] to convert from flows to loads.
    
    Args:
        K_vec (np.ndarray): Vector of line couplings K[e] (shape (Ne,)).
        E (np.ndarray): Node-edge incidence matrix (shape (Nn, Ne)).
        psi_lin: Phase differences along edges in the linear approximation.

    Returns:
        kappa_over_K (np.ndarray): Line-wise kappa values normalized by K[e] (shape (Ne,)).
        chi_over_K (np.ndarray): Line-wise chi values normalized by K[e] (shape (Ne,)).
    """
    Ne = len(K_vec)
    K_matrix = np.diag(K_vec)

    L = E @ K_matrix @ E.T
    L_plus = np.linalg.pinv(L)
    Pi = np.eye(Ne) - K_matrix @ E.T @ L_plus @ E
    g = np.asarray(gradient(psi_lin), dtype=float)

    Kg = K_matrix @ g
    PiKg = Pi @ Kg             
    gPiKg = g @ PiKg              
    KPigPiKg = K_vec * np.diag(Pi) * gPiKg
    KPigPiKg[abs(KPigPiKg) < 1e-10] = 0 #numerically zero

    kappa_flow = 0.5 * PiKg 
    chi_flow   = 0.5 * np.sqrt(KPigPiKg)

    kappa_over_K    = kappa_flow / K_vec
    chi_over_K = chi_flow   / K_vec

    return kappa_over_K, chi_over_K


# ===== Figures 4, 5 =====

def get_max_loads(psi_lin: np.ndarray, psi_nonlin: np.ndarray, kappa_over_K: np.ndarray, chi_over_K: np.ndarray):
    """
    Computes maximum load and phase cohesiveness.

    This function returns:
      - the maximum absolute nonlinear phase difference |psi_nonlin|,
      - the maximum  phase-cohesiveness bound implied by the region of trust
           [psi_lin - kappa_over_K - chi_over_K, psi_lin - kappa_over_K + chi_over_K].

    Args:
        psi_lin (np.ndarray): Linear (DC) edge phase differences.
        psi_nonlin (np.ndarray): Nonlinear (AC) edge phase differences.
        kappa_over_K (np.ndarray), chi_over_K (np.ndarray): Error bound terms.

    Returns:
        max_load (float): Maximum absolute nonlinear phase difference, max_e |psi_nonlin,e|.
        max_PC (float): Maximum absolute phase-cohesiveness bound over all edges.
    """
    # Maximum difference in nonlinear solution for line loads 
    max_load = np.max(np.abs(psi_nonlin))

    # Maximum cohesiveness
    lower = psi_lin - kappa_over_K - chi_over_K
    upper = psi_lin - kappa_over_K + chi_over_K
    max_PC = np.max(np.abs(np.concatenate((lower, upper))))

    return max_load, max_PC

def compute_S1_S2(net, p: np.ndarray, E_matrix: np.ndarray, K_vec: np.ndarray, norepeat: bool = False):
    """
    Compute metrics S1 and S2 for a given balanced power injection vector.

    S1 is defined as the maximum absolute nonlinear phase difference
        S1 = max_e |psi_nonlin,e|.

    S2 is defined as the maximum phase-cohesiveness bound implied by the
    region of trust,
        [psi_lin - kappa_over_K - chi_over_K,
         psi_lin - kappa_over_K + chi_over_K].

    Args:
        net (pandapower object): Network, having run power flow.
        E_matrix (np.ndarray): Node–edge incidence matrix.
        K_vec (np.ndarray): Line coupling coefficients.
        norepeat (bool, optional): If True, solve once without perturbing
            initial conditions; otherwise allow multiple attempts at finding convergent solution.

    Returns:
        S1 (float): Maximum line load.
        S2 (float): Maximum phase-cohesiveness.
        Returns (np.nan, np.nan) if no convergent solution is found.
    """
    nn_resolve_tries = 1 if norepeat else 250

    psi_lin, psi_nonlin, _, _, _, _ = get_convergent_sol_given_p(net, p, nn_resolve_tries)

    if psi_nonlin is np.nan or np.isnan(psi_nonlin).any():
        return np.nan, np.nan

    kappa_over_K, chi_over_K = compute_kappa_chi(K_vec, E_matrix, psi_lin)
    S1, S2 = get_max_loads(psi_lin, psi_nonlin, kappa_over_K, chi_over_K)

    return S1, S2

def get_max_line_load_default_powinj(net, p_base: np.ndarray, pow_fac_list, E_matrix, K_vec):
    """
    Fig 4a: Compute S1 and S2 for scaled versions of a fixed base power injection.
    """
    all_S1s, all_S2s = [], []

    for pow_fac in pow_fac_list:
        p_vec = p_base * pow_fac
        S1, S2 = compute_S1_S2(net, p_vec, E_matrix, K_vec, norepeat=False)

        all_S1s.append(S1)
        all_S2s.append(S2)

    return all_S1s, all_S2s


def get_p_coeff_list(net, pow_fac_list, trial_num: int, min_success: int = 1, max_tries: int = 10000, verbose: bool = True):
    """
    Generate coefficient vectors for randomized balanced power injections
    using a given number of attempts.

    The function attempts at most trial_num calls to
    get_convergent_init_conds at the maximum loading level
    max(pow_fac_list) and collects all successful coefficient vectors
    for the power injection vector.
    Sampling stops early if min_success successful samples are obtained.

    This is useful in high-loading regimes where successful convergence
    tends to be rare.

    Args:
        net: pandapower network having run power flow.
        pow_fac_list (array-like): Power-factor scalings; the maximum value
            is used as the convergence target.
        trial_num (int): Maximum number of attempts.
        min_success (int, optional): Minimum number of successful coefficient
            vectors considered sufficient. Defaults to 1.
        norepeat (bool, optional): Passed to ``get_convergent_init_conds``.
            Defaults to False.
        max_tries (int, optional): Maximum internal solver attempts per trial.
            Defaults to 10000.
        verbose (bool, optional): If True, prints a warning when fewer than
            ``min_success`` successes are found. Defaults to True.

    Returns:
        list[np.ndarray]: List of successful coefficient vectors. Length is
            <= min(trial_num, number of successes).
    """
    target_pow_fac = float(np.max(pow_fac_list))
    p_coeff_list: List[np.ndarray] = []

    for attempt in range(trial_num):
        p_temp = get_convergent_init_conds(net, target_pow_fac, max_tries)

        if not np.isnan(p_temp[0]):
            p_coeff_list.append(p_temp)

            if len(p_coeff_list) >= min_success:
                break

    if verbose and len(p_coeff_list) < min_success:
        print(
            f"Warning: only {len(p_coeff_list)} successful samples found "
            f"(required {min_success}) after {trial_num} attempts."
        )

    return p_coeff_list

'''
#a previous version of the function that is replaced with something better.
def get_max_line_load_randomized_powinj(net, pow_fac_list, p_coeff_list, E_matrix, K_vec, basis_for_p_plane=None):
    """
    Fig 4b: Compute averaged S1 and S2 over randomized balanced injections.
    """
    if basis_for_p_plane is None:
        NN = len(net.bus)
        basis_for_p_plane = get_orth_basis_hyperplane(np.ones(NN), check=True)

    all_S1s, all_S2s = [], []

    for pow_fac in pow_fac_list:
        sel_S1s, sel_S2s = [], []

        for p_c in p_coeff_list:
            p_vec = sum(
                coeff_n * np.asarray(basis_for_p_plane[nn]) * pow_fac
                for nn, coeff_n in enumerate(p_c)
            )

            S1, S2 = compute_S1_S2(net, p_vec, E_matrix, K_vec, norepeat=False)
            sel_S1s.append(S1)
            sel_S2s.append(S2)

            n_nan = np.isnan(sel_S1s).sum()
            if n_nan > 0:
                print(pow_fac, "NaNs:", n_nan, "out of", len(sel_S1s))


        all_S1s.append(sel_S1s)
        all_S2s.append(sel_S2s)

    avg_S1s = [np.nanmean(S1s) for S1s in all_S1s]
    avg_S2s = [np.nanmean(S2s) for S2s in all_S2s]

    return avg_S1s, avg_S2s
'''

def get_max_line_load_randomized_powinj_fixedNsamples(net, pow_fac_list, p_coeff_list, E_matrix, K_vec,
    N_target: int, norepeat: bool = False):
    """
    Fig 4b:
    Compute averaged S1 and S2 over randomized balanced injections,
    using exactly N_target samples that converge for ALL pfs.

    Any sample that fails at any pf is discarded entirely.
    This is relevant because, for Case 30 in approx 1% of cases,
    some samples do not converge at lower pf than the maximal.
    """

    NN = E_matrix.shape[0]
    basis_for_p_plane = get_orth_basis_hyperplane(np.ones(NN), check=True)

    pow_fac_list = list(pow_fac_list)
    Neval = len(pow_fac_list)

    S1_vals = [[] for _ in pow_fac_list]
    S2_vals = [[] for _ in pow_fac_list]

    n_accepted = 0
    n_attempted = 0

    for p_c in p_coeff_list:
        if n_accepted >= N_target:
            break

        n_attempted += 1

        p_dir = sum(
            coeff_n * np.asarray(basis_for_p_plane[nn])
            for nn, coeff_n in enumerate(p_c)
        )

        S1_tmp = np.zeros(Neval)
        S2_tmp = np.zeros(Neval)

        failed = False

        for i, pf in enumerate(pow_fac_list):
            p_vec = pf * p_dir

            S1, S2 = compute_S1_S2(net, p_vec, E_matrix, K_vec, norepeat=norepeat)

            if np.isnan(S1) or np.isnan(S2):
                failed = True
                break

            S1_tmp[i] = S1
            S2_tmp[i] = S2

        if failed:
            continue

        # accept sample
        for i in range(Neval):
            S1_vals[i].append(S1_tmp[i])
            S2_vals[i].append(S2_tmp[i])

        n_accepted += 1

    if n_accepted < N_target:
        raise RuntimeError(
            f"Only {n_accepted} fully valid samples found "
            f"(target was {N_target})."
        )

    assert all(len(v) == N_target for v in S1_vals)
    assert all(len(v) == N_target for v in S2_vals)

    avg_S1s = [np.mean(vals) for vals in S1_vals]
    avg_S2s = [np.mean(vals) for vals in S2_vals]

    return avg_S1s, avg_S2s


def get_non_default_S1_S2_hist(net, powfac, trialnum, E_matrix, K_vec):
    """
    Fig 4c: Generate S1 and S2 samples for random balanced injections at fixed power factor.
    """
    NN = len(net.bus)
    basis_for_p_plane = get_orth_basis_hyperplane(np.ones(NN), check=True)

    all_S1, all_S2 = [], []

    for t_ind in range(1, trialnum + 1):
        p_vec = sample_balanced_p(NN, power_factor=powfac, basis_for_p_plane=basis_for_p_plane)

        if abs(np.sum(p_vec)) >= 1e-10:
            raise ValueError("p input not balanced")

        S1, S2 = compute_S1_S2(net, p_vec, E_matrix, K_vec, norepeat=True)

        all_S1.append(S1)
        all_S2.append(S2)

        if t_ind % 10 == 0:
            print(t_ind, end=" ")

    return all_S1, all_S2

def get_S1_S2_hist_pf1_target_success(net, N_success: int, E_matrix: np.ndarray, K_vec: np.ndarray,
                                      max_attempts: int = 200000, norepeat: bool = True, seed: int | None = None,
                                      progress_every: int = 2000):
    """
    Collect histogram samples (S1, S2) at power factor p_f = 1.0,
    keeping only converged trials.

    Sampling stops when N_success converged samples are collected or
    max_attempts attempts are reached.

    Args:
        net (pandapowerNet): Pandapower network, having run power flow.
        N_success (int): Number of converged samples to collect.
        E_matrix, K_vec: Node-edge incidence matrix and coupling coefficients passed through to compute_S1_S2.
        max_attempts (int, optional): Upper limit on number of attempts. Defaults to 200000.
        norepeat (bool, optional): Whether or not to try multiple attempts for solving power flow in case initial does not converge.
                                   Passed to compute_S1_S2 (True = 1 try). Defaults to True.
        seed (int | None, optional): Random seed for reproducibility.
        progress_every (int, optional): Print progress every this many attempts. Defaults to 2000.

    Returns:
        tuple:
            S1 (np.ndarray): Converged S1 values (length <= N_success).
            S2 (np.ndarray): Converged S2 values (length <= N_success).
            attempted (int): Number of attempted samples.
    """
    if seed is not None:
        np.random.seed(seed)

    NN = E_matrix.shape[0]
    basis_for_p_plane = get_orth_basis_hyperplane(np.ones(NN), check=True)

    S1_list, S2_list = [], []
    attempted = 0

    while len(S1_list) < N_success and attempted < max_attempts:
        attempted += 1

        p_vec = sample_balanced_p(NN, power_factor=1.0, basis_for_p_plane=basis_for_p_plane)

        if abs(np.sum(p_vec)) >= 1e-10:
            raise ValueError("p input not balanced")

        S1, S2 = compute_S1_S2(net, p_vec, E_matrix, K_vec, norepeat=norepeat)

        if not (np.isnan(S1) or np.isnan(S2)):
            S1_list.append(S1)
            S2_list.append(S2)

        #track progress: this can take days to complete for large samples.
        if progress_every and attempted % progress_every == 0:
            print(f"attempted={attempted}  converged={len(S1_list)}", end="\r")

    return np.asarray(S1_list, dtype=float), np.asarray(S2_list, dtype=float), attempted


# ===== Figures 6, 7 =====

def adjust_incidence_matrix_for_positive_flows(E: np.ndarray, flows: np.ndarray):
    """
    Flip incidence columns so that all flows are nonnegative (see text just before Eq 73).
    """
    E_adj = E.copy()
    neg = flows < 0
    E_adj[:, neg] *= -1.0
    return E_adj

def get_K_min_max_edges_pp(ppc_branch: np.ndarray, v_min: np.ndarray, v_max: np.ndarray):
    """Eq. 93, computed per physical branch from PPC branch data."""
    br = np.asarray(ppc_branch)

    status = br[:, BR_STATUS].real.astype(int)
    br = br[status > 0]

    f = br[:, F_BUS].real.astype(int)
    t = br[:, T_BUS].real.astype(int)

    x = br[:, BR_X].real.astype(float)
    if np.any(np.isclose(x, 0.0)):
        raise ValueError("Found branch with x almost 0; cannot form 1/x safely.")

    tau = br[:, TAP].real.astype(float)
    tau = np.where((tau == 0.0) | np.isnan(tau), 1.0, tau)

    base = 1.0 / (x * tau)

    Kmin = base * v_min[f] * v_min[t]
    Kmax = base * v_max[f] * v_max[t]
    return Kmin, Kmax

def compute_edge_metrics(E, K_min, K_max, f_lin, list_of_bridges, f, t):
    """
    Computes error bars around linear solution for the case of uncertain voltages: Eqs 77-79.

    Parameters:
    - E: Node-edge incidence matrix.
    - K_min, K_max: Minimum and maximum K for edges.
    - f_lin: Linearized power flows.
    - list_of_bridges: List of bridges.
    - f, t: Edge start and end node lists.

    Returns:
    - kappa_max: kappa_max values for each edge.
    - kappa_min: kappa_min values for each edge.
    - chi_max: chi_max values for each edge.
    """

    # Compute L_max and Pi_max projection matrix (Eq 75)
    L_max = E @ np.diag(K_max) @ E.T
    Pi_max = np.eye(E.shape[1]) - np.diag(K_max) @ E.T @ np.linalg.pinv(L_max) @ E

    # Compute g_min and g_max for each edge (Eq 73)
    g_min = np.arcsin(f_lin / K_max)
    g_max = np.arcsin(f_lin / K_min)

    # Allocate arrays
    edgenum = len(K_max)
    kappa_max = np.zeros(edgenum)
    kappa_min = np.zeros(edgenum)
    chi_max_sq = np.zeros(edgenum)
    chi_max_sq_temp = np.zeros(edgenum)

    for e in range(edgenum):
        # Find all f where Pi_max[e, f] > 0 or Pi_max[e, f] < 0
        positive_f = Pi_max[e, :] > 0
        negative_f = Pi_max[e, :] < 0

        # Compute kappa_max and kappa_min and chi_max_sq_temp
        kappa_max[e] = 0.5 * (
            np.sum(Pi_max[e, positive_f] * K_max[positive_f] * g_max[positive_f]) +
            np.sum(Pi_max[e, negative_f] * K_max[negative_f] * g_min[negative_f])
        )

        kappa_min[e] = 0.5 * (
            np.sum(Pi_max[e, positive_f] * K_max[positive_f] * g_min[positive_f]) +
            np.sum(Pi_max[e, negative_f] * K_max[negative_f] * g_max[negative_f])
        )

        chi_max_sq_temp[e] = (
            np.sum(Pi_max[e, positive_f] * K_max[positive_f] * g_max[e] * g_max[positive_f]) +
            np.sum(Pi_max[e, negative_f] * K_max[negative_f] * g_min[e] * g_min[negative_f])
        )

    # Sum over chi_max_sq_temp for final chi_max_sq computation
    chi_max_sq_total = np.sum(chi_max_sq_temp)

    for e in range(edgenum):
        # Set bridges to zero
        if (f[e], t[e]) in list_of_bridges or (t[e], f[e]) in list_of_bridges:
            kappa_max[e] = 0
            kappa_min[e] = 0
            chi_max_sq[e] = 0

        chi_max_sq[e] = (1 / 4) * K_max[e] * Pi_max[e, e] * chi_max_sq_total

    # Compute chi_max as the square root of chi_max_sq
    chi_max_sq[abs(chi_max_sq) < 1e-10] = 0
    chi_max = np.sqrt(chi_max_sq)

    return kappa_max, kappa_min, chi_max

def get_bus_masks_and_vmref(net):
    """
    Return boolean masks for PQ/PV/Slack buses and a vm_ref vector where PV+slack
    are set to their reference values and all others are 1.0.
    """
    Nn = len(net.bus)
    vm_ref = np.ones(Nn, dtype=float)

    is_slack = np.zeros(Nn, dtype=bool)
    is_pv    = np.zeros(Nn, dtype=bool)

    if "ext_grid" in net and not net.ext_grid.empty:
        slack_b = net.ext_grid["bus"].to_numpy(dtype=int)
        slack_vm = net.ext_grid["vm_pu"].to_numpy(dtype=float)
        is_slack[slack_b] = True
        vm_ref[slack_b] = slack_vm

    if "gen" in net and not net.gen.empty:
        pv_b = net.gen["bus"].to_numpy(dtype=int)
        pv_vm = net.gen["vm_pu"].to_numpy(dtype=float)
        is_pv[pv_b] = True
        vm_ref[pv_b] = pv_vm

    is_pq = ~(is_slack | is_pv)
    return is_pq, is_pv, is_slack, vm_ref

def compute_v_min(net, v_min_pq: float):
    """
    v_min: PQ buses set to v_min_pq; PV+slack fixed to reference values.
    """
    is_pq, _, _, vm_ref = get_bus_masks_and_vmref(net)
    v_min = vm_ref.copy()
    v_min[is_pq] = float(v_min_pq)
    return v_min

def compute_v_max_from_B(net, B: np.ndarray) -> np.ndarray:
    """
    v_max: PV+slack fixed to reference voltages; PQ buses set using Lemma-1
    formula and then unified to a single max across PQ buses.
    """
    is_pq, is_pv, is_slack, vm_ref = get_bus_masks_and_vmref(net)

    v_max = vm_ref.copy()
    pv_slack = np.where(is_pv | is_slack)[0]
    pq_buses = np.where(is_pq)[0]

    Bshunts = np.sum(B, axis=1)

    vals = []
    for j in pq_buses:
        Bjjshunt = Bshunts[j]
        num = np.sum(B[j, pv_slack] * v_max[pv_slack])
        denom = np.sum(B[j, pv_slack]) - Bjjshunt
        if denom != 0:
            vals.append(num / denom)

    if len(vals) == 0:
        # fallback
        return v_max

    max_v_max = float(np.max(vals))
    v_max[pq_buses] = max_v_max
    return v_max

def get_vm_profile_pv_fixed_others_one(net):
    """
    Construct a voltage-magnitude profile where:
      - all buses are initialized to 1.0 pu,
      - slack (ext_grid) buses are set to their reference vm_pu,
      - PV buses (net.gen) are set to their reference vm_pu.

    Args:
        net (pandapowerNet)

    Returns:
        np.ndarray: Voltage magnitudes per bus for use in K_e = B_ij v_i v_j.
    """
    Nn = len(net.bus)
    vm = np.ones(Nn, dtype=float)

    # Slack buses (ext_grid)
    if "ext_grid" in net and not net.ext_grid.empty and "vm_pu" in net.ext_grid:
        for _, row in net.ext_grid.iterrows():
            bus_idx = int(row["bus"])
            vm[bus_idx] = float(row["vm_pu"])

    # PV buses (generators)
    if "gen" in net and not net.gen.empty and "vm_pu" in net.gen:
        for _, row in net.gen.iterrows():
            bus_idx = int(row["bus"])
            vm[bus_idx] = float(row["vm_pu"])

    return vm


def get_Psi(flows_lin, kappa_min, kappa_max, chi_max, K_min):
    # Eq 94: max_e max( |f - kappa_min + chi|/Kmin, |f - kappa_max - chi|/Kmin )
    if np.any(K_min == 0):
        raise ValueError("K_min contains zero values, division error.")

    psi_lo = np.abs(flows_lin - kappa_min + chi_max) / K_min
    psi_hi = np.abs(flows_lin - kappa_max - chi_max) / K_min
    return float(np.max(np.maximum(psi_lo, psi_hi)))

def get_upper_bound_loop(all_v_mins, net, B, ppc_branch, E_adjusted, flows_lin, f, t):
    Nn = len(net.bus)
    list_of_bridges = find_bridges(Nn, f, t)
    v_max = compute_v_max_from_B(net, B)

    Psi_vals = np.empty(len(all_v_mins), dtype=float)

    for i, v_min_set in enumerate(all_v_mins):
        v_min = compute_v_min(net, v_min_set)

        K_min, K_max = get_K_min_max_edges_pp(ppc_branch, v_min, v_max)
        kappa_max, kappa_min, chi_max = compute_edge_metrics(
            E_adjusted, K_min, K_max, flows_lin, list_of_bridges, f, t
        )

        Psi_vals[i] = get_Psi(flows_lin, kappa_min, kappa_max, chi_max, K_min)

    with np.errstate(invalid="ignore"):
        upper_bound = np.arcsin(Psi_vals)  # will be nan where Psi_vals > 1, ignore error in output
    return upper_bound, Psi_vals

