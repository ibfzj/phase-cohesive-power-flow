"""
==================================================================================================
Existence of phase-cohesive solutions of the lossless power flow equations - Core Model Routines
==================================================================================================
Model definitions and numerical solvers for power-flow–based phase dynamics.

This module contains the core mathematical and numerical components of the model, including:
  - definition of the edge-wise objective function and its derivative,
  - construction of graph representations from pandapower networks,
  - linear (DC) and nonlinear (AC) power-flow solvers,
  - functions for sampling balanced power injections and initializing solvers.

"""

import numpy as np
import pandapower as pp
import pandapower.pypower.makeYbus as makeYbus
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, BR_STATUS

from scipy.optimize import root
from scipy.stats import ortho_group


# --- Functions related to objective function ---

def n_e(f_e, K_e):
    return np.floor(0.5 * (f_e / K_e + 1.0))

def dF(f_e, K_e):
    """The derivative."""
    f_e = np.asarray(f_e, dtype=float)
    K_e = float(K_e)

    n = n_e(f_e, K_e)
    f_s = f_e - 2.0 * n * K_e # Shift the flow to the interval [-K_e, K_e] 
    return np.arcsin(f_s / K_e) + np.pi * n # The derivative is the arcsin + a stepwise constant term to make it continuous

def F(f_e, K_e):
    """The objective function."""
    f_e = np.asarray(f_e, dtype=float)
    K_e = float(K_e)

    n = n_e(f_e, K_e)
    f_s = f_e - 2.0 * n * K_e # Shift the flow to the interval [-K_e, K_e] 
 
    asin_int = f_s * np.arcsin(f_s / K_e) + np.sqrt(K_e**2 - f_s**2) # The integral of the arcsin
    lin_term = np.pi * n * f_e # The integral of the term pi * n_e:
    int_const = -np.pi * (n**2) * K_e # A stepwise constant term that makes F continuous (arises from integration constants)

    return asin_int + lin_term + int_const

# --- Functions for building underlying graphs and calculating power flow ---

def construct_ppgraph_from_pandapower_and_run_pf(net):
    """
    Runs lossless power flow analysis on a given pandapower network and extracts relevant data.

    Parameters:
    net (pandapowerNet): The input power system network.

    Returns:
        - net (pandapowerNet): The updated pandapower network after power flow calculation.
        - vm_pu (np.ndarray): Bus voltage magnitudes (per unit), shape (N,).
        - thetas (np.ndarray): Bus voltage angles (radians), shape (N,).
        - p (np.ndarray): Active power injections (per unit), shape(N,).
        - q (np.ndarray): Reactive power injections (per unit), shape(N,).
        - B (np.ndarray): Nodal susceptance matrix, shape(N,N).
        - f (np.ndarray): Array of "from" buses in branches, shape(N,).
        - t (np.ndarray): Array of "to" buses in branches, shape(N,).
    """

    #Remove losses
    if "shunt" in net and not net.shunt.empty: # Set all shunt conductances to zero
        net.shunt["g_mw"] = 0.0 
    
    if "line" in net and not net.line.empty: # Remove line resistances
        net.line["r_ohm_per_km"] = 0.0 

    if "trafo" in net and not net.trafo.empty: # Remove transformer resistances
        net.trafo["r_pu"] = 0.0

    baseMVA = net.sn_mva  # System base power (MVA)

    # Run power flow
    try:
        pp.runpp(net, tolerance_mva=1e-6)
    except pp.pandapowerNetError as e:
        raise RuntimeError("Power flow calculation did not converge.") from e
    
    vm_pu = net.res_bus.vm_pu.to_numpy()  # Voltage magnitudes in per unit
    thetas = np.radians(net.res_bus.va_degree.to_numpy())  

    p = - net.res_bus.p_mw.to_numpy().copy()  # Active power injections in MW
    q = - net.res_bus.q_mvar.to_numpy().copy()  # Reactive power injections in MVar

    p /= baseMVA
    q /= baseMVA

    # Assign imbalance to slack bus
    imbalance = np.sum(p)
    if abs(imbalance) > 1e-10:
        slack_bus = int(net.ext_grid["bus"].iloc[0])  # first slack bus
        p[slack_bus] -= imbalance
        if p[slack_bus] <= 0:
            raise ValueError("Generator at slack bus must have positive power output")
    
    # Extract Ybus (nodal admittance matrix)
    Ybus, _, _ = makeYbus.makeYbus(net._ppc["baseMVA"], net._ppc["bus"], net._ppc["branch"])
    
    # Extract the imaginary part (susceptance matrix B)
    B_matrix = Ybus.imag 
    B = np.array(B_matrix.toarray())

    # Extract edges (branches)
    br = np.asarray(net._ppc["branch"])
    br = br[br[:, BR_STATUS].real.astype(int) > 0]
    f = br[:, F_BUS].real.astype(int)
    t = br[:, T_BUS].real.astype(int)

    return net, vm_pu, thetas, p, q, B, f, t


def build_E_and_K_from_ppc_branch(ppc_branch: np.ndarray, vm_pu: np.ndarray):
    """
    Build incidence matrix E and per-branch couplings K_vec from the pandapower
    branch matrix (ppc['branch']), preserving parallel branches.

    Assumptions:
      - No phase-shifting transformers (SHIFT ignored / assumed 0).
      - Series resistance has been removed.
      - Coupling uses the branch series reactance x and off-nominal tap ratio tau.

    For each in-service branch e=(f,t):
        tau_e = TAP if TAP != 0 else 1
        K_e   = (1 / (x_e * tau_e)) * v_f * v_t

    Args:
        ppc_branch (np.ndarray): pandapower branch matrix, shape (Ne, ncols).
        vm_pu (np.ndarray): Bus voltage magnitudes (pu), shape (Nn,).

    Returns:
        f (np.ndarray): From-bus indices for each in-service branch, shape (Ne_on,).
        t (np.ndarray): To-bus indices for each in-service branch, shape (Ne_on,).
        E (np.ndarray): Node-edge incidence matrix, shape (Nn, Ne_on).
        K_vec (np.ndarray): Per-branch coupling coefficients, shape (Ne_on,).
    """
    br = np.asarray(ppc_branch)

    # Keep only in-service branches
    status = br[:, BR_STATUS].real.astype(int)
    br = br[status > 0]

    f = br[:, F_BUS].real.astype(int)
    t = br[:, T_BUS].real.astype(int)

    x = br[:, BR_X].real.astype(float)
    if np.any(np.isclose(x, 0.0)):
        bad = np.where(np.isclose(x, 0.0))[0][:10]
        raise ValueError(f"Found branch reactance x=0 for indices {bad}; cannot form 1/x couplings safely.")

    # Off-nominal tap ratio: pandapower uses 0 to mean "no transformer"
    tau = br[:, TAP].real.astype(float)
    tau = np.where((tau == 0.0) | np.isnan(tau), 1.0, tau)

    Nn = len(vm_pu)
    Ne = len(f)

    # Incidence matrix
    E = np.zeros((Nn, Ne), dtype=float)
    cols = np.arange(Ne)
    E[f, cols] = + 1.0
    E[t, cols] = - 1.0

    # Per-branch coupling (preserves parallel edges if they exist)
    K_vec = (1.0 / (x * tau)) * vm_pu[f] * vm_pu[t]

    return f, t, E, K_vec

def get_loads(K_matrix, flows):
    """Calculate line loadings from given flows and diagonal coupling matrix.

    Args:
        K_matrix (np.ndarray): Diagonal matrix of line couplings (shape (Ne, Ne)).
        flows: Line flows corresponding to each edge (length Ne).

    Returns:
        list[float]: Load values per edge, defined as flows[e] / K_matrix[e, e].
    """
    return [flows[e]/K_matrix[e, e] for e in range(len(flows))]

## functions below are adapted from Philipp and Carsten's code with some changes:
def lin_power_flow(p: np.ndarray, K_mat: np.ndarray, E: np.ndarray):
    """Solve the linear power flow by using the node edge incidence
    matrix E and the diagonal matrix K that collects edge susceptances,
    for injections p.
    Args:
        p (np.ndarray): Power injections
        K_mat (np.ndarray): Susceptance matrix
        E (np.ndarray): Node-edge incidence matrix
        
    Returns:
        theta_lin, flows_lin: Linear phase angles, Power flows

    """
    
    L = E @ K_mat @ E.T
    theta_lin = np.linalg.pinv(L) @ p
    flows_lin = K_mat @ (E.T @ theta_lin)
    
    return theta_lin, flows_lin


def nonlin_power_flow(p: np.ndarray, K_mat: np.ndarray, E: np.ndarray,
                      initial_guess_theta: np.ndarray | None, tol: float = 1e-10, solver_method="hybr"):
    """Find the non-linear power flow solution given the power injections and capacities
    for a given method.

    Args:
        p (np.ndarray): Vector with power injections.
        K_mat (np.ndarray): Matrix that collects the transmission capacities on the diagonal.
        E (np.ndarray): Node-edge incidence matrix.
        initial_guess_theta (np.ndarray, optional): initial guess of the phase angles as a vector. Defaults to None.
        tol (float, optional): Tolerance of the solver. Defaults to 1e-10.
        solver_method (str, optional): Solver methods used by scipy. 
        Possible values: 'hybr' (default) or 'broyden1'.
        
    Returns:
        sol_theta (np.ndarray): Array with phase angles for each node. If 'np.nan' is returned,
        no solution was found by the solver.
    """
    
    def non_lin_prob(state):
        theta_arr = state
        power_flow = E @ K_mat @ np.sin(E.T @ theta_arr)
        diff_power = p - power_flow
        
        return diff_power
        
    if initial_guess_theta is None:
        initial_guess_theta = lin_power_flow(p, K_mat, E)[0]
    
    if solver_method in ['broyden1', 'hybr']:
        sol = root(non_lin_prob, initial_guess_theta, method=solver_method, tol=tol)
    
        if sol.success:
            return sol.x
        else:
            return np.nan
        
    else:
        raise NotImplementedError("Solver method '{}' not implemented!".format(solver_method))
    

def get_convergent_sol_given_p(net, p: np.ndarray, nn_resolve_tries = 250):
    """
    Finds convergent solution directly from pandapower network and given power injections.

    Args:
        net (pandapowerNet): Pandapower network (having run power flow).
        p (np.ndarray): Power injections at nodes (length Nn).
        nn_resolve_tries (int): how many times to retry solving with a small perturbation.

    Returns:
        linear and nonlinear solution in terms of loads (psi), angles (theta) and flows,
        psi_lin, psi_nonlin, thetas_lin, thetas_nonlin, flows_lin, flows_nonlin
    """
    ppc_branch = net._ppc["branch"]
    vm_pu = net.res_bus.vm_pu.to_numpy()
    _, _, E, K_vec = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu)
    K_matrix = np.diag(K_vec)

    count_not_converged = 0
    count_converged = 0

    perturb_factor = 1e-5

    # Linear solution
    thetas_lin, flows_lin = lin_power_flow(p, K_matrix, E)
    psi_lin = get_loads(K_matrix, flows_lin)

    # Nonlinear solution
    thetas_nonlin = nonlin_power_flow(p, K_matrix, E, initial_guess_theta=thetas_lin)

    # Retry solving with perturbation if no solution was found for nn_resolve_tries times.
    retry_count = 0
    while nn_resolve_tries is not None and np.isnan(thetas_nonlin).any() and retry_count < nn_resolve_tries:
        initialize_guess_theta = thetas_lin + np.random.random(len(thetas_lin)) * perturb_factor
        thetas_nonlin = nonlin_power_flow(p, K_matrix, E, initial_guess_theta=initialize_guess_theta)
        retry_count += 1

    if np.isnan(thetas_nonlin).any():
        count_not_converged += 1
        psi_nonlin = np.nan
        flows_nonlin = np.nan
    else:
        flows_nonlin = K_matrix @ np.sin(E.T @ thetas_nonlin)
        psi_nonlin = get_loads(K_matrix, flows_nonlin)
        count_converged += 1

    return psi_lin, psi_nonlin, thetas_lin, thetas_nonlin, flows_lin, flows_nonlin

# --- Functions related to finding valid alternative power injection vectors ---
# entirely copied from philipp and carsten
def get_orth_basis_hyperplane(normal_vector, check=True):
    """Finds n-1 vectors that are orthonormal to the normal vector of the hyperplane. These n-1 "basis" vectors represent coordinates on the hyperplane.

    Args:
        normal_vector (list): Normal vector of the hyperplane, for which a basis is calculated.
        check (bool, optional): Check wether the set of vectors, before gram schmidt, is linearly independent. Defaults to True.

    returns:
        basis (list): list of orthonormal basis vectors for hyperplane.
    """

    d_space = len(normal_vector)

    # normalize normal_vector if not normalized
    if np.linalg.norm(normal_vector) != 1:
        normal_vector = np.asarray(normal_vector) / np.linalg.norm(normal_vector)

    # Construct linear set:
    number_of_zeros = d_space - np.count_nonzero(normal_vector)
    if number_of_zeros == (d_space - 1) and normal_vector[-1] == 0:
        set_independent = np.asarray(ortho_group(d_space).rvs()[:, 1:])
    else:
        set_independent = np.vstack([np.identity(d_space - 1), np.zeros(d_space - 1)])

    # Check wether the set is linear independent
    if check is True:
        set_check = np.hstack([set_independent, [[n_i] for n_i in normal_vector]])
        assert not np.linalg.det(set_check) == 0.0, "Basis vectors not linearly independent!"
            
    # Gram-Schmidt
    basis = []
    basis.append(np.asarray(normal_vector))

    for n in range(d_space - 1):
        new_basis_vector = set_independent[:, n] * 1
        for k in range(n + 1):
            new_basis_vector -= (
                np.dot(basis[k], set_independent[:, n])
                / np.dot(basis[k], basis[k])
                * basis[k]
            )
        new_basis_vector = new_basis_vector / np.linalg.norm(new_basis_vector)
        basis.append(new_basis_vector)

    return basis[1:]


def create_reconfig(p_list: list, pos: int = 0):
    """Sets p[pos] = 0, and changes other p such that \\sum_i p_i = 0.

    Args:
        p_list (list): List of power injections.
        pos (int, optional): Position, which power injection is set to zero. Defaults to 0.

    return:
        p_list (list): List of reconfigured power injections, such that one injection is zero and the grid
            is still balanced.
    """

    added = p_list[pos]

    for n in range(len(p_list)):
        if n == pos:
            p_list[n] -= added
        else:
            p_list[n] += added / (len(p_list) - 1)

    return p_list


def sample_balanced_p(N_nodes: int, basis_for_p_plane: list, power_factor: float = 1.0, reconf: bool = False):
    """Sample balanced power injections using a basis for the hyperplane,
    which can be calculated using get_orth_basis_hyperplane for
    normal vector (1,1,....,1).

    Args:
        N_nodes (int): Number of nodes of the grid.
        basis_for_p_plane (list): List of basis vectors for the hyperplane.
        power_factor (float, optional): Scalar to scale the power injections. Defaults to 1.
        reconf (bool, optional): Whether to set the first power injection to zero. Defaults to False.

    returns:
        p_list (list): List of power injections.
    """

    coeffs_for_p = np.random.rand(N_nodes - 1)
    coeffs_for_p = 2 * coeffs_for_p - 1
    p_list = np.zeros(N_nodes)
    for nn, coeff_n in enumerate(coeffs_for_p):
        p_list += coeff_n * np.asarray(basis_for_p_plane[nn]) * power_factor

    if reconf is True:
        p_list = create_reconfig(p_list, 0)

    return p_list


def get_convergent_init_conds(net, power_factor: float = 1.0, max_trials: int = 10000):
    """Find coefficients for creating power injections that will surely converge up to given power factor.
    Try max_trials times until successful."""
    ppc_branch = net._ppc["branch"]
    vm_pu = net.res_bus.vm_pu.to_numpy()
    _, _, E, K_vec = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu)
    NN = len(vm_pu)
    K_mat = np.diag(K_vec)
    # get hyperplane, where all vectors have sum(p) = 0
    basis_for_p_plane = get_orth_basis_hyperplane(np.ones(NN), check=True)

    count_not_converged = 0
    count_converged = 0
    while count_converged < 1:
        if count_not_converged < max_trials:
            # sample power injections
            coeffs_for_p = np.random.rand(NN - 1)
            coeffs_for_p = 2 * coeffs_for_p - 1
            pp_vec = np.zeros(NN)
            for nn, coeff_n in enumerate(coeffs_for_p):
                pp_vec += coeff_n * np.asarray(basis_for_p_plane[nn]) * power_factor

            assert abs(np.sum(pp_vec)) < 1e-10, "p input not balanced"

            # calculate power flows
            thetas_lin, _ = lin_power_flow(pp_vec, K_mat, E)
            thetas_nonlin = nonlin_power_flow(pp_vec, K_mat, E, initial_guess_theta=thetas_lin)

            if np.isnan(thetas_nonlin).any():
                count_not_converged += 1
            else:
                count_converged += 1
        else:
            print('Limit not converged has been reached. Only {} samples have been found.'.format(count_converged))
            coeffs_for_p.fill(np.nan)
            break
    return coeffs_for_p