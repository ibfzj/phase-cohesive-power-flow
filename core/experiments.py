"""
===========================================================================================================
Existence of phase-cohesive solutions of the lossless power flow equations - Numerical Experiment Routines
===========================================================================================================
Numerical experiments, data pipelines, and figure generation.

This module collects routines that combine model construction and analytical tools to reproduce
the numerical experiments presented in the paper.

"""


import numpy as np
import pandapower.networks as pn
from copy import deepcopy
from scipy import stats
import pandas as pd

from .model import F, construct_ppgraph_from_pandapower_and_run_pf, build_E_and_K_from_ppc_branch, get_convergent_sol_given_p, get_orth_basis_hyperplane
from .analysis import compute_kappa_chi, get_max_line_load_default_powinj, get_max_line_load_randomized_powinj_fixedNsamples, get_S1_S2_hist_pf1_target_success, get_p_coeff_list, get_vm_profile_pv_fixed_others_one, adjust_incidence_matrix_for_positive_flows, get_upper_bound_loop, find_bridges, compute_edge_metrics, get_K_min_max_edges_pp

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 24, 'text.latex.preamble': r'\usepackage{amsmath}'})
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D

# ===== Figure 2 =====

def plot_Fig2(K_list=(0.5, 1.0, 1.5), fmin=-5.5, fmax=5.5, npts=5000, filename="F_fe_plot.pdf"):
    """
    Plot the function mathcal{F}_e(f_e) for several fixed coupling values K_e.

    This produces the introductory figure by evaluating F(f_e, K_e) on a grid of flow values
    and plotting one curve per K_e.

    Args:
        K_list (tuple[float, ...]): Coupling values K_e to include in the plot.
        fmin (float): Minimum flow value f_e.
        fmax (float): Maximum flow value f_e.
        npts (int): Number of sampling points for f_e.
        filename (str | None): If not None, saves the figure as a PDF to this path.
            If None, the plot is shown interactively.

    Returns:
        None
    """
    f_e = np.linspace(fmin, fmax, npts)

    plt.figure(figsize=(6, 2.5))

    colors = ['#DC143C', 'navy', '#228B22']

    for K_e, c in zip(K_list, colors):
        y = F(f_e, K_e)
        plt.plot(f_e, y, color=c, linewidth=1.5, label=rf'$K_e = {K_e:g}$')

    plt.xlabel(r'$f_e$', fontsize=16)
    plt.ylabel(r'$\mathcal{F}_e(f_e)$', fontsize=16)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlim(fmin, fmax)

    # choose y-limits based on K=1 curve (or just autoscale)
    y_ref = F(f_e, 1.0)
    plt.ylim(np.min(y_ref) - 1, np.max(y_ref) + 1)

    plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=4, width=1)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    else:
        plt.show()

# ===== Figure 3 =====

def compute_region_of_trust(pow_fac: float, casenum):
    """
    Compute line loads and deterministic error bounds for a test case.

    For the specified IEEE test case (case30 or case118), this routine:
      - builds the pandapower network and runs a lossless AC power flow,
      - scales the base injections by pow_fac,
      - computes the linear (DC) and nonlinear (AC) edge loads,
      - computes the region of trust bounds (kappa/K and chi/K).

    Args:
        pow_fac (float): Scalar factor multiplying the base active power injections.
        casenum (int): Test case identifier (30 or 118).

    Returns:
        dict: Dictionary containing:
            net (pandapowerNet): Solved pandapower network.
            vm_pu (np.ndarray): Bus voltage magnitudes from the AC solution.
            thetas (np.ndarray): Bus voltage angles (radians) from the AC solution.
            p (np.ndarray): Base active-power injections (per-unit).
            psi_lin (np.ndarray): Linear (DC) edge loads.
            psi_nonlin (np.ndarray): Nonlinear (AC) edge loads (np.nan if not converged).
            kappa_over_K (np.ndarray): Error term per edge, normalized by K.
            chi_over_K (np.ndarray): Error term per edge, normalized by K.
    """
    # Build and solve pandapower case30
    if casenum == 30:
        net_pp = pn.case30()
    elif casenum == 118:
        net_pp = pn.case118()
    else:
        raise NotImplementedError("Case '{}' not implemented!".format(casenum))

    #Run power flow
    net_pp, vm_pu, thetas, p, _, _, _, _ = construct_ppgraph_from_pandapower_and_run_pf(net_pp)

    # Power vector for this power factor
    power_vector = p * pow_fac

    # Linear and nonlinear solutions on edges
    psi_lin, psi_nonlin, _, _, _, _ = get_convergent_sol_given_p(net_pp, power_vector)

    # Calculate E, K
    ppc_branch = net_pp._ppc["branch"]
    _, _, E, K_vec = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu)
    
    # Error bounds (region of trust) based on DC solution
    kappa_over_K, chi_over_K = compute_kappa_chi(K_vec, E, psi_lin)

    return {
        "net": net_pp,
        "vm_pu": vm_pu,
        "thetas": thetas,
        "p": p,
        "psi_lin": np.asarray(psi_lin, dtype=float),
        "psi_nonlin": np.asarray(psi_nonlin, dtype=float),
        "kappa_over_K": np.asarray(kappa_over_K, dtype=float),
        "chi_over_K": np.asarray(chi_over_K, dtype=float)
        }


def plot_region_of_trust(pow_fac_list_a, pow_fac_b=1.0, pow_fac_c=8.0, casenum = 30, filename = None):
    """
    Make a 3-panel figure for case 30 (default) or case 118:
      (a) region of trust vs edge index for several power factors (linear/DC load)
      (b) scatter: difference between load_lin and load_nonlin (y-axis) vs load_lin (x-axis) for pow_fac_b, with error bars (region of trust)
      (c) same as (b) for pow_fac_c

    Args:
        pow_fac_list_a (array-like): Power factors shown in panel (a).
        pow_fac_b (float): Power factor used for panel (b).
        pow_fac_c (float): Power factor used for panel (c).
        casenum (int): Test case identifier (30 or 118).
        filename (str | None): If not None, saves the figure as a PDF to this path.
            If None, the plot is shown interactively.

    Returns:
        None
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'wspace': 0.2})
    ax_a, ax_b, ax_c = axes

    # Panel labels
    ax_a.text(0.02, 0.02, "(a)", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=16)
    ax_b.text(0.02, 0.02, "(b)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=16)
    ax_c.text(0.02, 0.02, "(c)", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=16)

    # ---- Panel (a): region of trust vs edge index ----
    cmap = mpl.colormaps['viridis']
    colors_list = cmap(np.linspace(0, 1, len(pow_fac_list_a)))

    base_data = compute_region_of_trust(pow_fac_list_a[0], casenum)
    Ne = len(base_data["psi_lin"])
    edge_indices = np.arange(Ne)

    for i, pow_fac in enumerate(pow_fac_list_a):
        data = compute_region_of_trust(pow_fac, casenum)
        psi_lin = data["psi_lin"]
        kappa_over_K = data["kappa_over_K"]
        chi_over_K = data["chi_over_K"]

        color = colors_list[i]

        # Region of trust
        lower = psi_lin - kappa_over_K - chi_over_K
        upper = psi_lin - kappa_over_K + chi_over_K

        ax_a.fill_between(edge_indices, lower, upper, alpha=0.65, color=color, edgecolor='none')

        # errorbars: center at midpoint of the interval
        y_center = psi_lin - kappa_over_K
        yerr = chi_over_K

        ax_a.errorbar(edge_indices, y_center, yerr=yerr, elinewidth=0.5, linestyle='-', fmt='.',
                      ecolor=color, color=color, lw=0.5, label=rf'$p_f = {pow_fac}$')

    ax_a.set_xlabel(r'$e$')
    ax_a.set_ylabel(r'$\psi_e$')
    ax_a.legend(loc='lower right', fontsize = 14, borderaxespad=0)
    ax_a.grid(linestyle=':', alpha=0.7)
    ax_a.xaxis.set_minor_locator(AutoMinorLocator(10))  
    ax_a.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

    # ---- Helper for panels (b) and (c) ----

    def scatter_lin_vs_nonlin_with_bounds(ax, pow_fac, cbar_on = True):
        data = compute_region_of_trust(pow_fac, casenum)
        psi_lin    = data["psi_lin"]
        psi_nonlin = data["psi_nonlin"]
        kappa_over_K  = data["kappa_over_K"]        
        chi_over_K    = data["chi_over_K"]   

        Ne  = len(psi_lin)
        idx = np.arange(Ne)

        # Difference between linear and nonlinear solution
        diff   = psi_lin - psi_nonlin
        center = kappa_over_K
        err    = chi_over_K

        #Scientific format for both subplots
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Error bars
        ax.errorbar(psi_lin, center, yerr=err, fmt='none', ecolor='gray', alpha=0.7, elinewidth=0.7, capsize=2, label='Region of trust')

        # Discrete edge colors
        base_cmap   = mpl.colormaps['plasma']
        colors_41   = base_cmap(np.linspace(0, 1, Ne))  # one color per edge
        cmap_disc   = ListedColormap(colors_41)
        boundaries  = np.arange(Ne + 1) - 0.5
        norm        = BoundaryNorm(boundaries, cmap_disc.N)

        sc = ax.scatter(psi_lin, diff, s=40, c=idx, cmap=cmap_disc, norm=norm, alpha=0.95, edgecolor='k', linewidth=0.3)

        # Horizontal line y = 0
        ax.axhline(0.0, color='k', linestyle='--', linewidth=1)

        ax.set_xlabel(r'$\psi_e^{\circ}$')
        ax.set_ylabel(r'$\psi_e^{\circ} - \psi_e^{*}$', labelpad = -10)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        # Colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap_disc, norm=norm)
        sm.set_array([])

        # Only every second index on the bar
        tick_positions = np.arange(0, Ne, 2)
        if cbar_on == True:
            # Create separate axis for the colorbar (keeps all three panels same width)
            cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0]) # [left, bottom, width, height] in axis-relative coords
            cbar = plt.colorbar(sm, cax=cax, ticks=tick_positions)
            cbar.set_label('Edge index $e$')
            cbar.ax.tick_params(labelsize=8)

        # Add power factor
        ax.text(0.04, 0.96, rf"$p_f = {pow_fac}$", transform=ax.transAxes, ha="left", va="top", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.8))

    # ---- Panel (b): pow_fac_b ----
    scatter_lin_vs_nonlin_with_bounds(ax_b, pow_fac_b, False)

    # ---- Panel (c): pow_fac_c ----
    scatter_lin_vs_nonlin_with_bounds(ax_c, pow_fac_c, True)

    fig.subplots_adjust(right=0.88, wspace=0.2)

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()


# ===== Figure 4 =====

def build_case_context(casenum: int):
    """
    Calculate core data structures for a given test case.

    This helper:
      - loads the specified pandapower test network,
      - runs a lossless AC power flow,
      - builds the incidence matrix E and nominal couplings K_vec,
      - constructs a basis for the balanced-injection hyperplane (sum(p)=0).

    Args:
        casenum (int): Test case identifier (30 or 118).

    Returns:
        dict: Context dictionary containing:
            casenum (int): Case identifier.
            net (pandapowerNet): Solved pandapower network.
            vm_pu (np.ndarray): Bus voltage magnitudes from the AC solution.
            thetas (np.ndarray): Bus voltage angles (radians) from the AC solution.
            p_base (np.ndarray): Base active power injections (per-unit).
            E (np.ndarray): Node-edge incidence matrix.
            K_vec (np.ndarray): Line coupling coefficients (per branch).
            NN (int): Number of buses.
            basis_for_p_plane (list[np.ndarray]): Orthonormal basis spanning sum(p)=0.
    """
    if casenum == 30:
        net0 = pn.case30()
    elif casenum == 118:
        net0 = pn.case118()
    else:
        raise NotImplementedError(f"Case '{casenum}' not implemented!")

    net, vm_pu, thetas, p, _, _, _, _ = construct_ppgraph_from_pandapower_and_run_pf(net0)
    ppc_branch = net._ppc["branch"]
    _, _, E, K_vec = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu)

    return {
        "casenum": casenum,
        "net": net,
        "vm_pu": vm_pu,
        "thetas": thetas,
        "p_base": deepcopy(p),
        "E": E,
        "K_vec": K_vec,
        "NN": len(net.bus),
        "basis_for_p_plane": get_orth_basis_hyperplane(np.ones(len(net.bus)), check=True),
    }

def compute_fig4a(ctx, pow_fac_list):
    """
    Compute Fig. 4(a) data: max load and bound for scaled default injections.

    For each power factor in pow_fac_list, this evaluates:
      S1 = max_e |psi_nonlin,e| and
      S2 = max_e of the phase-cohesiveness bound.

    Args:
        ctx (dict): Case context from build_case_context.
        pow_fac_list: Power factors to evaluate.

    Returns:
        dict: Dictionary with:
            pow_fac_list (np.ndarray): Power factors (float).
            S1 (np.ndarray): Max nonlinear loads for each power factor.
            S2 (np.ndarray): Max bounds for each power factor.
    """
    net, p_base, E, K_vec = ctx["net"], ctx["p_base"], ctx["E"], ctx["K_vec"]
    S1def, S2def = get_max_line_load_default_powinj(net, p_base, pow_fac_list, E, K_vec)
    return {"pow_fac_list": np.asarray(pow_fac_list), "S1": np.asarray(S1def), "S2": np.asarray(S2def)}

def compute_fig4b_coeffs(ctx, pow_fac_list, trial_num, min_success, max_tries, norepeat=True):
    """
    Generate coefficient vectors for randomized balanced injections used in Fig. 4(b).

    The returned coefficient vectors parameterize balanced injection directions that
    are filtered by convergence criteria across the specified power-factor range.

    Args:
        ctx (dict): Case context from build_case_context.
        pow_fac_list (array-like): Power factors to evaluate.
        trial_num (int): Number of candidate random draws.
        min_success (int): Minimum number of accepted coefficient vectors.
        max_tries (int): Maximum number of attempts to reach min_success.
        norepeat (bool): Passed to the underlying solver screening routine.

    Returns:
        list[np.ndarray]: List of coefficient vectors defining balanced injection directions.
    """
    net = ctx["net"]
    p_coeff_list = get_p_coeff_list(net, pow_fac_list, trial_num, min_success, max_tries, norepeat)
    return p_coeff_list

def compute_fig4b(ctx, pow_fac_list, p_coeff_list, N_target=200):
    """
    Compute Fig. 4(b) data: averaged max load and bound over randomized injections.

    Using coefficient vectors p_coeff_list, this evaluates S1 and S2 at each power
    factor and averages them over exactly N_target samples that converge for all pfs.

    Args:
        ctx (dict): Case context from build_case_context.
        pow_fac_list (array-like): Power factors to evaluate.
        p_coeff_list (list[np.ndarray]): Coefficient vectors defining balanced injections.
        N_target (int): Number of fully valid samples to average over.

    Returns:
        dict: Dictionary with:
            pow_fac_list (np.ndarray): Power factors (float).
            S1 (np.ndarray): Averaged max nonlinear loads.
            S2 (np.ndarray): Averaged max bounds.
    """
    net, E, K_vec, basis_for_p_plane = ctx["net"], ctx["E"], ctx["K_vec"], ctx["basis_for_p_plane"]
    avg_S1s, avg_S2s = get_max_line_load_randomized_powinj_fixedNsamples(net, pow_fac_list, p_coeff_list, E, K_vec, N_target=N_target,norepeat=False)
    return {"pow_fac_list": np.asarray(pow_fac_list), "S1": np.asarray(avg_S1s), "S2": np.asarray(avg_S2s)}

def compute_fig4c_hist(ctx, N_success, max_attempts=200_000, norepeat=True, seed=None, numbins=60):
    """
    Compute Fig. 4(c) histogram data for the ratio S2/S1 at p_f = 1.

    The routine repeatedly samples balanced injections at p_f=1 and keeps only
    convergent trials until N_success samples are collected (or max_attempts is hit).
    It then forms the ratio S2/S1 and returns a normalized histogram.

    Args:
        ctx (dict): Case context from build_case_context.
        N_success (int): Number of convergent samples to collect.
        max_attempts (int): Hard cap on attempt number.
        norepeat (bool): If True, do not use perturbation-based retries inside the solver.
        seed (int | None): Random seed for reproducibility.
        numbins (int): Number of histogram bins.

    Returns:
        dict: Dictionary with:
            x (np.ndarray): Bin coordinate array (as constructed from relfreq output).
            freq (np.ndarray): Relative frequencies (normalized histogram).
            binsize (float): Bin width.
            attempted (int): Number of attempted samples.
    """
    net, E, K_vec = ctx["net"], ctx["E"], ctx["K_vec"]
    S1, S2, attempted = get_S1_S2_hist_pf1_target_success(
        net, N_success=N_success, E_matrix=E, K_vec=K_vec,
        max_attempts=max_attempts, norepeat=norepeat, seed=seed
    )
    ratio = S2 / S1
    res = stats.relfreq(ratio, numbins=numbins, defaultreallimits=[float(np.min(ratio)), float(np.max(ratio))])
    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    return {"x": x, "freq": res.frequency, "binsize": res.binsize, "attempted": attempted}

def plot_Fig_4_panels(
    # Panel (a): fixed power injections
    pow_fac_list_a,
    S1def_a,
    S2def_a,
    # Panel (b): randomized power injections (averaged)
    pow_fac_list_b,
    avg_S1s_b,
    avg_S2s_b,
    # Panel (c): histogram (pf = 1)
    hist_x,
    hist_freq,
    hist_binsize,
    # Metadata
    panel_labels=("a", "b", "c"),
    filename=None
):
    """
    Create Fig. 4 with three panels:

      (a) Maximum load (S1) and phase-cohesiveness bound (S2)
          for fixed default power injections vs power factor.
      (b) Same quantities averaged over randomized power injections.
      (c) Histogram of the ratio S2 / S1 for randomized injections at p_f = 1.

    Args:
        pow_fac_list_a (array-like): Power factors for panel (a).
        S1def_a (array-like): S1 values for panel (a).
        S2def_a (array-like): S2 values for panel (a).
        pow_fac_list_b (array-like): Power factors for panel (b).
        avg_S1s_b (array-like): Averaged S1 values for panel (b).
        avg_S2s_b (array-like): Averaged S2 values for panel (b).
        hist_x (array-like): Histogram x-coordinates (bin centers or left edges).
        hist_freq (array-like): Histogram frequencies (normalized).
        hist_binsize (float): Histogram bin width.
        panel_labels (tuple[str, str, str]): Panel label letters to display.
        filename (str | None): If not None, saves the figure as a PDF to this path.
            If None, the plot is shown interactively.

    Returns:
        None
    """
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, 4))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6))

    # ---------------- Panel (a): fixed injections ----------------
    ax0.grid(True)

    ax0.plot(pow_fac_list_a, S1def_a, '-d', mfc='none', color=colors[0], label=r'$\sin \gamma$')
    ax0.plot(pow_fac_list_a, S2def_a, '-v', mfc='none', color=colors[1], label=r'$\Psi$')

    ax0.set_xlabel(r'$p_f$', labelpad=-5)
    ax0.set_ylabel(r'$\max\limits_{e} f_e / K_e$')
    ax0.legend(loc='lower right', fontsize=16)
    ax0.set_title('Fixed power injections')

    ax0.text(0.02, 0.95, f"({panel_labels[0]})", transform=ax0.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

    # ---------------- Panel (b): randomized injections ----------------
    ax1.grid(True)

    ax1.plot(pow_fac_list_b, avg_S1s_b, '-d', mfc='none', color=colors[0], label=r'$\sin \gamma$')
    ax1.plot(pow_fac_list_b, avg_S2s_b, '-v', mfc='none', color=colors[1], label=r'$\Psi$')

    ax1.set_xlabel(r'$p_f$', labelpad=-5)
    ax1.set_ylabel(r'$\max\limits_{e} f_e / K_e$', labelpad=-5)
    ax1.legend(loc='lower right', fontsize=16)
    ax1.set_title('Randomized power injections')

    ax1.text(0.02, 0.95, f"({panel_labels[1]})", transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

    # ---------------- Panel (c): histogram ----------------
    ax2.grid(True)

    ax2.bar(hist_x, hist_freq, width=hist_binsize, color=colors[1], label=r'$p_f = 1$')

    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\Psi / \sin \gamma$')
    ax2.set_ylabel('Relative frequency')
    ax2.legend(loc='upper right', fontsize=16)

    ax2.axvline(1.0, linestyle='--', color='k')
    ax2.set_title('Randomized power injections')

    ax2.text(0.1, 0.95, f"({panel_labels[2]})", transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')

    # ---------------- Show/Save ----------------

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    else:
        plt.show()

# --- Functions for saving data that takes a long time to calculate ---

def save_p_coeff_list_csv_auto(p_coeff_list, case_name: str, pow_fac_list):
    pf_max = float(np.max(pow_fac_list))
    n_succ = len(p_coeff_list)

    arr = np.asarray(p_coeff_list, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    filename = (
        f"pcoeff_{case_name}"
        f"_pfmax{pf_max:.2f}"
        f"_values{n_succ}"
        ".csv"
    )

    pd.DataFrame(arr).to_csv(filename, index=False)
    print(f"Saved {n_succ} samples to '{filename}'")
    return filename

def load_p_coeff_list_csv(filename: str):
    arr = pd.read_csv(filename).to_numpy(dtype=float)
    return [arr[i, :] for i in range(arr.shape[0])]

def save_ratio_hist_csv_auto(ratio, case_name: str, pf: float = 1.0):
    arr = np.asarray(ratio, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array for ratio, got shape {arr.shape}")

    n_vals = arr.size

    filename = (
        f"ratio_case{case_name}"
        f"_pf{pf:.2f}"
        f"_values{n_vals}"
        ".csv"
    )

    df = pd.DataFrame({"ratio": arr})
    df.to_csv(filename, index=False)

    print(f"Saved {n_vals} ratio samples to '{filename}'")
    return filename

def load_ratio_hist_csv(filename: str):
    df = pd.read_csv(filename)
    arr = df["ratio"].to_numpy(dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"Loaded ratio array has shape {arr.shape}, expected 1D")

    return arr

def save_fig4b_avg_csv(avg_S1, avg_S2, pow_fac_list, case_name: str, N_target: int):
    """
    Save averaged Fig. 4(b) curves (S1, S2) to CSV.
    """
    avg_S1 = np.asarray(avg_S1, dtype=float)
    avg_S2 = np.asarray(avg_S2, dtype=float)
    pow_fac_list = np.asarray(pow_fac_list, dtype=float)

    if not (avg_S1.shape == avg_S2.shape == pow_fac_list.shape):
        raise ValueError("avg_S1, avg_S2, and pow_fac_list must have the same shape")

    pf_max = float(np.max(pow_fac_list))

    filename = (
        f"fig4b_avg_{case_name}"
        f"_pfmax{pf_max:.2f}"
        f"_Nsamples{N_target}"
        ".csv"
    )

    df = pd.DataFrame({
        "pf": pow_fac_list,
        "S1": avg_S1,
        "S2": avg_S2,
    })

    df.to_csv(filename, index=False)
    print(f"Saved Fig. 4(b) averages to '{filename}'")
    return filename

def load_fig4b_avg_csv(filename: str):
    """
    Load averaged Fig. 4(b) curves saved by save_fig4b_avg_csv.
    """
    df = pd.read_csv(filename)

    required = {"pf", "S1", "S2"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}")

    pow_fac_list = df["pf"].to_numpy(dtype=float)
    avg_S1 = df["S1"].to_numpy(dtype=float)
    avg_S2 = df["S2"].to_numpy(dtype=float)

    return pow_fac_list, avg_S1, avg_S2


# ===== Figure 5 =====

def calculate_impact_of_voltage_stability_on_phase_cohesiveness(net, all_v_mins, pow_fac: float = 1.0):
    """
    Compute Fig. 5 data: voltage-stability sweep and phase-cohesiveness bound.

    The routine:
      - runs a lossless AC power flow to obtain the AC angles,
      - constructs DC angles/flows using the Laplacian with nominal couplings,
      - orients edges so the DC flows are nonnegative,
      - sweeps the PQ-bus lower-voltage bound v_min and computes arcsin(Psi(v_min)),
      - returns the breaking point where Psi first becomes <= 1 (if it occurs).

    Args:
        net (pandapowerNet): Pandapower network to analyze.
        all_v_mins (array-like): Grid of PQ lower-bound values v_min to sweep.
        pow_fac (float): Scalar factor multiplying the base active-power injections.

    Returns:
        breaking_point (float): First v_min in the sweep for which Psi <= 1 (or the last value if never satisfied).
        upper_bound (np.ndarray): Array of arcsin(Psi(v_min)) values (may contain nan where Psi>1).
        x (np.ndarray): Constant x-coordinate array used for scatter points (min of AC vm_pu).
        theta_diff (np.ndarray): Absolute AC phase differences along (oriented) edges.
    """

    # Run power flow
    net, vm_pu, thetas_ac, p, q, B, f_ppc, t_ppc = construct_ppgraph_from_pandapower_and_run_pf(net)
    ppc_branch = net._ppc["branch"]
    p_used = p * float(pow_fac)

    # Build incidence and nominal K from PPC using PV and slack fixed to reference values, others at 1pu
    vm_pu_for_K = get_vm_profile_pv_fixed_others_one(net)
    f2, t2, E, K_nom = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu_for_K)

    # Ensure consistent edge ordering
    if not (np.array_equal(f2, f_ppc) and np.array_equal(t2, t_ppc)):
        raise ValueError("Edge ordering mismatch: construct_pf f/t != PPC-derived f/t. Use PPC consistently.")

    # DC angles and flows 
    L = E @ np.diag(K_nom) @ E.T

    # Slack bus index from pandapower (use the first ext_grid as slack)
    slack_bus = int(net.ext_grid["bus"].iloc[0])
    theta_dc = np.zeros(L.shape[0], dtype=float)

    # Build mask that removes slack bus
    mask = np.ones(L.shape[0], dtype=bool)
    mask[slack_bus] = False

    # Solve reduced system
    theta_dc[mask] = np.linalg.solve(L[np.ix_(mask, mask)], p_used[mask])
    theta_dc[slack_bus] = 0.0

    flows_lin = (np.diag(K_nom) @ (E.T @ theta_dc))  # f = K E^T theta

    # Flip edges so flows are nonnegative
    E_adjusted = adjust_incidence_matrix_for_positive_flows(E, flows_lin)
    flows_lin_flipped = (np.diag(K_nom) @ (E_adjusted.T @ theta_dc))
    # Flip endpoints too
    f_adj = f_ppc.copy()
    t_adj = t_ppc.copy()
    neg = flows_lin < 0
    f_adj[neg], t_adj[neg] = t_adj[neg], f_adj[neg]

    # Compute upper bound curve
    upper_bound, Psi_vals = get_upper_bound_loop(all_v_mins=all_v_mins, net=net, B=B, ppc_branch=ppc_branch,
        E_adjusted=E_adjusted, flows_lin=flows_lin_flipped, f=f_adj, t=t_adj)

    # Find "breaking point": first v_min where Psi <= 1
    ok = Psi_vals <= 1.0

    if np.any(ok):
        i0 = int(np.argmax(ok))  # first True
        breaking_point = float(all_v_mins[i0])
    else:
        breaking_point = float(all_v_mins[-1])  # never guaranteed in the sweep

    # Exact AC phase diffs on the same (flipped) edge orientation 
    theta_diff = np.abs(E_adjusted.T @ thetas_ac)

    # x-coordinates for scatter points (constant at min(vm_pu))
    x = np.full_like(theta_diff, float(np.min(vm_pu)))

    return breaking_point, upper_bound, x, theta_diff


def plot_Fig_5_two_cases(
    all_v_mins_a, upper_bound_a, x_a, theta_diff_a, breaking_point_a,
    all_v_mins_b, upper_bound_b, x_b, theta_diff_b, breaking_point_b,
    filename="Fig5.pdf",
):
    """
    Plot Fig. 5 for two test cases as vertically stacked panels.

    Inputs correspond to outputs of calculate_impact_of_voltage_stability_on_phase_cohesiveness
    for two different networks/cases.

    Args:
        all_v_mins_a: Sweep grid for panel (a).
        upper_bound_a: arcsin(Psi) curve for panel (a).
        x_a: Scatter x-coordinates for panel (a).
        theta_diff_a: Scatter phase differences for panel (a).
        breaking_point_a: v_min breaking point for panel (a).

        all_v_mins_b: Sweep grid for panel (b).
        upper_bound_b: arcsin(Psi) curve for panel (b).
        x_b: Scatter x-coordinates for panel (b).
        theta_diff_b: Scatter phase differences for panel (b).
        breaking_point_b: v_min breaking point for panel (b).

        filename (str | None): If not None, saves the figure as a PDF to this path.
            If None, the plot is shown interactively.

    Returns:
        None
    """

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5.5, 6.5))

    # ----- Panel (a) -----
    ax0.plot(all_v_mins_a, upper_bound_a, '-', label=r'$\arcsin \Psi$')
    ax0.scatter(x_a, theta_diff_a, color='r', s=2, label='numerically exact')
    ax0.set_ylabel(r'$|\theta_m - \theta_n|$')
    ax0.axvline(breaking_point_a, color='k', ls='dashed', lw=0.75)
    ax0.legend(loc='lower left', fontsize=16)
    ax0.set_yscale('log')

    ax0.text(0.4, 0.9, "Case 30", fontsize=20, fontweight='bold', transform=ax0.transAxes)
    ax0.text(-0.25, 0.9, "(a)", fontsize=20, fontweight='bold', transform=ax0.transAxes)

    # ----- Panel (b) -----
    ax1.plot(all_v_mins_b, upper_bound_b, '-', label=r'$\arcsin \Psi$')
    ax1.scatter(x_b, theta_diff_b, color='r', s=2, label='numerically exact')
    ax1.set_ylabel(r'$|\theta_m - \theta_n|$')
    ax1.set_xlabel(r'$v_{\min}$ [pu]')
    ax1.axvline(breaking_point_b, color='k', ls='dashed', lw=0.75)
    ax1.set_yscale('log')

    ax1.text(0.4, 0.9, "Case 118", fontsize=20, fontweight='bold', transform=ax1.transAxes)
    ax1.text(-0.25, 0.9, "(b)", fontsize=20, fontweight='bold', transform=ax1.transAxes)

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    else:
        plt.show()

# ===== Figure 6 =====

def get_v_bounds_pv_slack_fixed_pq_interval(net, vmin_pq: float, vmax_pq: float):
    """
    Construct per-bus voltage bounds with PV/slack fixed and PQ in an interval.

    PQ buses are assigned bounds [vmin_pq, vmax_pq]. PV and slack buses are fixed
    to their reference setpoints, i.e., v_min = v_max = vm_pu.

    Args:
        net (pandapowerNet): Pandapower network.
        vmin_pq (float): Lower bound applied to all PQ buses.
        vmax_pq (float): Upper bound applied to all PQ buses.

    Returns:
        v_min (np.ndarray): Bus-wise lower voltage bounds.
        v_max (np.ndarray): Bus-wise upper voltage bounds.
    """
    Nn = len(net.bus)
    v_min = vmin_pq * np.ones(Nn, dtype=float)
    v_max = vmax_pq * np.ones(Nn, dtype=float)

    # Slack buses fixed
    if "ext_grid" in net and not net.ext_grid.empty and "vm_pu" in net.ext_grid:
        for _, row in net.ext_grid.iterrows():
            b = int(row["bus"])
            vref = float(row["vm_pu"])
            v_min[b] = vref
            v_max[b] = vref

    # PV buses fixed
    if "gen" in net and not net.gen.empty and "vm_pu" in net.gen:
        for _, row in net.gen.iterrows():
            b = int(row["bus"])
            vref = float(row["vm_pu"])
            v_min[b] = vref
            v_max[b] = vref

    return v_min, v_max


def compute_case_data_for_powfac_uncertain(pow_fac: float = 1.0, vmin: float = 0.9, vmax: float = 1.1, casenum = 30):
    """
    Compute uncertain-voltage error bounds for a given loading level.

    Starting from the deterministic solution (compute_region_of_trust), this routine:
      - builds nominal couplings using PV/slack at reference voltages and PQ to 1 pu,
      - computes DC angles/flows for the base injections and orients edges so flows are nonnegative,
      - constructs bus-wise voltage bounds (PQ in [vmin, vmax], PV/slack fixed),
      - computes edge-wise coupling bounds (K_min, K_max),
      - computes Corollary 5 error bounds (kappa_min, kappa_max, chi_max),
      - converts bounds to load space by dividing by K_nom.

    Args:
        pow_fac (float): Scalar factor multiplying the base active-power injections.
        vmin (float): Lower bound for PQ voltages.
        vmax (float): Upper bound for PQ voltages.
        casenum (int): Test case identifier (30 or 118).

    Returns:
        dict: The deterministic base dictionary from compute_region_of_trust extended with:
            kappa_lo_unc (np.ndarray): Lower kappa bound in load units (kappa_min / K_nom).
            kappa_hi_unc (np.ndarray): Upper kappa bound in load units (kappa_max / K_nom).
            chi_unc (np.ndarray): Error radius in load units (chi_max / K_nom).
    """
    # Start from existing data when voltages are not uncertain
    base = compute_region_of_trust(pow_fac, casenum)

    net_pp   = base["net"]
    p = base["p"]

    Nn = len(net_pp.bus)

    # Nominal couplings K_nom from the nominal voltage (PV and slack fixed to reference values, others fixed to 1.)
    vm_pu_for_K = get_vm_profile_pv_fixed_others_one(net_pp)
    ppc_branch = net_pp._ppc["branch"]
    f_ppc, t_ppc, E, K_nom_vec = build_E_and_K_from_ppc_branch(ppc_branch, vm_pu_for_K)

    L = E @ np.diag(K_nom_vec) @ E.T

    slack_bus = int(net_pp.ext_grid["bus"].iloc[0])
    mask = np.ones(L.shape[0], dtype=bool)
    mask[slack_bus] = False

    theta_dc = np.zeros(L.shape[0], dtype=float)
    theta_dc[mask] = np.linalg.solve(L[np.ix_(mask, mask)], p[mask])
    theta_dc[slack_bus] = 0.0

    # DC flows and angles for nominal K
    f_lin = np.diag(K_nom_vec) @ (E.T @ theta_dc)

    # Flip edges so flows are nonnegative
    E_adjusted = adjust_incidence_matrix_for_positive_flows(E, f_lin)
    f_lin_flipped = np.diag(K_nom_vec) @ (E_adjusted.T @ theta_dc)

    # Flip endpoints too
    f_adj = f_ppc.copy()
    t_adj = t_ppc.copy()
    neg = f_lin < 0
    f_adj[neg], t_adj[neg] = t_adj[neg], f_adj[neg]

    # Voltage bounds at buses (PQ uncertain, PV and slack fixed)
    v_min, v_max = get_v_bounds_pv_slack_fixed_pq_interval(net_pp, vmin, vmax)

    # Edge-wise K_min, K_max
    K_min, K_max = get_K_min_max_edges_pp(ppc_branch, v_min, v_max)

    # Bridges as list of (u, v) tuples (for compute_edge_metrics function)
    list_of_bridges = find_bridges(Nn, f_ppc, t_ppc)

    # Corollary 5 metrics
    kappa_max, kappa_min, chi_max = compute_edge_metrics(E_adjusted, K_min, K_max, f_lin_flipped, list_of_bridges, f_adj, t_adj)

    # Convert to loads using nominal K
    kappa_lo_unc  = kappa_min / K_nom_vec    # kappa_min / K_nom
    kappa_hi_unc  = kappa_max / K_nom_vec    # kappa_max / K_nom
    chi_unc = chi_max   / K_nom_vec    # chi_max / K_nom

    base["kappa_lo_unc"]  = kappa_lo_unc
    base["kappa_hi_unc"]  = kappa_hi_unc
    base["chi_unc"] = chi_unc

    return base

def plot_error_bounds_with_uncertain_voltages(pow_fac: float = 1.0, v_range_a=(0.95, 1.05),
    v_range_b=(0.90, 1.10), casenum = 30, filename = None):
    """
    Plot voltage certain and voltage uncertain error bounds around the linear solution.

    The plot shows:
      - error bars for no voltage uncertainty,
      - two regions of trust for PQ voltages in intervals v_range_a and v_range_b,
        each shown as a shaded band plus error bars.

    Args:
        pow_fac (float): Scalar factor multiplying the base active-power injections.
        v_range_a (tuple[float, float]): First PQ voltage interval (vmin, vmax).
        v_range_b (tuple[float, float]): Second PQ voltage interval (vmin, vmax).
        casenum (int): Test case identifier (30 or 118).
        filename (str | None): If not None, saves the figure as a PDF to this path.
            If None, the plot is shown interactively.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    # No voltage uncertainty
    base_det = compute_region_of_trust(pow_fac, casenum)
    psi_lin_det      = base_det["psi_lin"]
    kappa_over_K_det = base_det["kappa_over_K"]
    chi_over_K_det   = base_det["chi_over_K"]

    Ne = len(psi_lin_det)
    e = np.arange(Ne)

    # Region of trust for no voltage uncertainty (same as Fig. 3(a))
    y_center_det = psi_lin_det - kappa_over_K_det
    yerr_det     = chi_over_K_det
    ax.errorbar(e, y_center_det, yerr=yerr_det, elinewidth=0.6, linestyle='-', fmt='.',
        ecolor="black", color="black", lw=0.6, zorder=6, label=r"no uncertainty")

    # Uncertain voltages
    cmap    = mpl.colormaps["viridis"]
    colors  = [cmap(0.15), cmap(0.75)]
    markers = [".", "."]
    ranges  = [v_range_a, v_range_b]
    labels  = [
        rf"$v \in [{v_range_a[0]:.2f}, {v_range_a[1]:.3f}]$",
        rf"$v \in [{v_range_b[0]:.2f}, {v_range_b[1]:.3f}]$",
    ]

    for (vmin, vmax), color, marker, label in zip(ranges, colors, markers, labels):
        data_unc = compute_case_data_for_powfac_uncertain(pow_fac, vmin=vmin, vmax=vmax, casenum = casenum)

        psi_lin       = data_unc["psi_lin"]
        kappa_lo_unc  = data_unc["kappa_lo_unc"]
        kappa_hi_unc  = data_unc["kappa_hi_unc"]
        chi_unc       = data_unc["chi_unc"]

        lower = psi_lin - (kappa_hi_unc + chi_unc)
        upper = psi_lin - (kappa_lo_unc - chi_unc)
        center = 0.5 * (lower + upper)
        yerr   = upper - center

        # Filled robust envelope
        ax.fill_between(e, lower, upper, color=color, alpha=0.08, linewidth=0, zorder=2)

        # Center + error bars
        ax.errorbar(e, center, yerr=yerr, fmt=marker, markersize=4, color=color, ecolor=color,
            lw=1.0, elinewidth=0.8, capsize=2, zorder=4, label=label)

    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$\psi_e$")
    ax.grid(linestyle=":", alpha=0.7)

    ax.legend(loc="best", fontsize=12)

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
