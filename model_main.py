"""
model_main.py
=============
Python translation of model_main.m

Heterogeneous-firm model with financial frictions, capital adjustment costs,
and an interest-rate shock sequence calibrated to Southern Europe.

Reference:
  "Capital Allocation and Productivity in South Europe"
  (see paper for model details, Section starting p. 1935)

Structure
---------
  1. Parameters and Setup
  2. Value Function Iteration (VFI) with Howard's improvement
  3. Simulation (50 000 firms, 1 000 periods)
  4. Dataset construction and export to Excel

Performance note
----------------
The VFI inner loop iterates over n_choice = nk*na = 14 400 current states,
and for each builds a (14 400 × n_state) utility matrix.  This is the same
algorithm as the MATLAB code; expect a wall-clock time of several hours on a
single CPU.  Installing Numba and enabling the JIT option (set USE_NUMBA=True
below) can cut that by 10-50×.  Alternatively, use fewer grid points while
testing (e.g., nk=na=30) and restore nk=na=120 for production runs.

Dependencies
------------
  numpy, scipy, pandas, openpyxl (for xlsxwriter)
  Optional: numba
"""

import numpy as np
import time
import pandas as pd
from scipy.stats import norm as scipy_norm

# ── Toggle for Numba JIT acceleration ────────────────────────────────────────
USE_NUMBA = False        # set True if numba is installed for a large speedup
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# 0.  Helper: Tauchen (1986) discretization of AR(1)
# ============================================================

def tauchen(n, mu, rho, sigma, m):
    """
    Discretize  x_t = mu + rho * x_{t-1} + sigma * eps_t,  eps ~ N(0,1)
    using Tauchen's (1986) equally-spaced grid method.

    Parameters
    ----------
    n     : int   – number of grid points
    mu    : float – intercept  (unconditional mean = mu / (1-rho))
    rho   : float – persistence
    sigma : float – std dev of innovations
    m     : float – width of grid in unconditional std devs

    Returns
    -------
    grid : (n,) array
    P    : (n, n) transition matrix  (rows sum to 1)
    """
    mean_unc = mu / (1.0 - rho)
    std_unc  = sigma / np.sqrt(1.0 - rho**2)

    grid = np.linspace(mean_unc - m * std_unc,
                       mean_unc + m * std_unc, n)
    step = grid[1] - grid[0]

    P = np.zeros((n, n))
    for i in range(n):
        cond_mean = mu + rho * grid[i]
        for j in range(n):
            lo = grid[j] - step / 2.0
            hi = grid[j] + step / 2.0
            if j == 0:
                P[i, j] = scipy_norm.cdf((hi - cond_mean) / sigma)
            elif j == n - 1:
                P[i, j] = 1.0 - scipy_norm.cdf((lo - cond_mean) / sigma)
            else:
                P[i, j] = (scipy_norm.cdf((hi - cond_mean) / sigma) -
                            scipy_norm.cdf((lo - cond_mean) / sigma))
    return grid, P


# ============================================================
# 1.  Parameters and Setup
# ============================================================
print("=" * 60)
print("  1. Parameters and Setup")
print("=" * 60)
t_start = time.time()

# ── Preference / technology parameters ───────────────────────
beta    = 0.87     # discount factor
delta   = 0.06     # capital depreciation rate
gamma   = 2.0      # coefficient of relative risk aversion
alpha   = 0.35     # capital share
wage    = 1.0      # normalised wage
za      = 1.0      # common TFP level
epsilon = 3.0      # elasticity of substitution (Dixit-Stiglitz)
mu_m    = epsilon / (epsilon - 1.0)   # markup  (called mu in paper)
D       = 1.0      # demand shifter

# ── Overhead labour ───────────────────────────────────────────
phil = 0.00        # baseline;  use 0.135 for robustness check

# ── Permanent productivity (zp): 2 discrete states ───────────
nzp            = 2
zp_sd_target   = 0.43        # target std dev of permanent component
piL            = 0.80        # fraction of firms in low-productivity state

# Solve for zp_l such that Var(zp) = zp_sd_target^2
# Var(zp) = piL*(zp_l - E[zp])^2 + (1-piL)*(zp_h - E[zp])^2
# with E[zp]=1 (normalisation: piL*zp_l + (1-piL)*zp_h = 1)
zLspace   = np.linspace(1e-4, 1 - 1e-4, 1_000_000)
zp_h_vec  = (1.0 - piL * zLspace) / (1.0 - piL)
difference = (piL * (zLspace - 1.0)**2
              + (1.0 - piL) * (zp_h_vec - 1.0)**2
              - zp_sd_target**2)
idx   = np.argmin(np.abs(difference))
zp_l  = zLspace[idx]
zp_h  = (1.0 - piL * zp_l) / (1.0 - piL)
zp_grid = np.array([zp_l, zp_h])
zpprob  = np.eye(nzp)          # permanent → identity (no switching)

print(f"  zp_l = {zp_l:.4f},  zp_h = {zp_h:.4f}")

# ── Exogenous labour wedges (tau): baseline = 1 state ────────
ntau = 1     # set 2 for extension
if ntau == 2:
    tau_grid = np.linspace(-0.29, 0.29, ntau)
    tauprob  = np.array([[0.81, 0.19],
                          [0.19, 0.81]])
else:
    tau_grid = np.array([0.00])
    tauprob  = np.array([[1.0]])

# ── Unmeasured capital (q): baseline = 1 state ───────────────
nq = 1       # set 2 for extension
if nq == 2:
    q_grid = np.linspace(-0.18, 0.18, nq)
    qprob  = np.eye(nq)
else:
    q_grid = np.array([0.00])
    qprob  = np.array([[1.0]])

# ── Transitory log productivity (zt): AR(1), Tauchen grid ────
nzt      = 11
rho_zt   = 0.59
sigma_zt = 0.13
mu_zt    = -(sigma_zt**2) / (2.0 * (1.0 + rho_zt))   # Jensen correction
zt_grid, ztprob = tauchen(nzt, mu_zt, rho_zt, sigma_zt, 3)

# ── Interest rate process ─────────────────────────────────────
# runexp=1: all changes are unexpected (baseline)
# runexp=0: AR(1) process
runexp  = 1
nr      = 6
r_grid  = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.10])

if runexp == 1:
    rprob = np.eye(nr)
else:
    rho_r   = 0.50
    sigma_r = 0.0086
    mu_r    = 0.03 * rho_r
    r_grid_ar, rprob_ar = tauchen(nr - 1, mu_r, rho_r, sigma_r, 2.014)
    r_grid  = np.append(r_grid_ar, r_grid_ar.max())
    rprob   = np.zeros((nr, nr))
    rprob[:nr-1, :nr-1] = rprob_ar
    rprob[nr-1, nr-1]   = 1.0

# ── Financial frictions (collateral constraint) ───────────────
# k' <= chi0*a' + chi1*(exp(k') - 1)
# Baseline HeF:
chi0 = 0.98
chi1 = 0.047
# Homogeneous frictions (HoF):  chi0=1.06, chi1=0.00
# No frictions (NoF):           chi0=1e10,  chi1=0.00

# ── Capital adjustment cost ───────────────────────────────────
psi = 3.2   # baseline HeF

# ── State-space grids for capital (k) and net worth (a) ──────
k_l, k_h, nk = 0.01, 6.0,  120
a_l, a_h, na = 0.01, 3.0,  120
k_grid = np.linspace(k_l, k_h, nk)
a_grid = np.linspace(a_l, a_h, na)

n_choice = nk * na                          # 14 400
n_state  = nzp * nzt * nr * ntau * nq      # 132 (baseline)

print(f"  n_choice = {n_choice},  n_state = {n_state}")

# ── Build exogenous state index arrays ────────────────────────
# Each column of EXOG / EXOG_ind encodes one exogenous state.
# Order of nesting (innermost → outermost): q, tau, r, zt, zp

def _build_exog(grids, sizes):
    """Replicate MATLAB's nested-repmat pattern for exogenous states."""
    # grids = list of 1-D arrays in order [q, tau, r, zt, zp]
    # sizes = [nq, ntau, nr, nzt, nzp]
    n = int(np.prod(sizes))
    arrays = []
    for i, (g, s) in enumerate(zip(grids, sizes)):
        n_inner = int(np.prod(sizes[:i]))   # product of inner sizes
        n_outer = int(np.prod(sizes[i+1:])) # product of outer sizes
        # repeat each element n_inner times, then tile n_outer times
        arr = np.tile(np.repeat(g, n_inner), n_outer)
        arrays.append(arr)
    return arrays

grids_list = [q_grid, tau_grid, r_grid, zt_grid, zp_grid]
sizes_list = [nq,      ntau,     nr,     nzt,     nzp]
Q_arr, TAU_arr, R_arr, ZT_arr, ZP_arr = _build_exog(grids_list, sizes_list)

grids_ind   = [np.arange(s) for s in sizes_list]
Q_ind, TAU_ind, R_ind, ZT_ind, ZP_ind = _build_exog(grids_ind, sizes_list)

# EXOG shape: (5, n_state) — rows: [ZP, ZT, R, TAU, Q]
EXOG     = np.vstack([ZP_arr,  ZT_arr,  R_arr,  TAU_arr,  Q_arr])
EXOG_ind = np.vstack([ZP_ind,  ZT_ind,  R_ind,  TAU_ind,  Q_ind]).astype(int)

# ── Choice-space arrays: all (a', k') combinations ───────────
# Index mapping: iak = ia * nk + ik  (Python 0-based)
A_choice = np.repeat(a_grid, nk)   # a' for each choice index
K_choice = np.tile(k_grid, na)     # k' for each choice index

# ── Collateral value for each choice (depends only on a', k') ─
collateral_choice = chi0 * A_choice + chi1 * (np.exp(K_choice) - 1.0) - K_choice

# ── Broadcast matrices (n_choice × n_state) ───────────────────
AP_mat = np.tile(A_choice[:, None], (1, n_state))
KP_mat = np.tile(K_choice[:, None], (1, n_state))
R_mat  = np.tile(EXOG[2, :][None, :], (n_choice, 1))
Q_mat  = np.tile(EXOG[4, :][None, :], (n_choice, 1))
collateral_mat = np.tile(collateral_choice[:, None], (1, n_state))

# ── Precompute pi and T for each current state (ia, ik, i_state) ─
# l = phil + (za*exp(zt)*zp)^A * (k+q)^B * D^C * mu^(-C) * (wage*(1+tau)/(1-alpha))^(-C)
# where A=(eps-1)/(1+alpha*(eps-1)), B=alpha*A, C=eps/(1+alpha*(eps-1))
A_exp  = (epsilon - 1.0) / (1.0 + alpha * (epsilon - 1.0))
B_exp  = alpha * A_exp
C_exp  = epsilon / (1.0 + alpha * (epsilon - 1.0))

# pi_vfi[iak, i_state], T_vfi[iak, i_state]
pi_vfi = np.zeros((n_choice, n_state))
T_vfi  = np.zeros((n_choice, n_state))

for iak in range(n_choice):
    ik = iak % nk
    ia = iak // nk
    k_cur = k_grid[ik]
    # Effective capital for each exog state
    k_eff = k_cur + EXOG[4, :]          # k + q[i_state]
    prod  = za * np.exp(EXOG[1, :]) * EXOG[0, :]   # za * exp(zt) * zp

    l = (phil
         + prod**A_exp
         * k_eff**B_exp
         * D**C_exp
         * mu_m**(-C_exp)
         * (wage * (1.0 + EXOG[3, :]) / (1.0 - alpha))**(-C_exp))

    y  = prod * k_eff**alpha * (l - phil)**(1.0 - alpha)
    pi = D * y**((epsilon - 1.0) / epsilon) - wage * (1.0 + EXOG[3, :]) * l
    T  = (wage * EXOG[3, :] * l
          + wage * (1.0 + EXOG[3, :]) * phil
          - (EXOG[2, :] + delta) * EXOG[4, :])

    pi_vfi[iak, :] = pi
    T_vfi[iak, :]  = T

# ── Feasibility indicator ─────────────────────────────────────
# For a current state, flag it infeasible if NO choice yields c>0
# (used in Howard's improvement to keep the old value)
indicator = np.ones((n_choice, n_state), dtype=bool)
for i_state in range(n_state):
    iq  = EXOG_ind[4, i_state]
    ir  = EXOG_ind[2, i_state]
    for iak in range(n_choice):
        ik = iak % nk
        ia = iak // nk
        k_cur = k_grid[ik]; a_cur = a_grid[ia]
        # c for all choices given this current state
        c_all = (pi_vfi[iak, i_state] + T_vfi[iak, i_state]
                 - (r_grid[ir] + delta) * k_cur
                 + (1.0 + r_grid[ir]) * a_cur
                 - A_choice
                 - psi * (K_choice - k_cur)**2 / (2.0 * k_cur))
        k_eff_all = K_choice + q_grid[iq]
        feasible  = (c_all > 0) & (collateral_choice >= 0) & (k_eff_all > 0)
        if not np.any(feasible):
            indicator[iak, i_state] = False

# ── Joint transition matrix (n_state × n_state) ───────────────
print("  Building transition matrix ...")
prob = np.zeros((n_state, n_state))
for i in range(n_state):
    izp  = EXOG_ind[0, i]; izt  = EXOG_ind[1, i]
    ir   = EXOG_ind[2, i]; itau = EXOG_ind[3, i]; iq = EXOG_ind[4, i]
    for j in range(n_state):
        izp2 = EXOG_ind[0, j]; izt2 = EXOG_ind[1, j]
        ir2  = EXOG_ind[2, j]; itau2= EXOG_ind[3, j]; iq2 = EXOG_ind[4, j]
        prob[i, j] = (ztprob[izt, izt2]
                      * zpprob[izp, izp2]
                      * rprob[ir,  ir2]
                      * tauprob[itau, itau2]
                      * qprob[iq, iq2])

print(f"  Setup complete in {(time.time()-t_start)/60:.1f} min")


# ============================================================
# 2.  Value Function Iteration
# ============================================================
print()
print("=" * 60)
print("  2. Value Function Iteration")
print("=" * 60)
t_vfi = time.time()

if gamma < 2.5 and runexp == 1:
    penalty = -1e9
else:
    penalty = -1e12

Nh              = 20          # Howard improvement steps (-1 to disable)
error_tolerance = 1e-6

V_old      = penalty * np.ones((n_choice, n_state))
V_new      = np.zeros((n_choice, n_state))
Vh_old     = penalty * np.ones((n_choice, n_state))
Vh_new     = penalty * np.ones((n_choice, n_state))
location   = np.zeros((n_choice, n_state), dtype=int)
pol_ind_kp = np.zeros((n_choice, n_state), dtype=int)
pol_ind_ap = np.zeros((n_choice, n_state), dtype=int)

iter_vfi  = 0
error_vfi = 1e15
VFIcontinue = True

# Utility function: CRRA with γ=2  →  u(c) = (1/|c|^(γ-1) - 1) / (1-γ)
# For γ=2: u(c) = 1/c - 1  scaled; more precisely (c^(1-γ)-1)/(1-γ)
# The MATLAB code uses: ((1./realpow(abs(c),(gamma-1)))-1)/(1-gamma)
# which equals (|c|^(1-gamma) - 1)/(1-gamma) = standard CRRA.

def crra(c):
    """CRRA utility, handles γ=1 (log) as limiting case."""
    if abs(gamma - 1.0) < 1e-10:
        return np.log(np.abs(c))
    return (np.abs(c)**(1.0 - gamma) - 1.0) / (1.0 - gamma)

while VFIcontinue:
    t_iter = time.time()
    error_vfi_old = error_vfi

    # ── Expected value: EXP = V_old @ prob.T  (n_choice × n_state) ──
    EXP = V_old @ prob.T     # shape: (n_choice, n_state)

    # ── Main VFI loop over current states ────────────────────────────
    for iak in range(n_choice):
        ik    = iak % nk
        ia    = iak // nk
        k_cur = k_grid[ik]
        a_cur = a_grid[ia]

        # Consumption for all choices × all exog states
        piplusT = pi_vfi[iak, :] + T_vfi[iak, :]   # (n_state,)
        c = (piplusT[None, :]                        # broadcast over n_choice choices
             - (R_mat + delta) * k_cur
             + (1.0 + R_mat) * a_cur
             - AP_mat
             - psi * (KP_mat - k_cur)**2 / (2.0 * k_cur))

        k_eff = KP_mat + Q_mat

        V_sub = crra(c) + beta * EXP
        infeasible = (c <= 0) | (collateral_mat < 0) | (k_eff <= 0)
        V_sub[infeasible] = penalty

        best_idx          = np.argmax(V_sub, axis=0)   # (n_state,)
        V_new[iak, :]     = V_sub[best_idx, np.arange(n_state)]
        location[iak, :]  = best_idx
        pol_ind_kp[iak, :] = best_idx % nk
        pol_ind_ap[iak, :] = best_idx // nk

    # ── Howard's improvement ──────────────────────────────────────────
    Vh_old = V_new.copy()
    for _ in range(Nh + 1):
        for iak in range(n_choice):
            ik    = iak % nk
            ia    = iak // nk
            k_cur = k_grid[ik]
            a_cur = a_grid[ia]

            kp_idx = pol_ind_kp[iak, :]   # (n_state,)
            ap_idx = pol_ind_ap[iak, :]   # (n_state,)
            kp_val = k_grid[kp_idx]
            ap_val = a_grid[ap_idx]

            c = (pi_vfi[iak, :]
                 + T_vfi[iak, :]
                 - (EXOG[2, :] + delta) * k_cur
                 + (1.0 + EXOG[2, :]) * a_cur
                 - ap_val
                 - psi * (kp_val - k_cur)**2 / (2.0 * k_cur))

            # E[Vh_old | policy]
            loc_row = location[iak, :]    # (n_state,) indices into n_choice
            ev = np.sum(prob * Vh_old[loc_row, :], axis=1)  # (n_state,)

            Vh_new[iak, :] = crra(c) + beta * ev

        # Keep old value for infeasible states
        Vh_new = Vh_new * indicator + Vh_old * (~indicator)
        Vh_old = Vh_new.copy()

    # ── Convergence check ─────────────────────────────────────────────
    if Nh == -1:
        error_vfi = np.max(np.abs(V_new - V_old))
        V_old = V_new.copy()
    else:
        error_vfi = np.max(np.abs(Vh_new - V_old))
        V_old = Vh_new.copy()

    iter_vfi += 1
    elapsed_iter = (time.time() - t_iter) / 3600.0
    print(f"  iter {iter_vfi:4d} | error_old = {error_vfi_old:.3e} "
          f"| error = {error_vfi:.3e} | {elapsed_iter:.3f} h/iter")

    if error_vfi < error_tolerance:
        VFIcontinue = False

time_VFI = (time.time() - t_vfi) / 3600.0
print(f"\n  VFI converged in {iter_vfi} iterations, {time_VFI:.3f} hours")

# ── Reshape policy functions to (na, nk, nzp, nzt, nr, ntau, nq) ──
shape7 = (na, nk, nzp, nzt, nr, ntau, nq)

V_final            = np.full(shape7, np.nan)
a_prime_val        = np.full(shape7, np.nan)
k_prime_val        = np.full(shape7, np.nan)
c_val              = np.full(shape7, np.nan)
binding_collateral = np.full(shape7, np.nan)
sim_pol_ind_ap     = np.full(shape7, np.nan)
sim_pol_ind_kp     = np.full(shape7, np.nan)
effective_lambda   = np.full(shape7, np.nan)

for i_state in range(n_state):
    izp  = EXOG_ind[0, i_state]; izt  = EXOG_ind[1, i_state]
    ir   = EXOG_ind[2, i_state]; itau = EXOG_ind[3, i_state]
    iq   = EXOG_ind[4, i_state]

    for iak in range(n_choice):
        ik = iak % nk
        ia = iak // nk

        idx7 = (ia, ik, izp, izt, ir, itau, iq)
        v = V_new[iak, i_state]
        if v == penalty:
            continue   # leave as NaN

        iap = pol_ind_ap[iak, i_state]
        ikp = pol_ind_kp[iak, i_state]
        kp  = k_grid[ikp]
        ap  = a_grid[iap]

        V_final[idx7]        = v
        sim_pol_ind_ap[idx7] = iap
        sim_pol_ind_kp[idx7] = ikp
        a_prime_val[idx7]    = ap
        k_prime_val[idx7]    = kp

        c_val[idx7] = (pi_vfi[iak, i_state]
                       - (r_grid[ir] + delta) * k_grid[ik]
                       + (1.0 + r_grid[ir]) * a_grid[ia]
                       + T_vfi[iak, i_state]
                       - ap
                       - psi * (kp - k_grid[ik])**2 / (2.0 * k_grid[ik]))

        # Collateral binding?
        if ikp < nk - 1:
            kp_next = k_grid[ikp + 1]
            eff_lam = chi0 + chi1 * (np.exp(kp_next) - 1.0) / ap
            if chi0 > 1e5 or chi1 > 1e5:
                eff_lam = 1e8
            effective_lambda[idx7] = eff_lam
            coll_slack = chi0 * ap + chi1 * (np.exp(kp_next) - 1.0) - kp_next
            binding_collateral[idx7] = 1.0 if coll_slack < 0 else 0.0
        else:
            binding_collateral[idx7] = 0.0

# Convert index arrays to integer (NaN positions remain NaN)
sim_pol_ind_ap = sim_pol_ind_ap.astype(float)
sim_pol_ind_kp = sim_pol_ind_kp.astype(float)


# ============================================================
# 3.  Simulation
# ============================================================
print()
print("=" * 60)
print("  3. Simulation")
print("=" * 60)
t_sim = time.time()

Nfirms  = 50_000
Tperiods = 1_000
Tshock   = 800       # period index (1-based as in MATLAB) → 799 in 0-based

rng = np.random.default_rng(seed=42)
rv_tau = rng.random((Nfirms, Tperiods))
rv_zt  = rng.random((Nfirms, Tperiods))
rv_q   = rng.random((Nfirms, Tperiods))

# ── Assign permanent productivity ────────────────────────────
n_low  = round(Nfirms * piL)
n_high = Nfirms - n_low
sim_i_zp = np.concatenate([np.zeros(n_low, dtype=int),
                             np.ones(n_high, dtype=int)])    # 0-based indices

# ── Assign unmeasured capital group ──────────────────────────
sim_i_q = np.arange(Nfirms) % nq      # cycles through nq groups

# ── Sort firms by (zp, q) group ──────────────────────────────
group_key = sim_i_zp * nq + sim_i_q
sort_order = np.argsort(group_key, kind='stable')
sim_i_zp = sim_i_zp[sort_order]
sim_i_q  = sim_i_q[sort_order]

sim_i_q_mat = np.tile(sim_i_q[:, None], (1, Tperiods))

# ── Labour wedge (tau) Markov chain ──────────────────────────
sim_i_tau = np.zeros((Nfirms, Tperiods), dtype=int)
sim_i_tau[:, 0] = ntau // 2
for it in range(1, Tperiods):
    cumP = np.cumsum(tauprob[sim_i_tau[:, it-1], :], axis=1)  # (Nfirms, ntau)
    sim_i_tau[:, it] = (rv_tau[:, it][:, None] <= cumP).argmax(axis=1)

# ── Transitory productivity (zt) Markov chain ────────────────
sim_i_zt = np.zeros((Nfirms, Tperiods), dtype=int)
sim_i_zt[:, 0] = nzt // 2
for it in range(1, Tperiods):
    cumP = np.cumsum(ztprob[sim_i_zt[:, it-1], :], axis=1)
    sim_i_zt[:, it] = (rv_zt[:, it][:, None] <= cumP).argmax(axis=1)

# ── Interest rate sequence ────────────────────────────────────
# sim_i_r is 0-based; MATLAB had 1-based indices mapped to r_grid
sim_i_r = np.zeros(Tperiods, dtype=int)
sim_i_r[:Tshock-1] = 5   # index 5 → r=0.10 (highest, pre-1994)
shock_seq = [4,4,4,3,2,0,1,1,0,0,0,0,1,2,1,2,1,1]  # 1994-2011 (0-based)
for t_off, idx in enumerate(shock_seq):
    if Tshock - 1 + t_off < Tperiods:
        sim_i_r[Tshock - 1 + t_off] = idx
sim_i_r[Tshock + 17:] = 1   # post-2011 → r=0.02

# ── Initialise firm positions ─────────────────────────────────
a0_init = min(a_grid.max() - 0.10, 1.00)
k0_init = 0.10
if chi0 > 1e5 or chi1 > 1e5:
    k0_init = 0.40
if nq > 1:
    k0_init = max(-q_grid.min() + 0.10, 0.10)

idx_a0 = np.argmin(np.abs(a_grid - a0_init))
idx_k0 = np.argmin(np.abs(k_grid - k0_init))

sim_i_a = np.zeros((Nfirms, Tperiods), dtype=int)
sim_i_k = np.zeros((Nfirms, Tperiods), dtype=int)
sim_i_a[:, 0] = idx_a0
sim_i_k[:, 0] = idx_k0

sim_a_val               = np.zeros((Nfirms, Tperiods))
sim_k_val               = np.zeros((Nfirms, Tperiods))
sim_c_val               = np.zeros((Nfirms, Tperiods))
sim_coll_val            = np.zeros((Nfirms, Tperiods))
sim_eff_lambda_val      = np.zeros((Nfirms, Tperiods))

sim_a_val[:, 0] = a_grid[idx_a0]
sim_k_val[:, 0] = k_grid[idx_k0]

# sim_pol_ind_ap / kp have shape (na, nk, nzp, nzt, nr, ntau, nq)
# We convert to int arrays for indexing (NaN → 0 temporarily)
_pol_ap = np.nan_to_num(sim_pol_ind_ap, nan=0).astype(int)
_pol_kp = np.nan_to_num(sim_pol_ind_kp, nan=0).astype(int)
_c_val  = np.nan_to_num(c_val,           nan=np.nan)
_bind   = np.nan_to_num(binding_collateral, nan=np.nan)
_eflam  = np.nan_to_num(effective_lambda,   nan=np.nan)

print("  Simulating firm paths ...")
for it in range(1, Tperiods):
    ia   = sim_i_a[:, it-1]
    ik   = sim_i_k[:, it-1]
    izp  = sim_i_zp
    izt  = sim_i_zt[:, it-1]
    ir   = sim_i_r[it-1] * np.ones(Nfirms, dtype=int)
    itau = sim_i_tau[:, it-1]
    iq   = sim_i_q_mat[:, it-1]

    sim_i_a[:, it] = _pol_ap[ia, ik, izp, izt, ir, itau, iq]
    sim_i_k[:, it] = _pol_kp[ia, ik, izp, izt, ir, itau, iq]

    ia2  = sim_i_a[:, it]
    ik2  = sim_i_k[:, it]
    izt2 = sim_i_zt[:, it]
    ir2  = sim_i_r[it] * np.ones(Nfirms, dtype=int)
    itau2= sim_i_tau[:, it]
    iq2  = sim_i_q_mat[:, it]

    sim_a_val[:, it]          = a_grid[ia2]
    sim_k_val[:, it]          = k_grid[ik2]
    sim_c_val[:, it]          = _c_val[ia2, ik2, izp, izt2, ir2, itau2, iq2]
    sim_coll_val[:, it]       = _bind [ia2, ik2, izp, izt2, ir2, itau2, iq2]
    sim_eff_lambda_val[:, it] = _eflam[ia2, ik2, izp, izt2, ir2, itau2, iq2]

    if it % 100 == 0:
        print(f"    period {it}/{Tperiods}")

# ── Derived simulation quantities ────────────────────────────
sim_zt_val   = zt_grid[sim_i_zt]
sim_zp_val   = np.tile(zp_grid[sim_i_zp][:, None], (1, Tperiods))
sim_tau_val  = tau_grid[sim_i_tau]
sim_q_val    = q_grid[sim_i_q_mat]
sim_r_val    = r_grid[sim_i_r]          # (Tperiods,)

sim_prod_val = za * np.exp(sim_zt_val) * sim_zp_val

# Optimal labour demand
sim_l_val = (phil
             + sim_prod_val**A_exp
             * (sim_k_val + sim_q_val)**B_exp
             * D**C_exp * mu_m**(-C_exp)
             * (wage * (1.0 + sim_tau_val) / (1.0 - alpha))**(-C_exp))

sim_y_val    = sim_prod_val * (sim_k_val + sim_q_val)**alpha * (sim_l_val - phil)**(1.0 - alpha)
sim_p_val    = D * sim_y_val**(-1.0 / epsilon)
sim_rev_val  = sim_p_val * sim_y_val
sim_mrpk_val = (alpha / mu_m) * sim_p_val * sim_y_val / sim_k_val
sim_mrpl_val = ((1.0 - alpha) / mu_m) * sim_p_val * sim_y_val / sim_l_val
sim_b_val    = sim_k_val - sim_a_val
sim_lev_val  = sim_b_val / sim_k_val

sim_Psi_val  = np.zeros((Nfirms, Tperiods))
for t in range(Tperiods - 1):
    sim_Psi_val[:, t] = (psi / 2.0) * (sim_k_val[:, t+1] - sim_k_val[:, t])**2 / sim_k_val[:, t]
sim_Psi_val[:, -1] = sim_Psi_val[:, -2]

# ── Aggregate statistics (across all Nfirms) ─────────────────
std_mrpk        = np.std(np.log(sim_mrpk_val), axis=0)
std_mrpl        = np.std(np.log(sim_mrpl_val), axis=0)
logTFP_eff      = np.log((np.sum(sim_prod_val**(epsilon - 1.0), axis=0))**(1.0 / (epsilon - 1.0)))
rev_agg         = np.sum(sim_rev_val, axis=0)
yy_agg          = np.sum(sim_y_val, axis=0)
YY_agg          = np.sum(sim_y_val**((epsilon-1.0)/epsilon), axis=0)**(epsilon/(epsilon-1.0))
KK_agg          = np.sum(sim_k_val + sim_q_val, axis=0)
LL_agg          = np.sum(sim_l_val, axis=0) - phil * Nfirms
AA_agg          = np.sum(sim_a_val, axis=0)
CC_agg          = np.sum(sim_c_val, axis=0)
logTFP_obs      = np.log(YY_agg / (KK_agg**alpha * LL_agg**(1.0 - alpha)))
MIS             = logTFP_obs - logTFP_eff
MEAN_coll       = np.mean(sim_coll_val, axis=0)

print(f"\n  Simulation complete in {(time.time()-t_sim)/3600:.3f} hours")


# ============================================================
# 4.  Dataset Construction and Export
# ============================================================
print()
print("=" * 60)
print("  4. Building and saving dataset")
print("=" * 60)

Ndataset  = min(10_000, Nfirms)
Ndataset_start = 0          # 0-based
Ndataset_end   = Ndataset   # exclusive

Tdataset_start = Tshock - 10 - 1   # convert to 0-based
Tdataset_end   = Tshock + 26       # exclusive
Tdataset       = Tdataset_end - Tdataset_start

# Helper: flatten a firm×time slice to a 1-D column (time-major)
def flat(arr2d):
    """Slice arr2d[firms, times] and flatten in time-major order."""
    return arr2d[Ndataset_start:Ndataset_end,
                 Tdataset_start:Tdataset_end].T.ravel()

idn_arr  = np.tile(np.arange(1, Ndataset + 1), Tdataset)
time_arr = np.repeat(np.arange(Tdataset_start + 1, Tdataset_end + 1), Ndataset)

# Aggregate series repeated for each firm
def flat_agg(arr1d):
    return np.tile(arr1d[Tdataset_start:Tdataset_end], Ndataset)

dataset = pd.DataFrame({
    'Tshock':              Tshock,
    'id':                  idn_arr,
    'time':                time_arr,
    'k':                   flat(sim_k_val),
    'a':                   flat(sim_a_val),
    'zt':                  flat(sim_zt_val),
    'zp':                  flat(sim_zp_val),
    'l':                   flat(sim_l_val),
    'y':                   flat(sim_y_val),
    'mrpk':                flat(sim_mrpk_val),
    'rev':                 flat(sim_rev_val),
    'constrained':         flat(sim_coll_val),
    'r':                   flat_agg(sim_r_val),
    'logTFP_observed':     flat_agg(logTFP_obs),
    'MIS':                 flat_agg(MIS),
    'RR':                  flat_agg(rev_agg),
    'yy':                  flat_agg(yy_agg),
    'YY':                  flat_agg(YY_agg),
    'KK':                  flat_agg(KK_agg),
    'LL':                  flat_agg(LL_agg),
    'AA':                  flat_agg(AA_agg),
    'mrpl':                flat(sim_mrpl_val),
    'Psi':                 flat(sim_Psi_val),
    'c':                   flat(sim_c_val),
    'effective_lambda':    flat(sim_eff_lambda_val),
    'CC':                  flat_agg(CC_agg),
    'CONSTRAINED_mean':    flat_agg(MEAN_coll),
    'STD_MRPK':            flat_agg(std_mrpk),
    'q':                   flat(sim_q_val),
})

out_path = 'dataset.xlsx'
dataset.to_excel(out_path, index=False)
print(f"  Saved {len(dataset):,} rows × {len(dataset.columns)} columns → {out_path}")
print(f"\n  Total wall time: {(time.time()-t_start)/3600:.3f} hours")
print("  Done.")