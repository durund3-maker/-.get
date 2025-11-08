# greenwash.py
"""
Greenwash integrated script (updated: psi correction, interior equilibrium printing)

Key changes/additions:
 - psi = (1 - alpha) * (gamma * zpe - zpe**2)  (corrected)
 - produce_symbolic_formulas_plain now prints:
     * corner Jacobians and eigenvalues (symbolic)
     * mixed (interior) equilibrium expressions:
         - expressed in A,B,C,D,E,F notation
         - full expression with A..F substituted (in terms of alpha, phi, V_H, V_L, U0, gamma, s)
         - symbolic Jacobian at the interior and (when possible) symbolic eigenvalues
     * numeric example that evaluates the interior candidate and prints numeric jacobian eigenvalues/stability
 - Avoids SymPy boolean simplification that previously raised "nan is not comparable"
 - **Plot fonts set to Times New Roman and size 小五 (mapped to 9 pt)**
"""
import os
import datetime
import traceback
import math
import numpy as np
from math import sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

# ------------------------
# Set plotting font to Times New Roman, size 小五 (9 pt)
# ------------------------
# In Chinese typesetting, 小五 is commonly mapped to 9 pt. Adjust rcParams so all plots use Times New Roman.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 18,                 # base font size (小五)
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18,
    # ensure math text uses the same (regular) font where possible
    'mathtext.fontset': 'dejavusans',
})

# ------------------------
# CONFIG: all manual parameters centralized here
# ------------------------
CONFIG = {
    # Economic parameters
    "alpha": 0.5,           # cost/interaction parameter
    "phi": 0.5,             # weight for p_pool (显性投资机会权重)
    "V_H": 4.0,
    "V_L": 2.0,
    "U0": 1.0,

    # Monte Carlo / deterministic control
    "default_n_runs": 500,      # reduce for quick tests
    "default_n_types": 5000,    # reduce for quick tests
    "tmax": 200.0,
    "nsteps": 200,

    # solver tolerances
    "rtol": 1e-7,
    "atol": 1e-9,

    # absorption tolerance (distance to corners)
    "eps_absorb": 1e-3,

    # RNG / gamma distribution
    "random_seed": 0,
    "gamma_mu": 0.0,
    "gamma_sigma": 0.5,

    # deterministic analysis s list (the three s to show qualitative changes)
    "s_list_det": [0.1, 0.05, 1],

    # sweep s values for baseline MC
    "s_vals_sweep": np.linspace(0.0, 2.0, 200),

    # outputs dir
    "outdir": "outputs"
}

# Pull config into local variables
OUTDIR = CONFIG["outdir"]
os.makedirs(OUTDIR, exist_ok=True)
alpha = CONFIG["alpha"]
phi_default = CONFIG["phi"]
V_H = CONFIG["V_H"]
V_L = CONFIG["V_L"]
U0 = CONFIG["U0"]
default_n_runs = CONFIG["default_n_runs"]
default_n_types = CONFIG["default_n_types"]
tmax = CONFIG["tmax"]
nsteps = CONFIG["nsteps"]
t_grid = np.linspace(0.0, tmax, nsteps)
rtol = CONFIG["rtol"]
atol = CONFIG["atol"]
eps_absorb = CONFIG["eps_absorb"]
random_seed = CONFIG["random_seed"]
np.random.seed(random_seed)
gamma_mu_default = CONFIG["gamma_mu"]
gamma_sigma_default = CONFIG["gamma_sigma"]
s_list_det = CONFIG["s_list_det"]
s_vals_sweep = CONFIG["s_vals_sweep"]

# ------------------------
# Safe save helpers
# ------------------------
def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_write_text(path, text):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved text to {path}")
        return path
    except PermissionError:
        alt = f"{os.path.splitext(path)[0]}_{timestamp_str()}{os.path.splitext(path)[1]}"
        with open(alt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[warning] PermissionError writing {path}; saved to alternative {alt}")
        return alt

def safe_save_df(df, path, **to_csv_kwargs):
    try:
        df.to_csv(path, index=False, **to_csv_kwargs)
        print(f"Saved DataFrame to {path}")
        return path
    except PermissionError:
        alt = f"{os.path.splitext(path)[0]}_{timestamp_str()}.csv"
        df.to_csv(alt, index=False, **to_csv_kwargs)
        print(f"[warning] PermissionError writing {path}; saved to alternative {alt}")
        return alt

# ------------------------
# Pricing function (uses phi for p_pool)
# ------------------------
def prices_user_formula(alpha_local, phi_local, V_H_local, V_L_local, U0_local):
    p_H = V_H_local - U0_local
    p_L = max(V_L_local - U0_local, 0.0)
    p_pool = phi_local * V_H_local + (1.0 - phi_local) * V_L_local - U0_local
    return p_H, p_L, p_pool

# ------------------------
# A..F calculators (vectorized and single-gamma)
# psi corrected as requested
# ------------------------
def compute_A_to_F_for_gammas(gammas, alpha_local, s, V_H_local, V_L_local, U0_local, phi_local, p_pool_override=None):
    p_H, p_L, p_pool = prices_user_formula(alpha_local, phi_local, V_H_local, V_L_local, U0_local)
    if p_pool_override is not None:
        p_pool = p_pool_override
    gammas = np.asarray(gammas, dtype=float)
    zpe_arr = gammas / 2.0
    arg = (V_H_local - max(V_L_local, U0_local)) / alpha_local
    sep_const = sqrt(arg) if arg > 0 else 0.0
    zsep_arr = np.maximum(zpe_arr, sep_const)
    # psi corrected: (1 - alpha) * (gamma*zpe - zpe^2)
    psi_arr = (1.0 - alpha_local) * (gammas * zpe_arr - zpe_arr**2)
    A_arr = p_H - alpha_local * (s * zsep_arr)**2 + gammas * alpha_local * s * zsep_arr + psi_arr
    B_arr = p_pool - alpha_local * (zpe_arr**2) + gammas * alpha_local * zpe_arr + psi_arr
    C_arr = p_H - alpha_local * (zpe_arr**2) + gammas * alpha_local * zpe_arr + psi_arr
    D_arr = p_L - alpha_local * (zpe_arr**2)
    E_arr = p_pool - alpha_local * (zpe_arr**2)
    F = p_L
    return A_arr, B_arr, C_arr, D_arr, E_arr, F

def compute_A_to_F_single_gamma(gamma_val, alpha_local, s, V_H_local, V_L_local, U0_local, phi_local, p_pool_override=None):
    p_H, p_L, p_pool = prices_user_formula(alpha_local, phi_local, V_H_local, V_L_local, U0_local)
    if p_pool_override is not None:
        p_pool = p_pool_override
    zpe = gamma_val / 2.0
    arg = (V_H_local - max(V_L_local, U0_local)) / alpha_local
    sep_const = math.sqrt(arg) if arg > 0 else 0.0
    zsep = max(zpe, sep_const)
    psi = (1.0 - alpha_local) * (gamma_val * zpe - zpe**2)   # corrected
    A = p_H - alpha_local * (s * zsep)**2 + gamma_val * alpha_local * s * zsep + psi
    B = p_pool - alpha_local * (zpe**2) + gamma_val * alpha_local * zpe + psi
    C = p_H - alpha_local * (zpe**2) + gamma_val * alpha_local * zpe + psi
    D = p_L - alpha_local * (zpe**2)
    E = p_pool - alpha_local * (zpe**2)
    F = p_L
    extras = {'zpe': zpe, 'zsep': zsep, 'psi': psi, 'p_H': p_H, 'p_L': p_L, 'p_pool': p_pool}
    return A, B, C, D, E, F, extras

# ------------------------
# Identify "special" firms (E <= F) but do NOT change p_pool (no automatic adjust)
# returns (p_pool, special_mask)
# ------------------------
def possibly_adjust_p_pool(alpha_local, gammas, s, V_H_local, V_L_local, U0_local, phi_local, enforce=True):
    gammas = np.atleast_1d(np.array(gammas, dtype=float))
    p_H, p_L, p_pool = prices_user_formula(alpha_local, phi_local, V_H_local, V_L_local, U0_local)
    zpe_arr = gammas / 2.0
    E_arr = p_pool - alpha_local * (zpe_arr**2)
    F_val = p_L
    tol = 1e-12
    special_mask = (E_arr <= F_val + tol)
    return p_pool, special_mask

# ------------------------
# ODE solver w/ events for corner absorption
# ------------------------
def _make_corner_event(xc, yc, eps_local):
    def event(t, state):
        x, y = state
        return math.hypot(x - xc, y - yc) - eps_local
    event.terminal = True
    event.direction = 0
    return event

def simulate_replicator_with_events(A_bar, B_bar, C_bar, D_bar, E_bar, F_bar,
                                    y0=(0.3,0.7), t_grid_local=t_grid,
                                    rtol_local=rtol, atol_local=atol, eps_local=eps_absorb):
    def rhs(t, state):
        Xv, Yv = state
        dX = Xv*(1-Xv)*(A_bar - (C_bar + (B_bar - C_bar)*Yv))
        dY = Yv*(1-Yv)*((E_bar + (D_bar - E_bar)*Xv) - F_bar)
        return [dX, dY]

    events = [
        _make_corner_event(0.0, 0.0, eps_local),
        _make_corner_event(1.0, 0.0, eps_local),
        _make_corner_event(0.0, 1.0, eps_local),
        _make_corner_event(1.0, 1.0, eps_local)
    ]
    try:
        sol = solve_ivp(rhs, [t_grid_local[0], t_grid_local[-1]], y0,
                        rtol=rtol_local, atol=atol_local, events=events, dense_output=True)
    except Exception:
        n = len(t_grid_local)
        return np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32), t_grid_local[-1]

    t_end = float(sol.t[-1])
    t_grid_clipped = t_grid_local[t_grid_local <= t_end]
    if t_grid_clipped.size == 0:
        last_state = sol.y[:, -1] if sol.y.size else np.array([np.nan, np.nan])
        X_full = np.full(len(t_grid_local), float(last_state[0]), dtype=np.float32)
        Y_full = np.full(len(t_grid_local), float(last_state[1]), dtype=np.float32)
        return X_full, Y_full, t_end

    try:
        yz = sol.sol(t_grid_clipped)
    except Exception:
        yz = np.vstack([np.interp(t_grid_clipped, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])

    X_full = np.empty(len(t_grid_local), dtype=np.float32)
    Y_full = np.empty(len(t_grid_local), dtype=np.float32)
    X_full[:t_grid_clipped.size] = yz[0, :].astype(np.float32)
    Y_full[:t_grid_clipped.size] = yz[1, :].astype(np.float32)
    last_x = float(yz[0, -1]); last_y = float(yz[1, -1])
    if t_grid_clipped.size < len(t_grid_local):
        X_full[t_grid_clipped.size:] = np.float32(last_x)
        Y_full[t_grid_clipped.size:] = np.float32(last_y)
    return X_full, Y_full, t_end

# ------------------------
# Monte Carlo experiment (baseline only)
# ------------------------
def run_mc_experiments(n_runs=default_n_runs, n_types=default_n_types,
                       alpha_local=alpha, s=0.5, V_H_local=V_H, V_L_local=V_L, U0_local=U0,
                       phi_local=phi_default,
                       gamma_mu=gamma_mu_default, gamma_sigma=gamma_sigma_default,
                       enforce_constraint=True, t_grid_local=t_grid):
    results = []
    all_X_incl = np.empty((n_runs, len(t_grid_local)), dtype=np.float32)
    all_Y_incl = np.empty((n_runs, len(t_grid_local)), dtype=np.float32)
    all_X_excl = np.empty((n_runs, len(t_grid_local)), dtype=np.float32)
    all_Y_excl = np.empty((n_runs, len(t_grid_local)), dtype=np.float32)
    special_fraction_per_run = np.empty(n_runs, dtype=np.float32)

    for run in range(n_runs):
        normals = np.random.normal(loc=gamma_mu, scale=gamma_sigma, size=n_types)
        gammas = np.exp(normals)

        p_pool0, special_mask = possibly_adjust_p_pool(alpha_local, gammas, s, V_H_local, V_L_local, U0_local, phi_local, enforce=enforce_constraint)
        special_fraction = float(np.mean(special_mask))
        special_fraction_per_run[run] = special_fraction

        A_arr, B_arr, C_arr, D_arr, E_arr, F = compute_A_to_F_for_gammas(gammas, alpha_local, s, V_H_local, V_L_local, U0_local, phi_local, p_pool_override=p_pool0)

        # inclusive averages
        A_bar_incl = float(np.mean(A_arr)); B_bar_incl = float(np.mean(B_arr)); C_bar_incl = float(np.mean(C_arr))
        D_bar_incl = float(np.mean(D_arr)); E_bar_incl = float(np.mean(E_arr)); F_bar = float(F)

        # exclusive averages (exclude special)
        non_special_idx = np.where(~special_mask)[0]
        if non_special_idx.size > 0:
            A_bar_excl = float(np.mean(A_arr[non_special_idx])); B_bar_excl = float(np.mean(B_arr[non_special_idx])); C_bar_excl = float(np.mean(C_arr[non_special_idx]))
            D_bar_excl = float(np.mean(D_arr[non_special_idx])); E_bar_excl = float(np.mean(E_arr[non_special_idx]))
            have_excl = True
        else:
            A_bar_excl = B_bar_excl = C_bar_excl = D_bar_excl = E_bar_excl = np.nan
            have_excl = False

        X_incl, Y_incl, t_end_incl = simulate_replicator_with_events(A_bar_incl, B_bar_incl, C_bar_incl, D_bar_incl, E_bar_incl, F_bar,
                                                                     y0=(0.3, 0.7), t_grid_local=t_grid_local)
        if have_excl:
            X_excl, Y_excl, t_end_excl = simulate_replicator_with_events(A_bar_excl, B_bar_excl, C_bar_excl, D_bar_excl, E_bar_excl, F_bar,
                                                                         y0=(0.3, 0.7), t_grid_local=t_grid_local)
        else:
            X_excl = np.full(len(t_grid_local), np.nan, dtype=np.float32)
            Y_excl = np.full(len(t_grid_local), np.nan, dtype=np.float32)
            t_end_excl = np.nan

        all_X_incl[run, :] = X_incl
        all_Y_incl[run, :] = Y_incl
        all_X_excl[run, :] = X_excl
        all_Y_excl[run, :] = Y_excl

        tail = max(1, int(0.1 * len(t_grid_local)))
        X_lr_incl = float(np.nanmean(X_incl[-tail:])) if not np.isnan(X_incl).all() else np.nan
        Y_lr_incl = float(np.nanmean(Y_incl[-tail:])) if not np.isnan(Y_incl).all() else np.nan
        X_lr_excl = float(np.nanmean(X_excl[-tail:])) if not np.isnan(X_excl).all() else np.nan
        Y_lr_excl = float(np.nanmean(Y_excl[-tail:])) if not np.isnan(Y_excl).all() else np.nan

        results.append({
            'run': run,
            'A_bar_incl': A_bar_incl, 'B_bar_incl': B_bar_incl, 'C_bar_incl': C_bar_incl,
            'D_bar_incl': D_bar_incl, 'E_bar_incl': E_bar_incl, 'F_bar': F_bar,
            'X_long_incl': X_lr_incl, 'Y_long_incl': Y_lr_incl,
            'A_bar_excl': A_bar_excl, 'B_bar_excl': B_bar_excl, 'C_bar_excl': C_bar_excl,
            'D_bar_excl': D_bar_excl, 'E_bar_excl': E_bar_excl,
            'X_long_excl': X_lr_excl, 'Y_long_excl': Y_lr_excl,
            'abs_time_incl': t_end_incl, 'abs_time_excl': t_end_excl,
            'special_fraction': special_fraction,
            'p_pool_used': p_pool0
        })

        if (run + 1) % 10 == 0:
            print(f"MC progress: {run+1}/{n_runs} runs done (special_frac={special_fraction:.3f})")

    df_results = pd.DataFrame(results)
    return df_results, all_X_incl, all_Y_incl, all_X_excl, all_Y_excl, special_fraction_per_run, t_grid

# ------------------------
# Deterministic two-player analysis helpers
# ------------------------
def analytic_interior_equilibrium(A,B,C,D,E,F):
    eps = 1e-12
    denomY = B - C
    denomX = D - E
    if abs(denomY) > eps and abs(denomX) > eps:
        Yc = (A - C) / denomY
        Xc = (F - E) / denomX
        if 0.0 <= Xc <= 1.0 and 0.0 <= Yc <= 1.0:
            return (Xc, Yc)
    return None

def analytic_jacobian_and_eigs(A,B,C,D,E,F,X,Y):
    alpha_H = A - C
    beta_H = B - C
    gamma_L = E - F
    delta_L = D - E
    df_dX = (1.0 - 2.0*X) * (alpha_H - beta_H * Y)
    df_dY = X * (1.0 - X) * (-beta_H)
    dg_dX = Y * (1.0 - Y) * (delta_L)
    dg_dY = (1.0 - 2.0*Y) * (gamma_L + delta_L * X)
    J = np.array([[df_dX, df_dY],[dg_dX, dg_dY]], dtype=float)
    eigs = np.linalg.eigvals(J)
    detJ = np.linalg.det(J)
    traceJ = np.trace(J)
    re = np.real(eigs)
    if np.all(re < 0):
        stability = 'asymptotically stable'
    elif np.any(re > 0) and np.any(re < 0):
        stability = 'saddle (unstable)'
    elif np.all(re > 0):
        stability = 'unstable (source)'
    else:
        stability = 'non-hyperbolic/center'
    return J, detJ, traceJ, eigs, stability

def simulate_deterministic(A,B,C,D,E,F, y0=(0.3,0.7), t_grid_local=t_grid):
    def rhs(t, state):
        X, Y = state
        dX = X * (1-X) * (A - (C + (B - C) * Y))
        dY = Y * (1-Y) * ((E + (D - E) * X) - F)
        return [dX, dY]
    try:
        sol = solve_ivp(rhs, [t_grid_local[0], t_grid_local[-1]], y0, t_eval=t_grid_local, rtol=rtol, atol=atol)
        X_t = sol.y[0]; Y_t = sol.y[1]
    except Exception:
        X_t = np.full_like(t_grid_local, np.nan)
        Y_t = np.full_like(t_grid_local, np.nan)
    return X_t, Y_t

def compute_avg_payoffs_time_series(A,B,C,D,E,F, X_t, Y_t):
    EU_H_over_arr = np.full_like(X_t, A, dtype=float)
    EU_H_norm_arr = C + (B - C) * Y_t
    EU_L_green_arr = E + (D - E) * X_t
    EU_L_not_arr = np.full_like(X_t, F, dtype=float)
    H_avg = X_t * EU_H_over_arr + (1 - X_t) * EU_H_norm_arr
    L_avg = Y_t * EU_L_green_arr + (1 - Y_t) * EU_L_not_arr
    total_avg = 0.5 * (H_avg + L_avg)
    tail = max(1, int(0.1 * len(X_t)))
    H_long = np.nanmean(H_avg[-tail:])
    L_long = np.nanmean(L_avg[-tail:])
    tot_long = np.nanmean(total_avg[-tail:])
    return {'H_avg': H_avg, 'L_avg': L_avg, 'total_avg': total_avg, 'H_long': H_long, 'L_long': L_long, 'tot_long': tot_long}

def plot_phase_deterministic(A,B,C,D,E,F, s_val, fname=None, t_grid_local=t_grid):
    xs = np.linspace(0.01, 0.99, 25); ys = np.linspace(0.01, 0.99, 25)
    Xg, Yg = np.meshgrid(xs, ys)
    U = np.zeros_like(Xg); V = np.zeros_like(Yg)
    for i in range(len(xs)):
        for j in range(len(ys)):
            x = Xg[j,i]; y = Yg[j,i]
            U[j,i] = x*(1-x)*(A - (C + (B-C)*y))
            V[j,i] = y*(1-y)*((E + (D-E)*x) - F)
    M = np.hypot(U, V)
    plt.figure(figsize=(6,6))
    M_safe = np.where(M == 0, 1.0, M)
    plt.quiver(Xg, Yg, U/M_safe, V/M_safe, M, pivot='mid', cmap='viridis', scale=30)
    eps = 1e-12
    if abs(B-C) > eps:
        y_null = (A - C) / (B - C)
        plt.axhline(y=y_null, color='red', linestyle='--', label='X-nullcline')
    if abs(D-E) > eps:
        x_null = (F - E) / (D - E)
        plt.axvline(x=x_null, color='blue', linestyle='--', label='Y-nullcline')
    initials = [(0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9),(0.5,0.5)]
    for init in initials:
        try:
            sol = solve_ivp(lambda t, state: [state[0]*(1-state[0])*(A - (C + (B - C)*state[1])),
                                              state[1]*(1-state[1])*((E + (D - E)*state[0]) - F)],
                            [t_grid_local[0], t_grid_local[-1]], init, t_eval=t_grid_local, rtol=rtol, atol=atol)
            plt.plot(sol.y[0,:], sol.y[1,:], '-', linewidth=1)
            plt.plot(sol.y[0,0], sol.y[1,0], 'o')
        except Exception:
            continue
    plt.xlim(0,1); plt.ylim(0,1); plt.xlabel('X (H over)'); plt.ylabel('Y (L green)')
    plt.title(f'Deterministic phase portrait (s={s_val})')
    plt.grid(True)
    if fname is None:
        fname = os.path.join(OUTDIR, f'deterministic_phase_s_{s_val:.2f}.png')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved deterministic phase plot (s={s_val}) to {fname}")

def run_two_player_analysis(gamma=None, s_list=None, alpha_local=alpha, phi_local=phi_default, V_H_local=V_H, V_L_local=V_L, U0_local=U0, enforce_constraint=True):
    if s_list is None:
        s_list = s_list_det
    if gamma is None:
        gamma = math.exp(gamma_mu_default)

    jac_rows = []
    constants_rows = []

    for s in s_list:
        p_pool_try, special_mask_one = possibly_adjust_p_pool(alpha_local, np.array([gamma]), s, V_H_local, V_L_local, U0_local, phi_local, enforce=enforce_constraint)
        A, B, C, D, E, F, extras = compute_A_to_F_single_gamma(gamma, alpha_local, s, V_H_local, V_L_local, U0_local, phi_local, p_pool_override=p_pool_try)

        # simulate deterministic
        X_t, Y_t = simulate_deterministic(A,B,C,D,E,F)
        payoff_ts = compute_avg_payoffs_time_series(A,B,C,D,E,F, X_t, Y_t)

        interior = analytic_interior_equilibrium(A,B,C,D,E,F)
        corners = [(0.0,0.0),(1.0,0.0),(0.0,1.0),(1.0,1.0)]
        for c in corners:
            J, detJ, trJ, eigs, stability = analytic_jacobian_and_eigs(A,B,C,D,E,F, c[0], c[1])
            jac_rows.append({
                's': s, 'eq_point': f'({c[0]},{c[1]})',
                'J11': J[0,0], 'J12': J[0,1], 'J21': J[1,0], 'J22': J[1,1],
                'det': detJ, 'trace': trJ, 'eig1': eigs[0], 'eig2': eigs[1], 'stability': stability
            })
        if interior is not None:
            Xc, Yc = interior
            J, detJ, trJ, eigs, stability = analytic_jacobian_and_eigs(A,B,C,D,E,F, Xc, Yc)
            jac_rows.append({
                's': s, 'eq_point': f'({Xc:.6f},{Yc:.6f})',
                'J11': J[0,0], 'J12': J[0,1], 'J21': J[1,0], 'J22': J[1,1],
                'det': detJ, 'trace': trJ, 'eig1': eigs[0], 'eig2': eigs[1], 'stability': stability
            })

        constants_rows.append({
            's': s, 'gamma': gamma, 'phi': phi_local,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F,
            'zpe': extras['zpe'], 'zsep': extras['zsep'],
            'p_pool_used': extras['p_pool'], 'is_special_gamma': bool(special_mask_one[0]) if special_mask_one.size>0 else False,
            'H_long': payoff_ts['H_long'], 'L_long': payoff_ts['L_long'], 'tot_long': payoff_ts['tot_long']
        })

        # save phase & time series per s
        plot_phase_deterministic(A,B,C,D,E,F, s, fname=os.path.join(OUTDIR, f'deterministic_phase_s_{s:.2f}.png'))
        plt.figure(figsize=(8,4))
        plt.plot(t_grid, X_t, label=f'X (H over), s={s}')
        plt.plot(t_grid, Y_t, label=f'Y (L green), s={s}')
        plt.xlabel('time'); plt.ylabel('proportion'); plt.legend(); plt.grid(True)
        plt.title(f'Deterministic time series (s={s})')
        plt.tight_layout()
        fname_ts = os.path.join(OUTDIR, f'deterministic_time_series_s_{s:.2f}.png')
        plt.savefig(fname_ts, dpi=300)
        plt.close()
        print(f"Saved deterministic time-series (s={s}) to {fname_ts}")

    df_jac = pd.DataFrame(jac_rows)
    df_consts = pd.DataFrame(constants_rows)
    safe_save_df(df_jac, os.path.join(OUTDIR, "deterministic_jacobian_summary_by_s.csv"))
    safe_save_df(df_consts, os.path.join(OUTDIR, "deterministic_constants_by_s.csv"))

    # summary plot comparing long-run X and Y across the s_list
    plt.figure(figsize=(8,4))
    s_vals = [row['s'] for row in constants_rows]
    X_longs = [row['H_long'] for row in constants_rows]
    Y_longs = [row['L_long'] for row in constants_rows]
    plt.plot(s_vals, X_longs, marker='o', label='H long-run (avg payoff proxy)')
    plt.plot(s_vals, Y_longs, marker='o', label='L long-run')
    plt.xlabel('s'); plt.ylabel('long-run avg'); plt.legend(); plt.grid(True)
    plt.title('Deterministic long-run payoffs by s (comparison)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "deterministic_longrun_by_s_summary.png"), dpi=300)
    plt.close()

    print("Deterministic analysis (multiple s) saved to outputs/")
    return {'df_jac': df_jac, 'df_consts': df_consts}

# ------------------------
# Symbolic formulas with corner eigen expressions (and numeric example)
#    Added: explicit interior / mixed-equilibrium expressions (A..F form and full param form)
#    AND: symbolic interior Jacobian & eigenvalues (attempted; fallbacks on failure)
# ------------------------
def produce_symbolic_formulas_plain(outdir=OUTDIR):
    # symbols
    alpha_s, phi_s, V_H_s, V_L_s, U0_s, gamma_s, s_s, X_s, Y_s = sp.symbols(
        'alpha phi V_H V_L U0 gamma s X Y', real=True
    )

    # pricing / thresholds symbolic
    p_H = V_H_s - U0_s
    p_L = sp.Max(V_L_s - U0_s, 0)
    p_pool = phi_s * V_H_s + (1 - phi_s) * V_L_s - U0_s

    zpe = gamma_s / 2
    sep_const = sp.sqrt((V_H_s - sp.Max(V_L_s, U0_s)) / alpha_s)
    zsep = sp.Max(zpe, sep_const)
    # psi corrected symbolically
    psi = (1 - alpha_s) * (gamma_s * zpe - zpe**2)

    # A..F symbolically
    A = sp.simplify(p_H - alpha_s * (s_s * zsep)**2 + gamma_s * alpha_s * s_s * zsep + psi)
    B = sp.simplify(p_pool - alpha_s * (zpe**2) + gamma_s * alpha_s * zpe + psi)
    C = sp.simplify(p_H - alpha_s * (zpe**2) + gamma_s * alpha_s * zpe + psi)
    D = sp.simplify(p_L - alpha_s * (zpe**2))
    E = sp.simplify(p_pool - alpha_s * (zpe**2))
    F = sp.simplify(p_L)

    # replicator dynamics symbolic (in terms of A..F)
    X, Y = X_s, Y_s
    dX = sp.simplify(X * (1 - X) * (A - (C + (B - C) * Y)))
    dY = sp.simplify(Y * (1 - Y) * ((E + (D - E) * X) - F))

    # Jacobian entries symbolic (in terms of A..F)
    J11 = sp.simplify(sp.diff(dX, X))
    J12 = sp.simplify(sp.diff(dX, Y))
    J21 = sp.simplify(sp.diff(dY, X))
    J22 = sp.simplify(sp.diff(dY, Y))
    J = sp.Matrix([[J11, J12], [J21, J22]])

    # eigenvalues at corners: compute and simplify (symbolically)
    corners = [(0,0),(1,0),(0,1),(1,1)]
    corner_eigs = {}
    for (xc,yc) in corners:
        Js = sp.simplify(J.subs({X: xc, Y: yc}))
        try:
            eigs = sp.simplify(sp.Matrix(Js).eigenvals())
            eig_list = list(eigs.keys())
        except Exception:
            # fallback: keep Jacobian matrix but avoid eigen decomposition failure
            eig_list = []
        corner_eigs[(xc,yc)] = (sp.simplify(Js), [sp.simplify(e) for e in eig_list])

    # interior / mixed equilibrium symbolic expressions
    # In A..F notation:
    Xc_AoF = sp.simplify((F - E) / (D - E))  # Xc in terms of A..F
    Yc_AoF = sp.simplify((A - C) / (B - C))
    # Also produce the "full" parameter-substituted expressions
    Xc_full = sp.simplify(sp.factor(Xc_AoF))
    Yc_full = sp.simplify(sp.factor(Yc_AoF))

    # prepare human-readable text
    lines = []
    lines.append("# Explicit formulas (Python-like) and symbolic corner eigen expressions")
    lines.append("# Generated: " + datetime.datetime.now().isoformat())
    lines.append("")
    lines.append("## Pricing & thresholds (python-style):")
    lines.append("p_H = V_H - U0")
    lines.append("p_L = max(V_L - U0, 0)")
    lines.append("p_pool = phi * V_H + (1 - phi) * V_L - U0")
    lines.append("")
    lines.append("zpe = gamma / 2")
    lines.append("sep_const = sqrt((V_H - max(V_L, U0)) / alpha)")
    lines.append("zsep = max(zpe, sep_const)")
    lines.append("psi = (1 - alpha) * (gamma * zpe - zpe**2)   # corrected")
    lines.append("")
    lines.append("A = p_H - alpha * (s * zsep)**2 + gamma * alpha * s * zsep + psi")
    lines.append("B = p_pool - alpha * zpe**2 + gamma * alpha * zpe + psi")
    lines.append("C = p_H - alpha * zpe**2 + gamma * alpha * zpe + psi")
    lines.append("D = p_L - alpha * zpe**2")
    lines.append("E = p_pool - alpha * zpe**2")
    lines.append("F = p_L")
    lines.append("")
    lines.append("replicator: dX = X(1-X)(A - (C + (B - C)Y)), dY = Y(1-Y)((E + (D - E)X) - F)")
    lines.append("")
    lines.append("## Symbolic Jacobian entries (in terms of A..F):")
    try:
        lines.append("J11 = " + sp.pretty(J11))
        lines.append("J12 = " + sp.pretty(J12))
        lines.append("J21 = " + sp.pretty(J21))
        lines.append("J22 = " + sp.pretty(J22))
    except Exception:
        # fallback if pretty fails
        lines.append("J11 symbolic (too complex to pretty-print)")
        lines.append("J12 symbolic (too complex to pretty-print)")
        lines.append("J21 symbolic (too complex to pretty-print)")
        lines.append("J22 symbolic (too complex to pretty-print)")
    lines.append("")

    lines.append("## Corner Jacobians and their eigenvalues (symbolic simplified where possible):")
    for (xc,yc), (Js_sym, eigs_sym) in corner_eigs.items():
        lines.append(f"Corner ({xc},{yc}): Jacobian =")
        try:
            lines.append(sp.pretty(Js_sym))
        except Exception:
            lines.append(str(Js_sym))
        if eigs_sym:
            for i, ev in enumerate(eigs_sym):
                try:
                    lines.append(f"eig_{i+1} = {sp.pretty(ev)}")
                except Exception:
                    lines.append(f"eig_{i+1} = {str(ev)}")
        else:
            lines.append("# eigenvalues symbolic decomposition unavailable or too complex")
        lines.append("")

    # --- Mixed / interior equilibrium presentation ---
    lines.append("## Mixed (interior) equilibrium expressions:")
    lines.append("# (A..F notation)")
    lines.append("Xc (in A..F) = (F - E) / (D - E)")
    lines.append("Yc (in A..F) = (A - C) / (B - C)")
    lines.append("")
    lines.append("# (Full parameter-substituted expressions; these may include Max(...) from p_L or zsep definitions)")
    try:
        lines.append("Xc (full) = " + sp.pretty(Xc_full))
        lines.append("Yc (full) = " + sp.pretty(Yc_full))
    except Exception:
        lines.append("Xc (full) (too complex to pretty-print)")
        lines.append("Yc (full) (too complex to pretty-print)")
    lines.append("")
    lines.append("# Note: interior existence requires denominators (B - C) and (D - E) not equal to zero, and resulting Xc,Yc in [0,1].")
    lines.append("# For safety, the symbolic file avoids boolean simplifications; numeric checks provided below.")
    lines.append("")

    # --- Symbolic Jacobian & eigenvalues at interior (attempt) ---
    lines.append("## Symbolic Jacobian at interior candidate and symbolic eigenvalues (attempted):")
    try:
        # attempt to construct the Jacobian substituted at the symbolic interior
        Js_interior_sym = sp.simplify(J.subs({X: Xc_AoF, Y: Yc_AoF}))
        # try to compute eigenvalues symbolically
        try:
            eigs_interior_dict = sp.Matrix(Js_interior_sym).eigenvals()
            eigs_interior_list = list(eigs_interior_dict.keys())
            lines.append("Interior Jacobian (symbolic):")
            try:
                lines.append(sp.pretty(Js_interior_sym))
            except Exception:
                lines.append(str(Js_interior_sym))
            if len(eigs_interior_list) > 0:
                for i, ev in enumerate(eigs_interior_list):
                    try:
                        lines.append(f"eig_{i+1} (symbolic) = {sp.pretty(sp.simplify(ev))}")
                    except Exception:
                        lines.append(f"eig_{i+1} (symbolic) = {str(ev)}")
            else:
                lines.append("# no symbolic eigenvalues returned (empty dict)")
        except Exception as sube:
            # eigen decomposition failed symbolically
            lines.append("# Could not compute symbolic eigenvalues at interior (decomposition failed or expression too complex).")
            try:
                lines.append("Interior Jacobian (symbolic, best-effort):")
                lines.append(sp.pretty(Js_interior_sym))
            except Exception:
                lines.append(str(Js_interior_sym))
            lines.append(f"# Sympy error detail: {str(sube)}")
    except Exception as e:
        lines.append("# Error when attempting to build/inspect symbolic interior Jacobian: " + str(e))
    lines.append("")

    # --- Numeric example substitution using current CONFIG values --- (and numeric interior check)
    lines.append("## Numeric example using CONFIG values and numeric eigenvalue check for interior candidate")
    alpha_val = float(alpha)
    phi_val = float(phi_default)
    V_H_val = float(V_H)
    V_L_val = float(V_L)
    U0_val = float(U0)
    # representative gamma chosen as exp(mu) (median of lognormal)
    gamma_val = float(math.exp(gamma_mu_default))
    s_list = list(s_list_det)

    lines.append(f"# numeric params: alpha={alpha_val}, phi={phi_val}, V_H={V_H_val}, V_L={V_L_val}, U0={U0_val}, gamma_rep={gamma_val}")
    lines.append("")

    def numeric_corner_eigs(A_val,B_val,C_val,D_val,E_val,F_val):
        # J entries as numeric functions
        def jac_at(xc,yc):
            alpha_H = A_val - C_val
            beta_H = B_val - C_val
            gamma_L = E_val - F_val
            delta_L = D_val - E_val
            df_dX = (1.0 - 2.0*xc) * (alpha_H - beta_H * yc)
            df_dY = xc * (1.0 - xc) * (-beta_H)
            dg_dX = yc * (1.0 - yc) * (delta_L)
            dg_dY = (1.0 - 2.0*yc) * (gamma_L + delta_L * xc)
            Jnum = np.array([[df_dX, df_dY],[dg_dX, dg_dY]], dtype=float)
            eigs = np.linalg.eigvals(Jnum)
            return Jnum, eigs
        corners_list = [(0.0,0.0),(1.0,0.0),(0.0,1.0),(1.0,1.0)]
        out = {}
        for (xc,yc) in corners_list:
            Jnum, eigs = jac_at(xc,yc)
            out[(xc,yc)] = (Jnum, eigs)
        return out

    for s_val in s_list:
        lines.append(f"--- s = {s_val} ---")
        A_val, B_val, C_val, D_val, E_val, F_val, extras = compute_A_to_F_single_gamma(gamma_val, alpha_val, s_val, V_H_val, V_L_val, U0_val, phi_val)
        lines.append(f"A = {A_val:.6g}, B = {B_val:.6g}, C = {C_val:.6g}, D = {D_val:.6g}, E = {E_val:.6g}, F = {F_val:.6g}")
        lines.append(f"(zpe={extras['zpe']:.6g}, zsep={extras['zsep']:.6g}, psi={extras['psi']:.6g}, p_pool={extras['p_pool']:.6g})")
        out = numeric_corner_eigs(A_val,B_val,C_val,D_val,E_val,F_val)
        for (xc,yc), (Jnum, eigs) in out.items():
            eigs_str = ", ".join([f"{ev.real:.6g}{('+'+str(ev.imag)+'j') if abs(ev.imag)>1e-12 else ''}" for ev in eigs])
            lines.append(f"Corner ({xc},{yc}) numeric J = {np.array2string(Jnum, precision=4, separator=', ')}, eigs = [{eigs_str}]")

        # interior numeric check
        denomY_num = B_val - C_val
        denomX_num = D_val - E_val
        interior_possible = False
        Xc_num = None; Yc_num = None
        if abs(denomY_num) > 1e-12 and abs(denomX_num) > 1e-12:
            Yc_num = (A_val - C_val) / denomY_num
            Xc_num = (F_val - E_val) / denomX_num
            if math.isfinite(Xc_num) and math.isfinite(Yc_num) and (0.0 <= Xc_num <= 1.0) and (0.0 <= Yc_num <= 1.0):
                interior_possible = True

        if interior_possible:
            # numeric jacobian & eigenvalues at the interior candidate
            Jnum, detJ, trJ, eigs_val, stab = analytic_jacobian_and_eigs(A_val,B_val,C_val,D_val,E_val,F_val, Xc_num, Yc_num)
            lines.append(f"Interior candidate numeric: Xc={Xc_num:.6g}, Yc={Yc_num:.6g}")
            lines.append(f"Interior numeric Jacobian = {np.array2string(Jnum, precision=4, separator=', ')}")
            eigs_list = [complex(ev) for ev in eigs_val]
            eigs_str = ", ".join([f"{ev.real:.6g}{('+'+str(ev.imag)+'j') if abs(ev.imag)>1e-12 else ''}" for ev in eigs_list])
            lines.append(f"Interior eigenvalues numeric: [{eigs_str}], stability = {stab}")
        else:
            lines.append("No valid interior equilibrium in [0,1] for this numeric parameter set (denoms or values invalid).")
        lines.append("")

    # write to file
    txt = "\n".join(lines)
    txtpath = os.path.join(outdir, "formulas_plain.txt")
    saved = safe_write_text(txtpath, txt)
    return txt, saved

# ------------------------
# sweep s baseline (include & exclude special) - unchanged
# ------------------------
def sweep_s_stability_baseline(s_values, n_runs=default_n_runs, n_types=default_n_types, outdir=OUTDIR, phi_local=phi_default):
    rows = []
    plot_data = {
        's': [], 'Y_mean_incl': [], 'Y_lo_incl': [], 'Y_hi_incl': [],
        'Y_mean_excl': [], 'Y_lo_excl': [], 'Y_hi_excl': [],
        'X_mean_incl': [], 'X_lo_incl': [], 'X_hi_incl': [],
        'X_mean_excl': [], 'X_lo_excl': [], 'X_hi_excl': [],
        'special_frac_mean': []
    }

    print(f"Baseline sweep: phi={phi_local}, gamma_mu={gamma_mu_default}, gamma_sigma={gamma_sigma_default}")
    for s in s_values:
        df_results, X_incl, Y_incl, X_excl, Y_excl, special_fracs, tgrid = run_mc_experiments(
            n_runs=n_runs, n_types=n_types,
            alpha_local=alpha, s=s, V_H_local=V_H, V_L_local=V_L, U0_local=U0,
            phi_local=phi_local, gamma_mu=gamma_mu_default, gamma_sigma=gamma_sigma_default,
            enforce_constraint=True, t_grid_local=t_grid
        )

        # Inclusive stats
        Y_vals_incl = df_results['Y_long_incl'].dropna().values if 'Y_long_incl' in df_results.columns else df_results['Y_long_incl'].dropna().values
        X_vals_incl = df_results['X_long_incl'].dropna().values

        # Exclusive stats
        Y_vals_excl = df_results['Y_long_excl'].dropna().values
        X_vals_excl = df_results['X_long_excl'].dropna().values

        def stats_arr(arr):
            if arr.size == 0:
                return np.nan, np.nan, np.nan
            return float(np.nanmean(arr)), float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))

        y_mean_incl, y_lo_incl, y_hi_incl = stats_arr(Y_vals_incl)
        y_mean_excl, y_lo_excl, y_hi_excl = stats_arr(Y_vals_excl)
        x_mean_incl, x_lo_incl, x_hi_incl = stats_arr(X_vals_incl)
        x_mean_excl, x_lo_excl, x_hi_excl = stats_arr(X_vals_excl)

        special_frac_mean = float(np.nanmean(special_fracs)) if special_fracs.size>0 else np.nan

        rows.append({
            'scenario': 'baseline', 's': s,
            'Y_mean_incl': y_mean_incl, 'Y_lo_incl': y_lo_incl, 'Y_hi_incl': y_hi_incl,
            'Y_mean_excl': y_mean_excl, 'Y_lo_excl': y_lo_excl, 'Y_hi_excl': y_hi_excl,
            'X_mean_incl': x_mean_incl, 'X_lo_incl': x_lo_incl, 'X_hi_incl': x_hi_incl,
            'X_mean_excl': x_mean_excl, 'X_lo_excl': x_lo_excl, 'X_hi_excl': x_hi_excl,
            'special_frac_mean': special_frac_mean,
            'phi': phi_local
        })

        plot_data['s'].append(s)
        plot_data['Y_mean_incl'].append(y_mean_incl); plot_data['Y_lo_incl'].append(y_lo_incl); plot_data['Y_hi_incl'].append(y_hi_incl)
        plot_data['Y_mean_excl'].append(y_mean_excl); plot_data['Y_lo_excl'].append(y_lo_excl); plot_data['Y_hi_excl'].append(y_hi_excl)
        plot_data['X_mean_incl'].append(x_mean_incl); plot_data['X_lo_incl'].append(x_lo_incl); plot_data['X_hi_incl'].append(x_hi_incl)
        plot_data['X_mean_excl'].append(x_mean_excl); plot_data['X_lo_excl'].append(x_lo_excl); plot_data['X_hi_excl'].append(x_hi_excl)
        plot_data['special_frac_mean'].append(special_frac_mean)

        print(f"  s={s:.3f} Y_mean_incl={y_mean_incl:.3f} Y_mean_excl={y_mean_excl:.3f} X_mean_incl={x_mean_incl:.3f} special_frac_mean={special_frac_mean:.3f}")

    df_all = pd.DataFrame(rows)
    safe_save_df(df_all, os.path.join(outdir, "sweep_results_baseline_include_exclude.csv"))

    # Plot Y include vs exclude
    plt.figure(figsize=(10,6))
    s_arr = np.array(plot_data['s'])
    plt.plot(s_arr, plot_data['Y_mean_incl'], label='Y mean (include all firms)', color='tab:blue')
    plt.fill_between(s_arr, plot_data['Y_lo_incl'], plot_data['Y_hi_incl'], color='tab:blue', alpha=0.15)
    plt.plot(s_arr, plot_data['Y_mean_excl'], label='Y mean (exclude special firms)', color='tab:orange', linestyle='--')
    plt.fill_between(s_arr, plot_data['Y_lo_excl'], plot_data['Y_hi_excl'], color='tab:orange', alpha=0.15)
    plt.xlabel('s'); plt.ylabel('Y long-run mean'); plt.title('Baseline: Y long-run vs s (include vs exclude special firms)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sweep_Y_vs_s_include_exclude.png"), dpi=300)
    plt.close()

    # Plot X include vs exclude
    plt.figure(figsize=(10,6))
    plt.plot(s_arr, plot_data['X_mean_incl'], label='X mean (include all firms)', color='tab:green')
    plt.fill_between(s_arr, plot_data['X_lo_incl'], plot_data['X_hi_incl'], color='tab:green', alpha=0.15)
    plt.plot(s_arr, plot_data['X_mean_excl'], label='X mean (exclude special firms)', color='tab:red', linestyle='--')
    plt.fill_between(s_arr, plot_data['X_lo_excl'], plot_data['X_hi_excl'], color='tab:red', alpha=0.15)
    plt.xlabel('s'); plt.ylabel('X long-run mean'); plt.title('Baseline: X long-run vs s (include vs exclude special firms)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sweep_X_vs_s_include_exclude.png"), dpi=300)
    plt.close()

    # Plot special fraction
    plt.figure(figsize=(8,4))
    plt.plot(s_arr, plot_data['special_frac_mean'], label='mean special fraction', marker='o')
    plt.xlabel('s'); plt.ylabel('mean fraction of special firms'); plt.title('Mean special fraction vs s')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "special_fraction_vs_s.png"), dpi=300)
    plt.close()

    return df_all

# ------------------------
# main orchestration
# ------------------------
def main_all():
    try:
        # produce formulas (symbolic + numeric example)
        produce_symbolic_formulas_plain(OUTDIR)

        # PART A: deterministic two-player (s list)
        print("=== PART A: Deterministic analysis for s in", s_list_det, "===")
        run_two_player_analysis(gamma=math.exp(gamma_mu_default), s_list=s_list_det, alpha_local=alpha, phi_local=phi_default, V_H_local=V_H, V_L_local=V_L, U0_local=U0, enforce_constraint=True)

        # PART B: baseline Monte Carlo
        print("=== PART B: Baseline Monte Carlo (may take time) ===")
        df_results, X_incl, Y_incl, X_excl, Y_excl, special_fracs, _ = run_mc_experiments(
            n_runs=default_n_runs, n_types=default_n_types,
            alpha_local=alpha, s=0.5, V_H_local=V_H, V_L_local=V_L, U0_local=U0,
            phi_local=phi_default,
            gamma_mu=gamma_mu_default, gamma_sigma=gamma_sigma_default,
            enforce_constraint=True, t_grid_local=t_grid
        )
        safe_save_df(df_results, os.path.join(OUTDIR, "mc_longrun_results_baseline_include_exclude.csv"))

        print("Running baseline s-sweep (include/exclude special firms)...")
        df_sweep = sweep_s_stability_baseline(s_vals_sweep, n_runs=default_n_runs, n_types=default_n_types, outdir=OUTDIR, phi_local=phi_default)
        print("All done. Check outputs/ for CSVs and PNGs.")

    except Exception as e:
        print("Error in main_all():", e)
        traceback.print_exc()

if __name__ == "__main__":
    main_all()
