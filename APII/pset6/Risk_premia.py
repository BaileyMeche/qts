"""
Risk_premia.py
==============
Implements risk premia estimation methods from Giglio & Xiu (2021):
"Asset Pricing with Omitted Factors"

Three estimators are provided:
  1. FM()            — Standard Fama-MacBeth two-pass cross-sectional regression
  2. PCAFM()         — Three-pass estimator (PCA + FM + time-series regression)
  3. main_simulation() — Monte Carlo simulation wrapper

Helper utilities:
  - vec()            — Column-major vectorization (MATLAB-style)
  - divisorGenerator() — Integer divisor generator
"""
from __future__ import annotations


import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power as mpower

from numpy import diag, sqrt, sort, argsort, log, isnan, var, std, nanmean, kron
from numpy.random import randn, choice, random
from numpy.linalg import norm, inv, lstsq, svd, eig
from numpy.matlib import repmat

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import numpy as np

Array = np.ndarray
ArrayLike = Union[np.ndarray]


# =============================================================================
# vec() — Column-major vectorization
# =============================================================================

def vec(a):
    """
    Vectorize a matrix in column-major (Fortran) order, matching MATLAB's vec().

    Parameters
    ----------
    a : np.ndarray
        Input matrix of shape (m, n).

    Returns
    -------
    np.ndarray
        Flattened 1D array of length m*n, stacked column by column.
    """
    return a.flatten('F')


# =============================================================================
# divisorGenerator() — Yield all divisors of n in ascending order
# =============================================================================

def divisorGenerator(n):
    """
    Generate all divisors of n in ascending order.

    Uses a square-root sweep: yields small divisors immediately and
    collects large divisors for reverse output at the end.
    If n is a perfect square, sqrt(n) is yielded only once.

    Parameters
    ----------
    n : int
        Positive integer whose divisors are generated.

    Yields
    ------
    int or float
        Divisors of n from smallest to largest.
    """
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


# =============================================================================
# FM() — Standard Fama-MacBeth two-pass regression
# =============================================================================

def FM(rt, vt, gt, q):
    """
    Fama-MacBeth two-pass cross-sectional regression for risk premia,
    using the TRUE latent factors vt (for benchmarking purposes).

    This function serves as a baseline comparison for the three-pass
    estimator (PCAFM). Because the true factors are observed, there is
    no omitted variable bias — but this is not realistic in practice.

    Steps
    -----
    1. Time-series regression: regress demeaned returns on demeaned factors
       to obtain factor loadings (betas).
    2. Cross-sectional regression (FM): regress average returns on betas
       to obtain latent factor risk premia (Gammatilde).
    3. Time-series projection: regress observable factor proxies g_t on
       the true factors to obtain eta and the cleaned factor (Ghat).
    4. Combine: risk premia of observables = eta @ Gammatilde.
    5. Newey-West HAC estimation of the asymptotic variance.

    Parameters
    ----------
    rt : np.ndarray, shape (n, T)
        Excess returns for n assets over T periods.
    vt : np.ndarray, shape (p, T)
        True latent factors (known here for benchmarking).
    gt : np.ndarray, shape (d, T)
        Observable factor proxies.
    q  : int or float
        Number of lags for Newey-West HAC standard errors.

    Returns
    -------
    res : dict
        'betahat'  : (n, p) estimated factor loadings
        'Gammahat' : (d, 1) estimated risk premia of observable factors
        'avarhat'  : (d,)   diagonal of asymptotic variance matrix
    """
    n, T = rt.shape
    d = gt.shape[0]
    p = vt.shape[0]

    # --- Demean returns and factors ---
    rtbar = rt - np.mean(rt, axis=1).reshape(-1, 1)  # (n, T)
    vtbar = vt - np.mean(vt, axis=1).reshape(-1, 1)  # (p, T)
    gtbar = gt - np.mean(gt, axis=1).reshape(-1, 1)  # (d, T)

    # --- Step 1: Time-series regression ---
    # betahat = Cov(r, v) * Var(v)^{-1}  =>  (n, p)
    betahat = rtbar @ vtbar.T @ inv(vtbar @ vtbar.T)

    # --- Step 2: Cross-sectional regression (Fama-MacBeth) ---
    # Gammatilde = (beta'beta)^{-1} beta' rbar  =>  (p, 1)
    rbar = np.mean(rt, axis=1).reshape(-1, 1)
    Gammatilde = inv(betahat.T @ betahat) @ betahat.T @ rbar

    # --- Step 3: Time-series regression of g_t on v_t ---
    # etahat: how observable factors load on the true latent factors  =>  (d, p)
    etahat = gtbar @ vtbar.T @ inv(vtbar @ vtbar.T)
    gthat = etahat @ vtbar   # cleaned factor proxies  (d, T)
    what = gtbar - gthat     # residual (measurement error)  (d, T)

    # --- Step 4: Risk premia of observable factors ---
    # Gammahat_g = eta * Gammatilde  =>  (d, 1)
    Gammahat = etahat @ Gammatilde

    # --- Step 5: Newey-West HAC estimation of asymptotic variance ---
    # Estimate Pi11, Pi12, Pi22 matrices for the sandwich formula
    Pi11hat = np.zeros([d * p, d * p])
    Pi12hat = np.zeros([d * p, p])
    Pi22hat = np.zeros([p, p])

    for t in range(T):
        # Outer product terms for the contemporaneous part
        wv_t = vec(what[:, t].reshape(-1, 1) @ vtbar[:, t].reshape(1, -1))  # vec(w_t * v_t')

        Pi11hat += wv_t.reshape(-1, 1) @ wv_t.reshape(1, -1) / T
        Pi12hat += wv_t.reshape(-1, 1) @ vtbar[:, t].reshape(1, -1) / T
        Pi22hat += vtbar[:, t].reshape(-1, 1) @ vtbar[:, t].reshape(1, -1) / T

        # Newey-West lag correction terms (Bartlett kernel)
        for s in range(1, int(min(t, q)) + 1):
            bartlett_weight = 1 - s / (q + 1)

            wv_ts = vec(what[:, t - s].reshape(-1, 1) @ vtbar[:, t - s].reshape(1, -1))

            Pi11hat += (1 / T) * bartlett_weight * (
                wv_t.reshape(-1, 1) @ wv_ts.reshape(1, -1) +
                wv_ts.reshape(-1, 1) @ wv_t.reshape(1, -1)
            )
            Pi12hat += (1 / T) * bartlett_weight * (
                wv_t.reshape(-1, 1) @ vtbar[:, t - s].reshape(1, -1) +
                wv_ts.reshape(-1, 1) @ vtbar[:, t].reshape(1, -1)
            )
            Pi22hat += (1 / T) * bartlett_weight * (
                vtbar[:, t].reshape(-1, 1) @ vtbar[:, t - s].reshape(1, -1) +
                vtbar[:, t - s].reshape(-1, 1) @ vtbar[:, t].reshape(1, -1)
            )

    # --- Asymptotic variance (sandwich formula) ---
    # phi = (Gamma' ⊗ I_d) Pi11 (Gamma ⊗ I_d) / T
    #     + (Gamma' ⊗ I_d) Pi12 eta' / T
    #     + transpose of above
    #     + eta Pi22 eta' / T
    Id = diag(np.ones(d))
    kron_Gamma_Id = kron(Gammatilde.T, Id)

    avarhat = diag(
        kron_Gamma_Id @ Pi11hat @ kron(Gammatilde, Id) / T
        + kron_Gamma_Id @ Pi12hat @ etahat.T / T
        + (kron_Gamma_Id @ Pi12hat @ etahat.T).T / T
        + etahat @ Pi22hat @ etahat.T / T
    )

    # --- Package results ---
    res = {
        'betahat': betahat,
        'Gammahat': Gammahat,
        'avarhat': avarhat,
    }
    return res


# =============================================================================
# PCAFM() — Three-pass estimator (Giglio-Xiu)
# =============================================================================

# Risk_premia.py
# Three-pass PCA + Fama–MacBeth (PCAFM) estimator (Giglio & Xiu, 2021)
#
# Implements Algorithm 1 as summarized in Giglio–Xiu–Zhang (2025), Appendix:
# S1: PCA via SVD on demeaned returns R̄ -> V = sqrt(T)*xi (right singular vectors)
# S2: Fama–MacBeth step to estimate premia of V
# S3: Time-series regression of g_t on V to map into risk premia for g
# Output: V, eta, gamma, and lambda_g = eta @ gamma
#
# Source summary of the three steps: :contentReference[oaicite:1]{index=1}



@dataclass
class PCAFMOutput:
    # Core outputs
    V: Array               # (T, p) latent factors from PCA
    beta: Array            # (N, p) test asset betas on V
    gamma: Array           # (p,) risk premia of V
    eta: Array             # (K, p) loadings of g on V
    lambda_g: Array        # (K,) estimated risk premia for g

    # Inference (optional)
    gamma_t: Optional[Array] = None      # (T, p) time series of cross-sectional premia
    gamma_se: Optional[Array] = None     # (p,) HAC s.e. for gamma
    lambda_se: Optional[Array] = None    # (K,) bootstrap s.e. for lambda_g
    lambda_t: Optional[Array] = None     # (K,) t-stats for lambda_g


def _ensure_NT(R: Array) -> Tuple[Array, bool]:
    """
    Convert R to shape (N, T).
    Accepts (T, N) or (N, T). Returns (R_NT, transposed_flag).
    """
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError("R must be 2D, either (T,N) or (N,T).")

    a, b = R.shape
    if a <= b:
        return R, False
    return R.T, True


def _coerce_R_G(R: ArrayLike, G: ArrayLike) -> Tuple[Array, Array]:
    """Return (R_NT, G_TK) with robust orientation handling."""
    R = np.asarray(R)
    if R.ndim != 2:
        raise ValueError("R must be 2D, either (T,N) or (N,T).")

    G = np.asarray(G)

    candidates = [R, R.T]  # (N,T) candidates

    def _try_G_with_T(T: int) -> Optional[Array]:
        if G.ndim == 1:
            return G[:, None] if G.shape[0] == T else None
        if G.ndim != 2:
            return None
        if G.shape[0] == T:
            return G
        if G.shape[1] == T:
            return G.T
        return None

    for R_NT in candidates:
        T = R_NT.shape[1]
        G_TK = _try_G_with_T(T)
        if G_TK is not None:
            return R_NT, G_TK

    # fallback: use heuristic on R and then parse G against resulting T
    R_NT, _ = _ensure_NT(R)
    T = R_NT.shape[1]
    G_TK = _try_G_with_T(T)
    if G_TK is None:
        raise ValueError("Could not align R and G dimensions; expected shared time dimension T.")
    return R_NT, G_TK


def _row_demean(A: Array) -> Tuple[Array, Array]:
    """
    Demean each row across columns: Ā = A - mean(A, axis=1).
    Returns (demeaned, row_means).
    """
    mu = A.mean(axis=1, keepdims=True)
    return A - mu, mu.squeeze(axis=1)


def _nw_long_run_cov(X: Array, L: int) -> Array:
    """
    Newey–West long-run covariance for sqrt(T)*mean(X_t).
    X: (T,k) time series.
    Returns S such that Var(mean(X)) ≈ S / T.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]

    T, k = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    S = (Xc.T @ Xc) / T
    L = int(max(0, L))
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (Xc[ell:].T @ Xc[:-ell]) / T
        S += w * (Gamma + Gamma.T)
    return S


def _auto_nw_lag(T: int) -> int:
    # Rule-of-thumb bandwidth (Andrews-type): floor(4*(T/100)^(2/9))
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def _pca_V_from_Rbar(Rbar: Array, p: int, sign_normalize: bool = True) -> Array:
    """
    Step 1 (Algorithm 1): SVD on R̄ (N×T). Let xi be first p right singular vectors.
    Return V = sqrt(T)*xi so V'V = T I_p.
    """
    N, T = Rbar.shape
    if not (1 <= p <= min(N, T)):
        raise ValueError(f"p must be in [1, min(N,T)] = [1, {min(N,T)}].")

    # Economy SVD: Rbar = U diag(s) Vt
    U, s, Vt = np.linalg.svd(Rbar, full_matrices=False)
    xi = Vt.T[:, :p]         # (T,p) right singular vectors
    V = np.sqrt(T) * xi      # (T,p)

    if sign_normalize:
        # deterministic sign convention (doesn't change invariants)
        for j in range(p):
            if V[0, j] < 0:
                V[:, j] *= -1.0
    return V


def PCAFM(
    R: ArrayLike,
    G: ArrayLike,
    p: int,
    nw_lag: Optional[int] = None,
    *,
    demean: bool = True,
    bootstrap_B: int = 0,
    bootstrap_block: Optional[int] = None,
    seed: int = 0,
) -> PCAFMOutput:
    """
    Three-pass PCA + Fama–MacBeth estimator.

    Inputs
    ------
    R: test asset excess returns, (T,N) or (N,T).
    G: observed factor(s) g_t, (T,K) or (K,T) or (T,).
    p: number of PCA factors.

    Returns
    -------
    PCAFMOutput with (V, beta, gamma, eta, lambda_g) and optional inference.
    """
    R_NT, G_TK = _coerce_R_G(R, G)
    N, T = R_NT.shape

    K = G_TK.shape[1]

    # Demean if requested
    if demean:
        Rbar, rbar = _row_demean(R_NT)             # R̄ is N×T, rbar is N
        Gbar_TK = G_TK - G_TK.mean(axis=0, keepdims=True)
    else:
        Rbar = R_NT.copy()
        rbar = R_NT.mean(axis=1)
        Gbar_TK = G_TK.copy()

    # --- Step 1: PCA ---
    V = _pca_V_from_Rbar(Rbar, p)                  # (T,p)
    VVT_inv = np.linalg.solve(V.T @ V, np.eye(p))  # (p,p)

    # --- Step 2: Fama–MacBeth-style premia of V ---
    beta = (Rbar @ V) @ VVT_inv                   # (N,p)
    BtB = beta.T @ beta                           # (p,p)

    # gamma from average returns (equivalently average of gamma_t)
    gamma = np.linalg.solve(BtB, beta.T @ rbar)   # (p,)

    # gamma_t time series for HAC se
    gamma_t = np.empty((T, p))
    for t in range(T):
        gamma_t[t, :] = np.linalg.solve(BtB, beta.T @ R_NT[:, t])

    if nw_lag is None:
        nw_lag = _auto_nw_lag(T)

    Sg = _nw_long_run_cov(gamma_t, nw_lag)        # cov of sqrt(T)*mean(gamma_t)
    gamma_se = np.sqrt(np.diag(Sg) / T)

    # --- Step 3: regress g on V ---
    # eta = Ḡ V (V'V)^(-1), where Ḡ is K×T
    eta = (Gbar_TK.T @ V) @ VVT_inv               # (K,p)

    # final premia for g
    lambda_g = eta @ gamma                         # (K,)

    out = PCAFMOutput(
        V=V, beta=beta, gamma=gamma, eta=eta, lambda_g=lambda_g,
        gamma_t=gamma_t, gamma_se=gamma_se
    )

    # Optional bootstrap for lambda_g (captures covariance across steps)
    if bootstrap_B and bootstrap_B > 0:
        rng = np.random.default_rng(seed)
        if bootstrap_block is None:
            bootstrap_block = max(1, int(nw_lag) + 1)

        def draw_block_indices() -> np.ndarray:
            idx = []
            while len(idx) < T:
                start = int(rng.integers(0, T))
                idx.extend(list(range(start, min(start + bootstrap_block, T))))
            return np.array(idx[:T], dtype=int)

        lam_boot = np.empty((bootstrap_B, K))
        for b in range(bootstrap_B):
            tidx = draw_block_indices()
            Rb = R_NT[:, tidx]
            Gb = G_TK[tidx, :]
            lam_boot[b, :] = PCAFM(Rb, Gb, p, demean=demean, nw_lag=nw_lag, bootstrap_B=0).lambda_g

        lambda_se = lam_boot.std(axis=0, ddof=1)
        out.lambda_se = lambda_se
        out.lambda_t = lambda_g / lambda_se

    return out


def _mimicking_portfolio_lambda(R_NT: Array, G_TK: Array, nw_lag: int) -> Tuple[Array, Array]:
    """Simple sample mimicking-portfolio estimate and NW s.e."""
    N, T = R_NT.shape
    K = G_TK.shape[1]
    Rbar, _ = _row_demean(R_NT)
    Srr = (Rbar @ Rbar.T) / T

    lam = np.zeros(K)
    se = np.zeros(K)
    for k in range(K):
        gk = G_TK[:, k]
        gkbar = gk - gk.mean()
        srg = (Rbar @ gkbar) / T
        w = np.linalg.solve(Srr, srg)
        ghat_t = w @ R_NT
        lam[k] = ghat_t.mean()
        S = _nw_long_run_cov(ghat_t, nw_lag)
        se[k] = np.sqrt(np.squeeze(S) / T)
    return lam, se

# =============================================================================
# main_simulation() — Monte Carlo simulation
# =============================================================================

def main_simulation(T, N):
    """
    Monte Carlo simulation comparing FM, three-pass, and MP estimators.

    Generates synthetic asset returns from a linear factor model with
    calibrated parameters, then estimates risk premia using each method
    across M = 1000 Monte Carlo replications.

    Parameters
    ----------
    T : list of int
        List of sample sizes (time periods) to simulate, e.g. [240].
    N : list of int
        List of cross-section sizes (number of assets), e.g. [200].

    Returns
    -------
    output : dict
        Contains MC distributions of estimators:
        'GammaFM'      : (d, M)           FM estimates across MC runs
        'avarhatFM'     : (d, M)           FM avar across MC runs
        'Gammahat'      : (d, pmax, M)     Three-pass estimates
        'avarhat'       : (d, pmax, M)     Three-pass avar
        'gtMP'          : (d, M)           MP estimates
        'gtavarMP'      : (d, M)           MP avar
        'gthat'         : (d, T, pmax, M)  Cleaned factors
        'Gammatrue'     : (d, 1)           True risk premia (for comparison)
        'GammaTrueFM'   : (d, M)           FM with true factors
        'd'             : int              Number of observable factor proxies
        'p'             : int              Number of true latent factors

    Side Effects
    ------------
    Saves a pickle file: 'py_Simulation_Results_T_{T}_N_{N}.pkl'
    """

    # --- Simulation grid ---
    Tlist = T
    Nlist = N

    for tt in range(len(Tlist)):
        for nn in range(len(Nlist)):

            # --- Configuration ---
            M = 1000            # Number of Monte Carlo replications
            T = Tlist[tt]       # Number of time periods
            n = Nlist[nn]       # Number of assets (stocks/portfolios)
            p = 5               # Number of true latent factors
            d = 4               # Number of observable factor proxies
            pmax = 10           # Maximum number of factors to estimate
            q = np.floor(T ** (1 / 4)) + 1  # Newey-West lag truncation

            # --- Load calibrated DGP parameters ---
            param = pd.read_pickle('Calibrated_Parameters.pkl')
            Sigmabeta = param['Sigmabeta']      # (p, p)  covariance of betas
            beta0 = param['beta0']              # (p, 1)  mean of betas
            Sigmau = param['Sigmau']             # (n, n)  idiosyncratic error covariance
            eta = param['eta']                   # (d, p)  factor proxy loadings on latent factors
            gamma = param['gamma']               # (p, 1)  true latent factor risk premia
            sigmaalpha = param['sigmaalpha']      # (1, 1)  pricing error std dev
            Sigmaw = param['Sigmaw']             # (d, d)  measurement error covariance
            Sigmav = param['Sigmav']             # (p, p)  factor innovation covariance

            # --- Allocate storage for MC results ---
            mu = np.zeros([p, 1])                # Mean of latent factors (zero)
            gamma0 = 0                           # Zero-beta rate (set to 0)
            alpha0 = 0                           # Mean pricing error

            Gammahat = np.zeros([d, pmax, M])
            GammaFM = np.zeros([d, M])
            GammaTrueFM = np.zeros([d, M])

            avarhat = np.zeros([d, pmax, M])
            avarhatFM = np.zeros([d, M])

            gthat = np.zeros([d, T, pmax, M])
            gtMP = np.zeros([d, M])
            gtavarMP = np.zeros([d, M])

            phat = np.zeros([3, M])
            gtavar = np.zeros([d, T, M])
            gttrue = np.zeros([d, T, M])

            xi = np.zeros([d, 1])                # Mean of factor proxies (zero)

            # --- Generate cross-sectional structure (fixed across MC) ---
            dlist = [int(x) for x in divisorGenerator(n)]
            dtemp = [abs(x - np.floor(np.sqrt(n))) for x in dlist]
            idx = dtemp.index(min(dtemp))

            # Factor loadings: beta_i ~ N(beta0, Sigmabeta) for each asset
            beta = (np.repeat(beta0.T, n, axis=0) +
                    randn(n, p) @ mpower((Sigmabeta - beta0 @ beta0.T), 0.5))  # (n, p)

            # True risk premia of observable factors: Gamma_true = eta * gamma
            Gammatrue = (eta @ gamma).reshape(-1, 1)  # (d, 1)

            # Unconditional return covariance: Sigma = beta Sigmav beta' + Sigmau
            Sigma = beta @ Sigmav @ beta.T + Sigmau  # (n, n)

            # Optimal MP weights (for reference): lambda* = eta Sigmav beta' Sigma^{-1} beta gamma
            lambdastar = eta @ Sigmav @ beta.T @ inv(Sigma) @ (beta @ gamma)  # (d, 1)

            # --- Monte Carlo loop ---
            for iMC in range(M):
                print('Currently on #', iMC + 1, '/', M, end='\r', flush=True)

                # Generate random data for this MC replication
                alpha = alpha0 + randn(n, 1) @ sigmaalpha               # (n, 1) pricing errors
                vt = mpower(Sigmav, 0.5) @ randn(p, T)                  # (p, T) factor innovations
                ut = mpower(Sigmau, 0.5) @ randn(n, T)                  # (n, T) idiosyncratic errors
                wt = mpower(Sigmaw[:d, :d], 0.5) @ randn(d, T)          # (d, T) measurement errors

                # Simulated returns: r_it = gamma0 + alpha_i + beta_i' gamma + beta_i' v_t + u_it
                rt = (np.repeat(np.ones([n, 1]) * gamma0 + alpha, T, axis=1) +
                      np.repeat(beta @ gamma, T, axis=1) + beta @ vt + ut)  # (n, T)

                # Simulated factor proxies: g_t = xi + eta * v_t + w_t
                gt = np.repeat(xi, T, axis=1) + eta @ vt + wt           # (d, T)
                gttrue[:, :, iMC] = eta @ vt                             # (d, T) true signal

                # --- Estimate risk premia ---
                resFMvsMP = FM(rt, vt, gt, q)           # FM with true factors (benchmark)
                GammaTrueFM[:, iMC] = resFMvsMP['Gammahat'].flatten()
                avarhatFM[:, iMC] = resFMvsMP['avarhat'].flatten()

                # FM using observable proxies only
                GammaFM[:, iMC] = FM(rt, gt, gt, q)['Gammahat'].flatten()

                # Three-pass for p = 1,...,pmax
                for pp in range(1, pmax + 1):
                    res_pp = PCAFM(rt, gt, pp, q)
                    Gammahat[:, pp - 1, iMC] = res_pp.lambda_g
                    cov_lam = res_pp.eta @ _nw_long_run_cov(res_pp.gamma_t, int(q)) @ res_pp.eta.T / T
                    avarhat[:, pp - 1, iMC] = np.diag(cov_lam)
                    gthat[:, :, pp - 1, iMC] = res_pp.eta @ res_pp.V.T

                # Mimicking-portfolio benchmark
                mp_lam, mp_se = _mimicking_portfolio_lambda(rt, gt.T, int(q))
                gtMP[:, iMC] = mp_lam
                gtavarMP[:, iMC] = mp_se ** 2

                phat[:, iMC] = np.array([p, p, p])

    # --- Save and return results ---
    output_list = ['GammaFM', 'avarhatFM', 'Gammahat', 'avarhat', 'gtMP', 'gtavarMP',
                   'gthat', 'Gammatrue', 'GammaTrueFM', 'd', 'p']

    output = {}
    for varname in output_list:
        output[varname] = eval(varname)

    pd.to_pickle(output, 'py_Simulation_Results_T_' + str(T) + '_N_' + str(n) + '.pkl')

    return output
