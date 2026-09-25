"""Exact, deterministic copula estimators (replaces the pair-subsampled estimators; added 2026-09-24).

Earlier scripts estimated the mean tetrachoric correlation from a random subsample of 200-400 model pairs, solved
each pair to xtol=1e-3, and clamped unsolvable pairs to 0.0 or 0.95. That made headline ratios depend on the
subsample (e.g. MATH-500: 2.50 vs 2.44 from two scripts). Everything here uses ALL pairs, solves each pair to
machine precision, and reports how many pairs sit on a boundary instead of silently clamping them.

  bvn_cdf(h, k, r)          P(X<h, Y<k) for a standard bivariate normal with correlation r (vectorized)
  tetrachoric_matrix(W)     m x m latent correlations reproducing every pair's joint wrong-rate, plus flags
  single_factor_beta(t, r)  P(all wrong) under a one-factor Gaussian copula with thresholds t, loading sqrt(r)
  full_sigma_beta(S, t, n)  Monte-Carlo P(all wrong) under MVN(0, S), with its standard error

W is a models x queries 0/1 matrix of WRONG indicators (1 = wrong).
"""
import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from scipy.integrate import quad

_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
R_MAX = 1 - 1e-9


def bvn_cdf(h, k, r):
    """Standard bivariate normal CDF, vectorized over broadcastable h, k, r with |r| < 1.
    Uses Phi2 = Phi(h)Phi(k) + (1/2pi) int_0^{asin r} exp(-(h^2 - 2hk sin t + k^2) / (2 cos^2 t)) dt,
    a smooth integrand after the sin-substitution, integrated by 96-point Gauss-Legendre."""
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    a = np.arcsin(np.clip(r, -R_MAX, R_MAX))
    th = 0.5 * a[..., None] * (_GL_X + 1.0)              # nodes mapped to [0, a]
    s, c2 = np.sin(th), np.cos(th) ** 2
    hh, kk = h[..., None], k[..., None]
    f = np.exp(-(hh * hh - 2 * hh * kk * s + kk * kk) / (2 * c2))
    integral = 0.5 * a * (f * _GL_W).sum(-1)
    return ndtr(h) * ndtr(k) + integral / (2 * np.pi)


def tetrachoric_pairs(p11, ti, tj, iters=80):
    """Solve bvn_cdf(ti, tj, r) = p11 for r by vectorized bisection (the CDF is strictly increasing in r).
    Returns (r, boundary) where boundary = -1/+1 marks pairs whose p11 lies outside the attainable range
    (r pinned at -R_MAX/+R_MAX) and 0 marks interior solutions."""
    p11, ti, tj = np.broadcast_arrays(np.asarray(p11, float), np.asarray(ti, float), np.asarray(tj, float))
    lo = np.full(p11.shape, -R_MAX); hi = np.full(p11.shape, R_MAX)
    f_lo = bvn_cdf(ti, tj, lo) - p11; f_hi = bvn_cdf(ti, tj, hi) - p11
    boundary = np.where(f_lo >= 0, -1, np.where(f_hi <= 0, 1, 0))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = bvn_cdf(ti, tj, mid) - p11
        go_up = f_mid < 0
        lo = np.where(go_up, mid, lo); hi = np.where(go_up, hi, mid)
    r = 0.5 * (lo + hi)
    r = np.where(boundary == -1, -R_MAX, np.where(boundary == 1, R_MAX, r))
    return r, boundary


def thresholds(W):
    """per-model latent thresholds t_i = Phi^{-1}(P(model i wrong))."""
    pw = W.mean(1)
    if np.any((pw <= 0) | (pw >= 1)):
        raise ValueError("a model is always right or always wrong; its threshold is infinite")
    return norm.ppf(pw)


def tetrachoric_matrix(W):
    """All-pairs latent correlation matrix for a models x queries wrong-matrix. Returns (S, t, info)."""
    W = np.asarray(W, float); m = W.shape[0]
    t = thresholds(W)
    P11 = (W @ W.T) / W.shape[1]                         # joint wrong-rates, all pairs at once
    iu = np.triu_indices(m, 1)
    r, b = tetrachoric_pairs(P11[iu], t[iu[0]], t[iu[1]])
    S = np.eye(m); S[iu] = r; S[(iu[1], iu[0])] = r
    info = {"n_pairs": int(len(r)), "n_boundary_low": int((b == -1).sum()), "n_boundary_high": int((b == 1).sum())}
    return S, t, info


def mean_offdiag(S, idx=None):
    if idx is not None:
        S = S[np.ix_(idx, idx)]
    m = S.shape[0]; iu = np.triu_indices(m, 1)
    return float(S[iu].mean())


def single_factor_beta(t, rho):
    """P(all wrong) = int phi(z) prod_i Phi((t_i - sqrt(rho) z) / sqrt(1 - rho)) dz, by adaptive quadrature on
    the log-integrand (accurate for large pools where the integrand is sharply peaked)."""
    t = np.asarray(t, float); rho = float(np.clip(rho, 1e-12, 1 - 1e-12))
    a, b = np.sqrt(rho), np.sqrt(1 - rho)

    def logf(z):
        return norm.logpdf(z) + np.sum(norm.logcdf((t - a * z) / b))
    zs = np.linspace(-12, 12, 4801); lv = np.array([logf(z) for z in zs])
    zmax = zs[lv.argmax()]; peak = lv.max()
    val, err = quad(lambda z: np.exp(logf(z) - peak), -14, 14, points=[zmax], limit=500, epsabs=0, epsrel=1e-10)
    return float(np.exp(peak) * val)


def full_sigma_beta(S, t, n_draws=4_000_000, chunk=100_000, seed=20260924):
    """Monte-Carlo P(all wrong) under MVN(0, S): wrong_i iff latent_i < t_i. Returns (beta, standard_error)."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(S)
    hits = done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        Z = rng.standard_normal((b, S.shape[0])) @ L.T
        hits += int(np.all(Z < t[None, :], axis=1).sum()); done += b
    p = hits / n_draws
    return p, float(np.sqrt(p * (1 - p) / n_draws))


def nearest_psd_correlation(S, eps=1e-4):
    """clip eigenvalues at eps and rescale to unit diagonal (the projection used since v1)."""
    w, V = np.linalg.eigh(S)
    S2 = (V * np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(S2))
    return S2 / np.outer(d, d), int((w < 0).sum())
