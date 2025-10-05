# statistical_tests.py
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.sandwich_covariance import cov_hac_simple

def orthogonal_complement(X_full: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Orthonormal basis W for the part of span(X_full) orthogonal to span([const, Z])."""
    T = X_full.shape[0]
    if Z.size == 0:
        Q, R = np.linalg.qr(X_full, mode="reduced")
        keep = (np.abs(np.diag(R)) > 1e-10)
        return Q[:, keep]
    Zc = np.column_stack([np.ones(T), Z])
    Pz = Zc @ np.linalg.pinv(Zc)
    X_perp = X_full - Pz @ X_full
    Q, R = np.linalg.qr(X_perp, mode="reduced")
    keep = (np.abs(np.diag(R)) > 1e-10)
    return Q[:, keep]

def wald_block_NW(y: np.ndarray, Z: np.ndarray, W: np.ndarray, lags: int = 18):
    """Wald χ² that coeffs on W are zero in y ~ const + Z + W with Newey–West(HAC, lags)."""
    T = len(y)
    X_aug = np.column_stack([np.ones(T), Z, W])
    mdl = sm.OLS(y, X_aug).fit()
    rob = mdl.get_robustcov_results(cov_type="HAC", maxlags=lags)

    k = X_aug.shape[1]
    r = W.shape[1]
    if r == 0:
        return 0.0, 0, 1.0
    R = np.zeros((r, k));  R[:, -r:] = np.eye(r)
    Rb = R @ rob.params
    RSRT = R @ rob.cov_params() @ R.T
    chi2_stat = float(Rb.T @ np.linalg.inv(RSRT) @ Rb)
    pval = 1.0 - chi2.cdf(chi2_stat, df=r)
    return chi2_stat, r, pval





