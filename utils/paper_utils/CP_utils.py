import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from numpy.linalg import qr
from typing import Optional
from statsmodels.stats.sandwich_covariance import cov_hac_simple
from utils.yield_processing import calculate_forward_rates, calculate_excess_returns, create_cochrane_piazzesi_factor
from utils.factor_models import extract_factors_pca
from scipy.stats import chi2
from utils.statistical_tests import (
    orthogonal_complement,
    wald_block_NW,
)
from statsmodels.tsa.api import VAR
import io
from contextlib import redirect_stdout



def cp_tent_shaped_regression(yields, fig_title: str = None):
    # Make a copy and basic checks
    yields = yields.copy().sort_index()
    assert all(c in yields.columns for c in ["y1", "y2", "y3", "y4", "y5"])

    # Use common helpers: forward rates and 12-month excess returns
    mats = [1, 2, 3, 4, 5]
    forwards = calculate_forward_rates(yields, mats)  # f1 == y1 by convention
    rx_all = calculate_excess_returns(yields, maturities = mats, horizon=12, risk_free_col="y1")
    rx = rx_all[[f'xr{n}' for n in [2,3,4,5]]] 
    # Align predictors at t with rx at t+12; drop rows with missing values
    valid_idx = rx.dropna().index.intersection(forwards.dropna().index)
    X = forwards.loc[valid_idx].copy()
    # Explicit y1 column (same as f1) to ensure consistent regressor naming
    X["y1"] = yields.loc[valid_idx, "y1"]

    # Regressor order for plotting: [y1, f2, f3, f4, f5]
    regressors = [c for c in ["y1", "f2", "f3", "f4", "f5"] if c in X.columns]


    coefs_unres = {}
    for n in [2, 3, 4, 5]:
        y = rx.loc[valid_idx, f"xr{n}"]  # NOTE: calculate_excess_returns uses 'xr{n}'
        Xn = sm.add_constant(X[regressors])
        model = sm.OLS(y, Xn, missing="drop").fit()
        coefs_unres[n] = model.params[regressors].values  # store only slopes

    # Restricted single-factor (Cochrane–Piazzesi style):
    # Step 1: regress avg xr across maturities on [const, y1, f2..f5] to get φ
    rx_bar = rx.loc[valid_idx, [f"xr{n}" for n in [2, 3, 4, 5]]].mean(axis=1)
    X_bar = sm.add_constant(X[regressors])
    model_bar = sm.OLS(rx_bar, X_bar, missing="drop").fit()
    alpha_hat = model_bar.params["const"]
    phi_hat = model_bar.params[regressors].values  # in order [y1, f2, f3, f4, f5]

    # Step 2: factor series z_t = α + sum_k φ_k * regressor_k
    z = alpha_hat + (X[regressors] @ phi_hat)

    # Step 3: for each n, regress xr^n on z_t to get b_n
    b_hat = {}
    for n in [2, 3, 4, 5]:
        y = rx.loc[valid_idx, f"xr{n}"]
        Zn = sm.add_constant(pd.Series(z, index=z.index, name="cp_factor"))
        model_bn = sm.OLS(y, Zn, missing="drop").fit()
        b_hat[n] = float(model_bn.params["cp_factor"])

    # Step 4: plot implied slopes = b_n * φ (exclude intercept)
    order = [5, 4, 3, 2]  # plot longer maturities first for visual match
    x = np.array([1, 2, 3, 4, 5])  # 1=y1, 2..5=f2..f5
    coefs_res = {n: b_hat[n] * phi_hat for n in [2, 3, 4, 5]}

    # Plot panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Top: Unrestricted
    for n in order:
        axes[0].plot(x, coefs_unres[n], marker="o", label=f"{n}")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Unrestricted regressions: slopes on [y1, f2, f3, f4, f5]")
    axes[0].set_xlabel("Maturity of forward (1=y1, 2=f2, 3=f3, 4=f4, 5=f5)")
    axes[0].set_ylabel("Coefficient")
    axes[0].legend(title="Bond maturity (years)")

    # Bottom: Restricted implied slopes
    for n in order:
        axes[1].plot(x, coefs_res[n], marker="o", label=f"{n}")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Restricted single-factor: implied slopes b_n × φ")
    axes[1].set_xlabel("Maturity of forward (1=y1, 2=f2, 3=f3, 4=f4, 5=f5)")
    axes[1].set_ylabel("Coefficient")
    axes[1].legend(title="Bond maturity (years)")

    if fig_title:
        fig.suptitle(fig_title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    plt.show()



def replicate_cp_table5(yields_df: pd.DataFrame,
                        horizon_months: int = 12,
                        max_k: int = 3,
                        hac_lags: int = 18,
                        tol: float = 1e-10,
                        max_iter: int = 200):

    # --- Coerce index to month-end timestamps (unchanged) ---
    y = yields_df.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y.index = y.index.to_timestamp(how="end")
    else:
        y.index = pd.to_datetime(y.index) + pd.offsets.MonthEnd(0)
    y = y.sort_index()

    # --- Predictor block f_t = [y1, f2..f5] via common helper ---
    mats = [1, 2, 3, 4, 5]
    forwards = calculate_forward_rates(y, mats)    # returns columns like f1..f5, with f1 == y1
    # Build the regressor frame with names used downstream
    f = pd.DataFrame(index=y.index, dtype=float)
    if 'y1' in y.columns:
        f['y1'] = y['y1']
    # Only keep forward columns that exist (robust to missing maturities)
    for k in [2, 3, 4, 5]:
        col = f"f{k}"
        if col in forwards.columns:
            f[col] = forwards[col]
    reg_names = [c for c in ["y1", "f2", "f3", "f4", "f5"] if c in f.columns]

    # --- One-year-ahead average excess return via common helper ---
    # calculate_excess_returns returns columns named 'xr{n}'
    rx_all = calculate_excess_returns(y, maturities = mats, horizon=12, risk_free_col="y1")
    rx = rx_all[[f'xr{n}' for n in [2,3,4,5]]] 
    # Be defensive: only average the xr columns that exist
    xr_cols = [c for c in [f"xr{n}" for n in [2, 3, 4, 5]] if c in rx.columns]
    if not xr_cols:
        raise ValueError("No excess return columns available to form rx_bar")
    rx_bar = rx[xr_cols].mean(axis=1)

    # --- Lagged predictor blocks F_lags[j] (unchanged) ---
    F_lags = {0: f.copy()}
    for j in range(1, max_k + 1):
        F_lags[j] = f.shift(j)

    def _valid_index_for_k(k: int):
        idx = rx_bar.dropna().index
        for j in range(0, k + 1):
            idx = idx.intersection(F_lags[j].dropna().index)
        return idx

    def _fit_restricted_k(k: int):
        """
        Solve y = c + sum_{j=0}^k α_j (F_j @ γ) + u, with sum α_j = 1.
        Returns (const, gamma[5], alpha[k+1], R2, t_alpha[k+1], idx_k).
        """
        idx_k = _valid_index_for_k(k)
        yk = rx_bar.loc[idx_k].values
        Fj = [F_lags[j].loc[idx_k, reg_names].to_numpy() for j in range(k + 1)]
        T = len(yk)

        import numpy as np
        import statsmodels.api as sm

        # Initialize α (uniform) and γ from k=0 OLS
        alpha = np.ones(k + 1) / (k + 1)
        X0 = sm.add_constant(Fj[0])
        mdl0 = sm.OLS(yk, X0).fit()
        const = float(mdl0.params[0])
        gamma = mdl0.params[1:].copy()   # in reg_names order

        # Project to enforce sum(alpha) = 1 during updates
        def _project_alpha(a_vec):
            s = a_vec.sum()
            if s == 0:
                return np.ones_like(a_vec) / len(a_vec)
            return a_vec / s

        # Objective (for convergence check only)
        def _loss(c, g, a_vec):
            Z = sum((Fj[j] @ g) * a_vec[j] for j in range(k + 1))
            res = yk - (c + Z)
            return float(res @ res / T)

        last = np.inf
        it = 0
        while it < max_iter:
            it += 1

            # Update gamma holding alpha, const fixed
            Z_a = sum(alpha[j] * Fj[j] for j in range(k + 1))            # T x p
            mdl_g = sm.OLS(yk - const, Z_a).fit()
            g_new = mdl_g.params

            # Update alpha holding gamma, const fixed
            Z_g = np.column_stack([Fj[j] @ g_new for j in range(k + 1)])  # T x (k+1)
            mdl_a = sm.OLS(yk - const, Z_g).fit()
            a_new = _project_alpha(mdl_a.params)

            # Update constant given new (g, a)
            yhat_za = Z_g @ a_new
            const_new = float(np.mean(yk - yhat_za))

            val = _loss(const_new, g_new, a_new)
            if abs(last - val) < tol:
                const, gamma, alpha = const_new, g_new, a_new
                break
            const, gamma, alpha, last = const_new, g_new, a_new, val

        # Sign normalization: ensure middle (f3) weight non-negative if present
        if "f3" in reg_names:
            idx_f3 = reg_names.index("f3")
            if gamma[idx_f3] < 0:
                gamma = -gamma
                alpha = -alpha

        # Fit stats & α t-stats with HAC
        Z = np.column_stack([Fj[j] @ gamma for j in range(k + 1)])
        yhat = const + Z @ alpha
        resid = yk - yhat
        R2 = 1.0 - (resid @ resid) / ((yk - yk.mean()) @ (yk - yk.mean()))

        mdl_alpha = sm.OLS(yk, sm.add_constant(Z)).fit()
        robust_alpha = mdl_alpha.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        t_alpha = robust_alpha.tvalues[1:1 + k + 1]  # skip const

        return const, gamma, alpha, R2, t_alpha, idx_k

    # --- Build panels (unchanged layout) ---
    rows_A, rows_B, details = [], [], {}
    for k in range(0, max_k + 1):
        const, gamma, alpha, R2, t_alpha, idx_k = _fit_restricted_k(k)

        # Panel A row
        rowA = {"k": k, "const": const}
        for i, nm in enumerate(reg_names):
            rowA[nm] = gamma[i]
        rowA["R2"] = R2
        rows_A.append(rowA)

        # Panel B row (pad to 4 α's to match paper layout)
        a_row = {f"alpha{j}": (alpha[j] if j <= k else np.nan) for j in range(4)}
        t_row = {f"t(alpha{j})": (t_alpha[j] if j <= k else np.nan) for j in range(4)}
        rows_B.append({"k": k, **a_row, **t_row})

        details[k] = {
            "const": const,
            "gamma": gamma,
            "alpha": alpha,
            "R2": R2,
            "index": idx_k
        }

    panelA = pd.DataFrame(rows_A).set_index("k")[["const", "y1", "f2", "f3", "f4", "f5", "R2"]]
    panelB = pd.DataFrame(rows_B).set_index("k")[[
        "alpha0", "alpha1", "alpha2", "alpha3",
        "t(alpha0)", "t(alpha1)", "t(alpha2)", "t(alpha3)"
    ]]

    return {"panelA": panelA, "panelB": panelB, "details": details}

def show_cp_table5(tbl5, digits=2, title = None):
    A = tbl5["panelA"].copy().round(digits)
    B = tbl5["panelB"].copy().round(digits)
    display(Markdown("#### Additional Lags — Panel A: γ estimates"))
    display(A)
    display(Markdown("#### Additional Lags — Panel B: α estimates and t-statistics"))
    display(B)

def replicate_cp_table4(yields_df: pd.DataFrame,
                        horizon_months: int = 12,
                        nw_lags: int = 18):
    """
    Recreate CP (2005) Table 4 *core* columns:
      - R² from restricted regressions
      - NW,18 Wald χ² (and p-value) that omitted W-block is zero
      - 5% χ² critical value (df = dim(W))
    """

    # ---------- 0) Month-end index + sanity ----------
    y = yields_df.copy()
    if isinstance(y.index, pd.PeriodIndex):
        y.index = y.index.to_timestamp(how="end")
    else:
        y.index = pd.to_datetime(y.index) + pd.offsets.MonthEnd(0)
    y = y.sort_index()

    req = ["y1", "y2", "y3", "y4", "y5"]
    if not all(c in y.columns for c in req):
        raise ValueError(f"Need columns {req}. Got {list(y.columns)}")

    # ---------- 1) Target: rx̄ (avg of xr2..xr5) via shared helper ----------
    mats = [1, 2, 3, 4, 5]  # include 1 so n-1 exists for n=2
    rx_all = calculate_excess_returns(y, maturities=mats, horizon=horizon_months, risk_free_col="y1")
    rx_bar = rx_all[[f"xr{n}" for n in [2, 3, 4, 5]]].mean(axis=1)

    # align to target
    idx = rx_bar.dropna().index
    X5 = y.loc[idx, req].copy()
    ybar = rx_bar.loc[idx].values
    X_full = X5.values
    T = len(ybar)

    # ---------- 2) PCA (covariance PCA → standardize=False) ----------
    # silence any prints from helper
    with redirect_stdout(io.StringIO()):
        factors_df, loadings, _ = extract_factors_pca(X5, n_factors=3, standardize=False)
    level     = factors_df.iloc[:, 0].to_numpy()
    slope     = factors_df.iloc[:, 1].to_numpy()
    curvature = factors_df.iloc[:, 2].to_numpy()

    # Sign conventions (readability only)
    V = loadings.copy()
    if V[:, 0].sum() < 0:
        V[:, 0] *= -1; level *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1; slope *= -1
    if V[2, 2] < 0:
        V[:, 2] *= -1; curvature *= -1
    pc_loadings = pd.DataFrame(V[:, :3], index=req,
                               columns=["PC1_level", "PC2_slope", "PC3_curvature"])

    # ---------- 3) Restricted specs ----------
    Z_specs = [
        ("Slope",                ["PC2_slope"],                               slope.reshape(-1, 1)),
        ("Level, slope",         ["PC1_level", "PC2_slope"],                  np.column_stack([level, slope])),
        ("Level, slope, curve",  ["PC1_level", "PC2_slope", "PC3_curvature"], np.column_stack([level, slope, curvature])),
        ("y(5) − y(1)",          ["y5 - y1"],                                 (X5["y5"] - X5["y1"]).to_numpy().reshape(-1, 1)),
        ("y(1), y(5)",           ["y1", "y5"],                                X5[["y1", "y5"]].to_numpy()),
        ("y(1), y(4), y(5)",     ["y1", "y4", "y5"],                          X5[["y1", "y4", "y5"]].to_numpy()),
    ]

    # ---------- 4) Run specs ----------
    summary_rows, coef_tables = [], {}
    for name, cols, Z in Z_specs:
        # Restricted regression (for R² and reporting)
        XZ = sm.add_constant(Z)
        mdl_Z = sm.OLS(ybar, XZ).fit()
        R2 = mdl_Z.rsquared

        # Orthogonal complement (df = #W columns)
        W = orthogonal_complement(X_full, Z)
        df_test = W.shape[1]
        crit_5pct = float(chi2.ppf(0.95, df=df_test)) if df_test > 0 else np.nan

        # NW(18) Wald test
        chi2_nw, df_nw, p_nw = wald_block_NW(ybar, Z, W, lags=nw_lags)

        summary_rows.append({
            "Right-hand variables": name,
            "R2": R2,
            "NW, 18 χ2": chi2_nw,
            "NW, 18 p-value": p_nw,
            "5 percent crit. value": crit_5pct,
        })

        # Coefficients with NW(18) SEs for the restricted regression
        robZ = mdl_Z.get_robustcov_results(cov_type="HAC", maxlags=nw_lags)
        coef_tbl = pd.DataFrame({"coef": robZ.params, "SE_NW18": robZ.bse},
                                index=["const"] + cols)
        coef_tables[name] = coef_tbl

    summary = pd.DataFrame(summary_rows).set_index("Right-hand variables")
    return {"summary": summary, "coefficients": coef_tables, "pc_loadings": pc_loadings}



def show_cp_table4(res, title=None):
    """
    Pretty printer for the dict returned by `replicate_cp_table4` (NW-only version).
    Shows: R², NW,18 (χ² and p-value), and 5% critical value.
    Displays once and returns None (prevents duplicate output in notebooks).
    """

    df = res["summary"].copy()
    crit_candidates = ["5 percent crit. value", "5% crit. value"]
    crit_col = next((c for c in crit_candidates if c in df.columns), None)

    cols = ["R2", "NW, 18 χ2", "NW, 18 p-value"]
    if crit_col:
        cols.append(crit_col)
    out = df[cols].copy()

    # formatters
    def _fmt_r2(x):  return "" if pd.isna(x) else f"{x:.2f}"
    def _fmt_chi2(x):return "" if pd.isna(x) else f"{x:.1f}"
    def _fmt_p(x):
        if pd.isna(x): return ""
        return "⟨0.00⟩" if x < 0.005 else f"{x:.2f}"
    def _fmt_cv(x): return "" if pd.isna(x) else f"{x:.1f}"

    fmt = {}
    for c in out.columns:
        if c == "R2":
            fmt[c] = _fmt_r2
        elif "p-value" in c:
            fmt[c] = _fmt_p
        elif "crit" in c:
            fmt[c] = _fmt_cv
        else:
            fmt[c] = _fmt_chi2

    # Build a simple two-level header so the first column shows 'R²'
    leaf_labels = ["", "χ²", "p-value"] + (["5 percent crit. value"] if crit_col else [])
    tuples = [("R²", "")]
    tuples += [("NW, 18", "χ²"), ("NW, 18", "p-value")]
    if crit_col:
        tuples += [(" ", "5 percent crit. value")]
    mcols = pd.MultiIndex.from_tuples(tuples)

    out2 = out.copy()
    out2.columns = leaf_labels

    if title:
        display(Markdown(f"### {title}"))
    try:
        sty = (out2.style
               .format(fmt, na_rep="")
               .set_properties(**{"text-align": "center"})
               .set_table_styles([
                   {"selector": "th.col_heading.level0", "props": "text-align:center; padding-bottom:4px;"},
                   {"selector": "th.col_heading.level1", "props": "text-align:center;"},
                   {"selector": "th.row_heading", "props": "text-align:left;"},
               ]))
        sty.data.columns = mcols
        display(sty)
    except Exception:
        display(out)
    return None
