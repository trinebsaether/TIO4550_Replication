"""
Ludvigson & Ng (2009) - Macro Factors in Bond Risk Premia

Key Methodology:
1. Factor Extraction: Extract 8 factors from 132 macro series using PCA
2. Model Selection: Use BIC to select optimal factor specifications
3. Predictive Regressions: Test macro factors' predictive power for bond returns
4. Validation: Compare results with paper's findings
"""
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("."))  # ensure project root on path
# Import our utility functions
from ..data_processing import (
    load_fred_md_data, apply_fred_transformations,
    standardize_data, prepare_balanced_panel, load_wrds_famabliss,
    load_updated_fhat, load_fed_gsw_daily_yields,
    get_ln_grouping,
)
from ..factor_models import (
    extract_factors_pca,
)
from ..yield_processing import (
    calculate_excess_returns, create_cochrane_piazzesi_factor,
)
from ..regression import (
    predictive_regression,
    create_summary_factor,
    model_selection_bic,
)
from itertools import combinations

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# Master switches
USE_RFS2009_FACTORS_AND_CP = True  # baseline in-sample benchmark (1964–2003)
START_DATE = '1964-01'
END_DATE = '2003-12'
OOS_START = '1964-01'
OOS_END = '2024-12'

def load_and_prepare_data():
    """Load and prepare all required data (FRED-MD, control, and bond yields)"""
    START_DATE = '1964-01'
    END_DATE = '2003-12'
    
    fred_data_path = "data/LN_data/LN_handbook_data.csv"
    control_data_path = "data/LN_data/RFS2009.CSV"
    wrds_path = "data/wrds_famabliss.csv"

    # Load FRED-MD macro data
    try:
        fred_data, transform_codes = load_fred_md_data(fred_data_path)
        
    except FileNotFoundError as e:
        print(f"error loading fmd {e}")
        fred_data, transform_codes = None, None

    # Load control data (RFS2009.CSV)
    try:
        control_data = pd.read_csv(control_data_path, index_col=0)
        control_data.index = pd.to_datetime(control_data.index, format='%m/%d/%Y')
        control_cp_factor = control_data['CP'].copy()
        
    except FileNotFoundError as e:
        print(f"error loading RFS2009 {e}")
        control_data, control_cp_factor = None, None

    # Load WRDS Fama-Bliss yields using utility
    bond_yields = None
    if control_data is not None:
        try:
            end_dt_ext = (pd.to_datetime(END_DATE + '-01') + pd.DateOffset(months=12)).strftime('%Y-%m')
            bond_yields = load_wrds_famabliss(wrds_path, START_DATE, end_dt_ext)
            for c in ['y1', 'y2', 'y3', 'y4', 'y5']:
                if c in bond_yields.columns:
                    bond_yields[c] = bond_yields[c] * 100.0
            
        except Exception as e:
            print("Error loading WRDS Fama-Bliss yields:", e)
            bond_yields = None

    return fred_data, transform_codes, bond_yields, control_cp_factor, control_data, START_DATE, END_DATE

def process_data_and_extract_factors(fred_data, transform_codes, bond_yields, control_cp_factor, control_data, START_DATE, END_DATE):
    """Process data; return both estimated (PCA) and control (RFS) factors and CP."""
    
    # Transform FRED-MD data (sample window) → PCA factors (estimated)
    sample_start_dt = pd.to_datetime(START_DATE + '-01')
    sample_end_dt = pd.to_datetime(END_DATE + '-01')
    fred_sample = fred_data[(fred_data.index >= sample_start_dt) & (fred_data.index <= sample_end_dt)].copy()
    
    fred_transformed = apply_fred_transformations(fred_sample, transform_codes)
    fred_balanced = prepare_balanced_panel(fred_transformed, min_obs_ratio=0.95)
    fred_standardized, _ = standardize_data(fred_balanced, method='zscore')
    fred_final = fred_standardized.dropna()
    

    N_FACTORS = 8
    est_factors, factor_loadings, factor_results = extract_factors_pca(
        fred_final, n_factors=N_FACTORS, standardize=False
    )
    est_factors.columns = [f"F{i}" for i in range(1, N_FACTORS + 1)]
    

    # Load control factors and CP from RFS (control)
    rfs = pd.read_csv("data/LN_data/RFS2009.CSV")
    date_col = rfs.columns[0]
    rfs[date_col] = pd.to_datetime(rfs[date_col], infer_datetime_format=True)
    rfs = rfs.set_index(date_col)
    rfs.index = rfs.index.to_period('M')
    s_period = pd.to_datetime(START_DATE + '-01').to_period('M')
    e_period = pd.to_datetime(END_DATE + '-01').to_period('M')
    rfs = rfs.loc[(rfs.index >= s_period) & (rfs.index <= e_period)]
    factor_cols = [f"f{i}" for i in range(1, N_FACTORS + 1)]
    assert all(c in rfs.columns for c in factor_cols), "RFS2009.CSV missing f1..f8"
    ctrl_factors = rfs[factor_cols].copy()
    ctrl_factors.columns = [f"F{i}" for i in range(1, N_FACTORS + 1)]
    cp_control = rfs['CP'].copy()
    cp_control.index = ctrl_factors.index        

    # AR(1) coefficients for estimated factors (used in summary)
    ar1_coeffs = [est_factors[f'F{i}'].autocorr(lag=1) for i in range(1, N_FACTORS + 1)]

    # Create CP once from yields (created) and obtain avg excess returns
    cp_results = create_cochrane_piazzesi_factor(
        bond_yields, method='bauer_hamilton', verbose=False, risk_free_col='y1'
    )
    cp_created = cp_results['cp_factor']
    avg_excess_returns = cp_results['avg_excess_return']
    if not isinstance(avg_excess_returns.index, pd.PeriodIndex):
        avg_excess_returns.index = avg_excess_returns.index.to_period('M')

    

    # Return both estimated and control sets, plus both CP variants
    return est_factors, ctrl_factors, factor_results, avg_excess_returns, ar1_coeffs, fred_final, cp_created, cp_control

def run_bic_model_selection(macro_factors, avg_excess_returns, cp_factor):
    """
    Run BIC model selection from scratch following LN methodology
    - Exhaustive combinations of exactly 6 factors (excluding CP)
    - Exhaustive combinations of exactly 5 factors (excluding CP), with CP added as regressor
    """
    # Align indices to monthly Periods and get common index
    mf = macro_factors.copy()
    ar = avg_excess_returns.copy()
    cp = cp_factor.copy()

    if not isinstance(mf.index, pd.PeriodIndex):
        mf.index = mf.index.to_period('M')
    if not isinstance(ar.index, pd.PeriodIndex):
        ar.index = ar.index.to_period('M')
    if not isinstance(cp.index, pd.PeriodIndex):
        cp.index = cp.index.to_period('M')

    common_index = mf.index.intersection(ar.index).intersection(cp.index)
    if len(common_index) == 0:
        raise ValueError("No common monthly periods between factors, excess returns, and CP factor after alignment.")

    # Use reindex to avoid KeyErrors
    factors_aligned = mf.reindex(common_index)
    returns_aligned = ar.reindex(common_index)
    cp_aligned = cp.reindex(common_index)

    factors_ext = factors_aligned.copy()
    # LN-mimic mode: allow F1^3 as an additional candidate term
    factors_ext_ln = factors_ext.copy()
    if 'F1' in factors_ext_ln.columns:
        factors_ext_ln['F1_cubed'] = factors_ext_ln['F1'] ** 3

    # Exact-size combinations: 6 without CP, 5 with CP (LN-style with optional F1^3)
    base_cols = [c for c in factors_ext.columns if c.startswith('F')]
    base_cols_ln = [c for c in factors_ext_ln.columns if (c.startswith('F') or c == 'F1_cubed')]
    if len(base_cols) < 5:
        raise ValueError("Insufficient candidate factors for BIC search (need at least 5).")

    combos_no_cp_ln = [factors_ext_ln[list(c)] for c in combinations(base_cols_ln, min(6, len(base_cols_ln)))]
    combos_with_cp_ln = [factors_ext_ln[list(c)] for c in combinations(base_cols_ln, min(5, len(base_cols_ln)))]

    # Run BIC selection using utilities (LN-style only)
    bic_results_without_cp_ln = model_selection_bic(returns_aligned, combos_no_cp_ln)
    bic_results_with_cp_ln = model_selection_bic(returns_aligned, combos_with_cp_ln, base_regressors=cp_aligned.rename('CP').to_frame())

    # Package overall results
    model_selection_results = {
        'without_cp': bic_results_without_cp_ln,
        'with_cp': bic_results_with_cp_ln
    }

    return model_selection_results, factors_aligned, returns_aligned, cp_aligned

def run_predictive_regressions(model_selection_results, macro_factors, avg_excess_returns, cp_factor):
    """Use the BIC-selected factor specifications for predictive regressions"""

    # Get BIC-selected specifications (guard against None)
    best_without_cp = None
    best_with_cp = None
    if model_selection_results and 'without_cp' in model_selection_results and model_selection_results['without_cp']:
        best_without_cp = model_selection_results['without_cp'].get('best_specification')
    if model_selection_results and 'with_cp' in model_selection_results and model_selection_results['with_cp']:
        best_with_cp = model_selection_results['with_cp'].get('best_specification')
    
    # Align inputs
    mf = macro_factors.copy()
    ar = avg_excess_returns.copy()
    cp = cp_factor.copy()

    if not isinstance(mf.index, pd.PeriodIndex):
        mf.index = mf.index.to_period('M')
    if not isinstance(ar.index, pd.PeriodIndex):
        ar.index = ar.index.to_period('M')
    if not isinstance(cp.index, pd.PeriodIndex):
        cp.index = cp.index.to_period('M')

    common_index = mf.index.intersection(ar.index).intersection(cp.index)
    if len(common_index) == 0:
        raise ValueError("No common monthly periods between factors, excess returns, and CP factor after alignment.")

    factors_aligned = mf.reindex(common_index)
    returns_aligned = ar.reindex(common_index)
    cp_aligned = cp.reindex(common_index)
    
    print(f"Regression sample: {len(factors_aligned)} observations")
    if len(factors_aligned) > 0:
        print(f"From {factors_aligned.index.min()} to {factors_aligned.index.max()}")
    
    # Extended factors (no manual nonlinear terms; powers handled post-selection)
    extended_factors = factors_aligned.copy()
    
    regression_results = {}
    
    # 1. CP factor only (benchmark)
    
    cp_data = pd.DataFrame({'CP': cp_aligned})
    reg_cp = predictive_regression(returns_aligned, cp_data, hac_lags=18)
    regression_results['cp_only'] = reg_cp
    
    # 2. BIC-selected specification without CP (F6)
    if best_without_cp:
        f6_factors = best_without_cp['factors']
        
        
        available_f6 = [f for f in f6_factors if f in extended_factors.columns]
        if available_f6:
            X_f6 = extended_factors[available_f6]
            reg_f6 = predictive_regression(returns_aligned, X_f6, hac_lags=18)
            regression_results['f6_bic'] = reg_f6    

    else:
        pass
    
    # 3. F6 SUMMARY FACTOR (if F6 was available)
    if best_without_cp:
        f6_factors = best_without_cp['factors']
        available_f6 = [f for f in f6_factors if f in extended_factors.columns]
        if available_f6:
            F6_summary = create_summary_factor(returns_aligned, extended_factors, available_f6)
            F6_summary.name = 'F6'
            
            reg_f6_summary = predictive_regression(returns_aligned, 
                                                  pd.DataFrame({'F6': F6_summary}), 
                                                  hac_lags=18)
            regression_results['f6_summary'] = reg_f6_summary
            
    # 4. BIC-selected specification with CP (F5 + CP)
    if best_with_cp:
        f5_factors = best_with_cp['factors']
        
        # Store the BIC model directly, as it's already fitted
        regression_results['f5_cp_bic'] = best_with_cp.get('model')

        available_f5 = [f for f in f5_factors if f in extended_factors.columns]
        if available_f5:
            F5_summary = create_summary_factor(returns_aligned, extended_factors, available_f5)
            F5_summary.name = 'F5'
            
            f5_cp_summary = pd.DataFrame({
                'F5': F5_summary,
                'CP': cp_aligned
            })
            reg_f5_cp_summary = predictive_regression(returns_aligned, f5_cp_summary, hac_lags=18)
            regression_results['f5_cp_summary'] = reg_f5_cp_summary
    else:
        pass
    
    return regression_results

def summarize_results(regression_results, model_selection_results, factor_results, ar1_coeffs, fred_final, START_DATE, END_DATE):
    """Comprehensive results summary and validation"""
    print("LUDVIGSON & NG (2009) REPLICATION SUMMARY")
    print("=" * 80)

    print("\n2. DATA AND FACTOR EXTRACTION:")
    print(f"   • Macro series: {fred_final.shape[1]} (target: ~132)")
    print(f"   • Sample: {START_DATE} to {END_DATE} ({len(fred_final)} obs)")
    print(f"   • Total variance explained: {factor_results['total_variance_explained']:.1%}")
    print(f"   • Factor persistence: {min(ar1_coeffs):.3f} to {max(ar1_coeffs):.3f}")
    
    print("\n3. BIC MODEL SELECTION RESULTS:")
    if model_selection_results:
        # Show significant factors
        if 'significant_factors' in model_selection_results:
            sig_factors = model_selection_results['significant_factors']
            print(f"   • Significant factors: {sig_factors}")
        
        # Show selected specifications (guard against None)
        without = model_selection_results.get('without_cp') or {}
        withcp = model_selection_results.get('with_cp') or {}

        if isinstance(without, dict) and without.get('best_specification'):
            f6_spec = without['best_specification']['factors']
            f6_bic = without['best_specification']['bic']
            print(f"   • F6 specification: {f6_spec}")
            print(f"   • F6 BIC: {f6_bic:.2f}")
        else:
            print("   • F6 specification: not available")
        
        if isinstance(withcp, dict) and withcp.get('best_specification'):
            f5_spec = withcp['best_specification']['factors']
            f5_bic = withcp['best_specification']['bic']
            print(f"   • F5 specification: {f5_spec} + CP")
            print(f"   • F5+CP BIC: {f5_bic:.2f}")
        else:
            print("   • F5+CP specification: not available")
    
    print("\n4. PREDICTIVE REGRESSION RESULTS:")
    print("   " + "-"*50)
    for name, result in regression_results.items():
        if result is None: continue
        spec_name = name.replace('_', ' ').title()
        r2_pct = result.rsquared * 100
        print(f"   • {spec_name:<25}: R² = {result.rsquared:.3f} ({r2_pct:.1f}%)")
    
    print("\n5. COMPARISON WITH LUDVIGSON & NG (2009):")
    print("   • LN Paper Results:")
    print("     - CP factor alone: R² ≈ 31%")
    print("     - F6 factors alone: R² ≈ 26%")
    print("     - Combined (F5+CP): R² ≈ 44-45%")
    print("     - F1 highly significant (t-stat > 5)")
    
    print("   • Our Replication:")
    if 'cp_only' in regression_results and regression_results['cp_only'] is not None:
        cp_r2 = regression_results['cp_only'].rsquared * 100
        print(f"     - CP factor alone: R² = {cp_r2:.1f}%")
    
    if 'f6_summary' in regression_results and regression_results['f6_summary'] is not None:
        f6_r2 = regression_results['f6_summary'].rsquared * 100
        print(f"     - F6 summary: R² = {f6_r2:.1f}%")
    
    combined_key = None
    for key in ['f5_cp_summary', 'f5_cp_bic']:
        if key in regression_results and regression_results[key] is not None:
            combined_key = key
            break
    
    if combined_key:
        combined_r2 = regression_results[combined_key].rsquared * 100
        print(f"     - Combined (F5+CP): R² = {combined_r2:.1f}%")
        
        # Assessment
        if combined_r2 >= 35:
            print("   ✓ REPLICATION SUCCESSFUL: Results consistent with paper")
        elif combined_r2 >= 25:
            print("   ⚠ PARTIAL SUCCESS: Results in reasonable range")
        else:
            print("   ✗ REPLICATION ISSUES: Results differ significantly")
    
    print("\n" + "=" * 80)
    print("REPLICATION COMPLETED WITH ENDOGENOUS MODEL SELECTION")
    print("=" * 80)

def plot_factor_r2_grid(fred_panel: pd.DataFrame, factors: pd.DataFrame,
                        start: str = '1964-01', end: str = '2003-12',
                        factor_cols: list = None, max_bars: int = None,
                        title_prefix: str = 'R² of factor vs FRED-MD series',
                        ordered_series: list = None,
                        series_to_group: dict = None):
    """Render all requested factor barplots in a single window (grid of subplots)."""

    fred = fred_panel.copy()
    fac = factors.copy()
    if not isinstance(fred.index, pd.PeriodIndex):
        fred.index = fred.index.to_period('M')
    if not isinstance(fac.index, pd.PeriodIndex):
        fac.index = fac.index.to_period('M')

    s = pd.to_datetime(start + '-01').to_period('M')
    e = pd.to_datetime(end + '-01').to_period('M')
    fred = fred.loc[(fred.index >= s) & (fred.index <= e)]
    fac = fac.loc[(fac.index >= s) & (fac.index <= e)]

    series_names = fred.columns.tolist()
    if factor_cols is None:
        factor_cols = [c for c in fac.columns if c.startswith('F')]

    computation_order = ordered_series if ordered_series is not None else series_names

    n = len(factor_cols)
    if n >= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 1, n
    fig, axes = plt.subplots(rows, cols, figsize=(22, 10), dpi=180)
    if hasattr(axes, 'flatten'):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, fcol in enumerate(factor_cols[:rows*cols]):
        ax = axes[idx]
        if fcol not in fac.columns:
            ax.axis('off')
            continue
        results = []
        for scol in computation_order:
            if scol not in fred.columns:
                continue
            try:
                res = predictive_regression(fac[fcol], fred[[scol]], add_constant=True, hac_lags=18)
                results.append((scol, float(res.rsquared)))
            except Exception:
                print("Error computing R² for", fcol, "vs", scol)
                results.append((scol, np.nan))

        r2_df = pd.DataFrame(results, columns=['series', 'r2']).dropna()
        if ordered_series is not None:
            r2_df['order_idx'] = r2_df['series'].apply(lambda x: ordered_series.index(x) if x in ordered_series else 10**9)
            r2_df = r2_df.sort_values('order_idx')
        else:
            r2_df = r2_df.sort_values('r2', ascending=False)

        plot_df = r2_df.copy() if (max_bars is None or max_bars <= 0) else r2_df.head(max_bars).copy()

        if series_to_group is not None:
            plot_df['group'] = plot_df['series'].map(series_to_group).fillna('Other')
            palette = sns.color_palette('tab20', n_colors=20)
            unique_groups = plot_df['group'].unique().tolist()
            group_to_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
            sns.barplot(data=plot_df, x='series', y='r2', hue='group', palette=group_to_color, dodge=False, edgecolor='black', alpha=0.9, ax=ax)
            ax.legend(title='Group', fontsize=14, loc='upper right')
        else:
            sns.barplot(data=plot_df, x='series', y='r2', color='C0', edgecolor='black', alpha=0.9, ax=ax)

        ax.set_title(f"{fcol} ({start}–{end})", fontsize=14)
        ax.set_ylabel('R²')
        ax.set_xlabel('')
        # Hide x-axis tick labels for cleaner single-factor view
        ax.set_xticklabels([])
    plt.suptitle(title_prefix, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    """Main execution function with only final summary printed; plots kept."""
    fred_data, transform_codes, bond_yields, control_cp_factor, control_data, START_DATE, END_DATE = load_and_prepare_data()
    if fred_data is None or bond_yields is None:
        return

    results = process_data_and_extract_factors(fred_data, transform_codes, bond_yields, control_cp_factor, control_data, START_DATE, END_DATE)
    if results[0] is None:
        return

    est_factors, ctrl_factors, factor_results, avg_excess_returns, ar1_coeffs, fred_final, cp_created, cp_control = results

    # Let user choose inputs for in-sample modeling
    print("\nChoose inputs for in-sample modeling:")
    print("  1) Estimated PCA factors + Created CP (from yields)")
    print("  2) Control RFS factors + Control CP (from RFS2009)")
    choice = input("Select [1/2] (default=2): ").strip()
    use_estimated = (choice == '1')
    macro_factors = est_factors if use_estimated else ctrl_factors
    cp_factor = cp_created if use_estimated else cp_control

    model_selection_results, factors_aligned, returns_aligned, cp_aligned = run_bic_model_selection(
        macro_factors, avg_excess_returns, cp_factor
    )
    regression_results = run_predictive_regressions(
        model_selection_results, macro_factors, avg_excess_returns, cp_factor
    )

    summarize_results(regression_results, model_selection_results, factor_results, 
                     ar1_coeffs, fred_final, START_DATE, END_DATE)

    # Visualizations: ask user
    plot_choice = input("Do you want to plot factor-vs-series R²? (y/n, default=n): ").lower().strip()
    if plot_choice in ['y', 'yes']:
        ordered_series, series_to_group = get_ln_grouping()
        # Single window with all eight factors
        plot_factor_r2_grid(
            fred_panel=fred_final,
            factors=macro_factors,
            start=START_DATE,
            end=END_DATE,
            factor_cols=[f'F{i}' for i in range(1, 9)],
            max_bars=None,
            title_prefix='R² of factor vs FRED-MD series',
            ordered_series=ordered_series,
            series_to_group=series_to_group
        )

    # Optional Out-of-sample run
    # try:
    gsw_daily_path = "data/FED_GSW_daily.csv"
    fhat_path = "data/LN_data/updated_fhat.xlsx"
    yields_oos = load_fed_gsw_daily_yields(gsw_daily_path, start=OOS_START, end=OOS_END)
    needed_y = [c for c in [f'y{i}' for i in [1,2,3,4,5]] if c in yields_oos.columns]
    yields_oos = yields_oos[needed_y]
    macro_oos = load_updated_fhat(fhat_path, start=OOS_START, end=OOS_END)
    macro_oos_cols = [c for c in macro_oos.columns if c.startswith('F')]
    macro_oos = macro_oos[macro_oos_cols]

    cp_oos_results = create_cochrane_piazzesi_factor(
        yields_oos, method='bauer_hamilton', verbose=False, risk_free_col='y1'
    )
    cp_oos = cp_oos_results['cp_factor']
    xr_oos = cp_oos_results['avg_excess_return']
    if not isinstance(cp_oos.index, pd.PeriodIndex):
        cp_oos.index = cp_oos.index.to_period('M')
    if not isinstance(xr_oos.index, pd.PeriodIndex):
        xr_oos.index = xr_oos.index.to_period('M')
    if not isinstance(macro_oos.index, pd.PeriodIndex):
        macro_oos.index = macro_oos.index.to_period('M')
    common = macro_oos.index.intersection(xr_oos.index).intersection(cp_oos.index)
    macro_oos = macro_oos.reindex(common)
    xr_oos = xr_oos.reindex(common)
    cp_oos = cp_oos.reindex(common)
    model_sel_oos, mf_aligned_oos, xr_aligned_oos, cp_aligned_oos = run_bic_model_selection(macro_oos, xr_oos, cp_oos)
    reg_results_oos = run_predictive_regressions(model_sel_oos, macro_oos, xr_oos, cp_oos)
    # Print concise OOS results
    print("\nOOS Predictive regressions (key R²):")
    for k in ['cp_only', 'f6_summary', 'f5_cp_summary', 'f5_cp_bic']:
        if k in reg_results_oos and reg_results_oos[k] is not None:
            print(f"  {k}: R² = {reg_results_oos[k].rsquared:.3f}")
    # except Exception:
    #     pass

if __name__ == "__main__":
    main()
