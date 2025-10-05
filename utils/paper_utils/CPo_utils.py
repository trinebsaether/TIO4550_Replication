import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

def build_annual_horizon_excess_returns(yields_df: pd.DataFrame, n_min=2, n_max=30) -> pd.DataFrame:
    """
    Overlapping 1-year (12-month) holding-period excess returns:
    r_{t+1}^{(n)} = -(n-1) y_{t+12}^{(n-1)} + n y_t^{(n)} - y_t^{(1)}
    (paper's Eq. (2) with t+1 meaning +1 year; monthly data so shift -12).
    Yields in percent; output in decimal. Last 12 months become NaN.
    """
    y = yields_df.copy()
    out = {}
    for n in range(max(2, n_min), min(30, n_max)+1):
        y_n_t = y[f"y{n}"]
        y_n1_tplus12 = y[f"y{n-1}"].shift(-12)
        y_1_t = y["y1"]
        out[f"rx1y({n})"] = ( - (n-1)*y_n1_tplus12 + n*y_n_t - y_1_t )
    return pd.DataFrame(out, index=y.index)

def tau_dma(infl_series: pd.Series, nu: int, window: int) -> pd.Series:
    """
    Tau per Eq. (19): tau_t = (1-nu) * sum_{j=0}^{window-1} nu^j * pi_{t-j}
    Truncated at window. Returns NaN until enough history.
    """
    pi = infl_series.copy()
    w = (1 - nu) * (nu ** np.arange(window))  # length window
    vals = pi.values
    out = np.full(len(pi), np.nan)
    for t in range(window-1, len(pi)):
        window_vals = vals[t-window+1:t+1][::-1]  # current pi_t at weight (1-nu)
        out[t] = np.dot(w, window_vals)

    return pd.Series(out, index=pi.index, name="Tau")

def duration_weighted_average(df: pd.DataFrame, type_name="excess_return", start_mat=2, end_mat=15) -> pd.Series:
    """Calculate duration-weighted average across different types of data (returns, yields, cycles)"""
    
    # Configure column pattern and output name based on type
    patterns = {
        "excess_return": ("rx1y({})", "rx_bar"),
        "yield": ("y{}", "y_bar"),
        "cycle": ("cycl{}", "c_bar")
    }
    
    col_pattern, output_name = patterns.get(type_name, (None, None))
    if col_pattern is None:
        raise ValueError(f"Unknown type_name: {type_name}. Choose from {list(patterns.keys())}")
    
    # Calculate duration-weighted average using available columns
    if type_name == "excess_return":
        parts = [df[col_pattern.format(k)] / k 
                 for k in range(start_mat, end_mat+1) 
                 if col_pattern.format(k) in df.columns]
    else: # For yields and cycles, it's a simple average
        parts = [df[col_pattern.format(k)] 
                 for k in range(start_mat, end_mat+1) 
                 if col_pattern.format(k) in df.columns]
    
    if not parts:
        return pd.Series(index=df.index, dtype=float, name=output_name)

    avg = pd.concat(parts, axis=1).mean(axis=1)
    avg.name = output_name
    return avg    

def run_cpo_regressions_set(xret_df: pd.DataFrame,
                            yields_df: pd.DataFrame,
                            Tau_df: pd.Series,
                            cycles_df: pd.DataFrame,
                            mats_yields=(1,2,5,7,10,15)):
    """
    Build and run regressions (22)-(25):

    (22) rx_bar_{t+1} ~ yields (selected maturities)
    (23) rx_bar_{t+1} ~ yields + Tau
    (24) rx_bar_{t+1} ~ y_bar_t + y_t^{(1)} + Tau_t
    (25) rx_bar_{t+1} ~ c_bar_t + c_t^{(1)}    (cycles already orthogonal to Tau)

    Returns dict with models and aligned dataframes.
    """
    # import statsmodels.formula.api as smf

    # Ensure alignment
    rx_bar = xret_df[["rx_bar"]].copy()

    # Add yields needed
    Y_cols = [f"y{m}" for m in mats_yields if f"y{m}" in yields_df.columns]
    df_y = yields_df[Y_cols]

    # Yield averages
    y_bar = duration_weighted_average(yields_df, type_name="yield")

    # Cycle pieces
    c_bar = duration_weighted_average(cycles_df, type_name="cycle")
    c1 = cycles_df["cycl1"] if "cycl1" in cycles_df.columns else pd.Series(index=cycles_df.index, dtype=float, name="cycl1")

    # DataFrames for each equation
    # Eq (22)
    df22 = pd.concat([rx_bar, df_y], axis=1).dropna()

    # Eq (23)
    df23 = pd.concat([rx_bar, df_y, Tau_df], axis=1).dropna()

    # Eq (24)
    df24 = pd.concat([rx_bar,
                      y_bar.rename("y_bar"),
                      yields_df["y1"].rename("y1"),
                      Tau_df],
                     axis=1).dropna()

    # Eq (25)
    df25 = pd.concat([rx_bar,
                      c_bar.rename("c_bar"),
                      c1.rename("cycl1")],
                     axis=1).dropna()

    models = {} 

    # Fit with HAC (12-month overlap => maxlags=11)
    models["22"] = smf.ols("rx_bar ~ " + " + ".join(Y_cols), data=df22).fit(cov_type="HAC", cov_kwds={"maxlags":11})
    models["23"] = smf.ols(f"rx_bar ~ {' + '.join(Y_cols)} + Tau", data=df23).fit(cov_type="HAC", cov_kwds={"maxlags":11})
    models["24"] = smf.ols("rx_bar ~ y_bar + y1 + Tau", data=df24).fit(cov_type="HAC", cov_kwds={"maxlags":11})
    models["25"] = smf.ols("rx_bar ~ c_bar + cycl1", data=df25).fit(cov_type="HAC", cov_kwds={"maxlags":11})

    out = {
        "models": models,
        "data": {"22": df22, "23": df23, "24": df24, "25": df25}
    }
    return out


def get_test_df(model_key: str, utils_data: list, maturities: list) -> pd.DataFrame:
    """
    Prepare the test DataFrame for a given model key using provided utility data.
    """
    # unpack utils_data:
    xret_test, yields_test, Tau_test, y_bar_test, c_bar_test, c1_test = utils_data
    
    if model_key == "22":
        Y_cols = [f"y{m}" for m in maturities if f"y{m}" in yields_test.columns]
        return pd.concat([xret_test["rx_bar"], yields_test[Y_cols]], axis=1).dropna()
    elif model_key == "23":
        Y_cols = [f"y{m}" for m in maturities if f"y{m}" in yields_test.columns]
        return pd.concat([xret_test["rx_bar"], yields_test[Y_cols], Tau_test], axis=1).dropna()
    elif model_key == "24":
        return pd.concat([xret_test["rx_bar"], y_bar_test.rename("y_bar"), yields_test["y1"].rename("y1"), Tau_test], axis=1).dropna()
    elif model_key == "25":
        return pd.concat([xret_test["rx_bar"], c_bar_test.rename("c_bar"), c1_test.rename("cycl1")], axis=1).dropna()
    else:
        raise ValueError("Unknown model key")
    

def plot_model_predictions(results_set, figsize=(16, 12)):
    """
    Plot predictions from models (22)-(25) against actual excess returns.
    Left: Time series comparison
    Right: Predicted vs Actual scatter plot with regression line
    
    Parameters:
    results_set: Dictionary with 'models' and 'data' from run_cpo_regressions_set()
    figsize: Tuple of figure dimensions
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import YearLocator, DateFormatter
    import numpy as np
    
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    
    # For each model (22, 23, 24, 25)
    for i, eq_num in enumerate(['22', '23', '24', '25']):
        model = results_set['models'][eq_num]
        data = results_set['data'][eq_num]
        
        # Get actual values
        actual = data['rx_bar']
        # Generate predictions
        predicted = model.predict()
        
        # Time series plot (left)
        ax1 = axs[i, 0]
        ax1.plot(data.index, actual, 'b-', alpha=0.7, label='Actual')
        ax1.plot(data.index, predicted, 'r-', alpha=0.7, label='Predicted')
        ax1.set_title(f'Eq. ({eq_num}) - Time Series (R² = {model.rsquared:.3f})')
        ax1.set_ylabel('Excess Return')
        
        # Format x-axis for dates
        ax1.xaxis.set_major_locator(YearLocator(5))
        ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Add recession shading
        # Add vertical lines at business cycle turning points
        for recession_start, label in [(pd.Timestamp('1973-11-01'), '73'),
                                      (pd.Timestamp('1980-01-01'), '80'),
                                      (pd.Timestamp('1990-07-01'), '90'),
                                      (pd.Timestamp('2001-03-01'), '01'),
                                      (pd.Timestamp('2007-12-01'), '08')]:
            if recession_start in data.index:
                ax1.axvline(recession_start, color='gray', linestyle='--', alpha=0.7)
                ax1.text(recession_start, ax1.get_ylim()[1]*0.9, label, 
                        fontsize=8, rotation=90, ha='right')
        
        # Scatter plot (right)
        ax2 = axs[i, 1]
        ax2.scatter(predicted, actual, alpha=0.5)
        
        # Add regression line
        min_pred, max_pred = predicted.min(), predicted.max()
        x_range = np.linspace(min_pred, max_pred, 100)
        # Simple OLS for the scatter plot trend line
        slope, intercept = np.polyfit(predicted, actual, 1)
        ax2.plot(x_range, slope * x_range + intercept, 'r-')
        
        # Add diagonal line (perfect prediction)
        ax2.plot(x_range, x_range, 'k--', alpha=0.3)
        
        # Annotations
        ax2.set_title(f'Eq. ({eq_num}) - Predicted vs Actual')
        ax2.set_xlabel('Predicted Excess Return')
        ax2.set_ylabel('Actual Excess Return')
        ax2.text(0.05, 0.9, f'R² = {model.rsquared:.3f}', transform=ax2.transAxes)
        
        # Add legend to first row only
        if i == 0:
            ax1.legend()
    
    plt.tight_layout()
    return fig, axs


def OOS_plot_model_predictions(results_set, oos_r2, utils_data: list, maturities: list) -> tuple:
    """
    Plot in-sample and out-of-sample predictions from models (22)-(25) against actual excess returns.
    Left: Time series comparison
    Middle: In-sample Predicted vs Actual scatter plot with regression line
    Right: Out-of-sample Predicted vs Actual scatter plot with regression line
    """
    import matplotlib.pyplot as plt
  
    # Visualize in-sample and out-of-sample predictions vs actuals for each model in a 3x3 grid
    model_keys = list(results_set["models"].keys())
    n_models = len(model_keys)
    fig, axs = plt.subplots(n_models, 3, figsize=(18, 4 * n_models))

    for i, key in enumerate(model_keys):
        model = results_set["models"][key]
        # In-sample
        df_train = results_set["data"][key]
        X_train = df_train.drop(columns=["rx_bar"])
        X_train = sm.add_constant(X_train)
        X_train = X_train.reindex(columns=model.model.exog_names, fill_value=0)
        y_train = df_train["rx_bar"]
        y_pred_train = model.predict(X_train)

        # Out-of-sample
        df_test = get_test_df(key, utils_data, maturities)
        if df_test.empty:
            for j in range(3):
                axs[i, j].set_visible(False)
            continue
        X_test = df_test.drop(columns=["rx_bar"])
        X_test = sm.add_constant(X_test)
        X_test = X_test.reindex(columns=model.model.exog_names, fill_value=0)
        y_test = df_test["rx_bar"]
        y_pred_test = model.predict(X_test)

        # Combine for plotting
        y_all = pd.concat([y_train, y_test])
        y_pred_all = pd.concat([y_pred_train, y_pred_test])
        idx_split = y_train.index[-1]

        # 1. Time series plot (column 0)
        ax = axs[i, 0]
        ax.plot(y_all.index, y_all, label="Actual", color="blue")
        ax.plot(y_all.index, y_pred_all, label="Predicted", color="red", alpha=0.7)
        ax.axvline(idx_split, color="black", linestyle="--", lw=1.2, label="OOS Start")
        ax.set_title(f"Eq ({key}) - Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("rx_bar")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. In-sample scatter (column 1)
        ax = axs[i, 1]
        ax.scatter(y_pred_train, y_train, alpha=0.6, label="In-sample")
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', alpha=0.5)
        ax.set_xlabel("Predicted rx_bar")
        ax.set_ylabel("Actual rx_bar")
        ax.set_title(f"Eq ({key}) - In-sample Scatter (R²={model.rsquared:.3f})")
        ax.grid(alpha=0.3)

        # 3. Out-of-sample scatter (column 2)
        ax = axs[i, 2]
        ax.scatter(y_pred_test, y_test, alpha=0.6, label="OOS")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5)
        ax.set_xlabel("Predicted rx_bar")
        ax.set_ylabel("Actual rx_bar")
        ax.set_title(f"Eq ({key}) - OOS Scatter (R²={oos_r2[key]:.3f})")
        ax.grid(alpha=0.3)

        plt.tight_layout()
    return fig, axs


def run_oos_prediction(results_set: dict,
                       yields: pd.DataFrame,
                       xret: pd.DataFrame,
                       Tau: pd.Series,
                       cycle_models: list,
                       maturities: list,
                       test_start: str,
                       test_end: str) -> tuple:
    """
    Performs out-of-sample (OOS) prediction and calculates OOS R-squared.

    Args:
        results_set (dict): The dictionary of fitted in-sample models.
        yields (pd.DataFrame): Full DataFrame of yields.
        xret (pd.DataFrame): Full DataFrame of excess returns.
        Tau (pd.Series): Full Series for the trend inflation proxy.
        cycle_models (list): List of fitted models for calculating yield cycles.
        maturities (list): List of maturities used.
        test_start (str): The start date for the OOS period (e.g., 'YYYY-MM-DD').
        test_end (str): The end date for the OOS period (e.g., 'YYYY-MM-DD').

    Returns:
        tuple: A tuple containing:
            - oos_r2 (dict): Dictionary of OOS R-squared values for each model.
            - utils_data (list): List of prepared test DataFrames needed for plotting.
    """
    from sklearn.metrics import r2_score

    # 1. Prepare out-of-sample data for the specified period
    yields_test = yields.loc[test_start:test_end].copy()
    xret_test = xret.loc[test_start:test_end].copy()
    Tau_test = Tau.loc[test_start:test_end].copy()

    # 2. Calculate out-of-sample cycles using the in-sample models
    cycles_test = pd.DataFrame(index=yields_test.index, columns=[f'cycl{n}' for n in maturities], dtype=float)
    for n, model_fit in cycle_models:
        col = f'y{n}'
        if col not in yields_test.columns:
            continue
        
        df_pred = pd.concat([yields_test[col], Tau_test], axis=1).dropna()
        X_pred = sm.add_constant(df_pred['Tau'])
        
        fitted_test = model_fit.predict(X_pred)
        cycles_test.loc[df_pred.index, f'cycl{n}'] = df_pred[col] - fitted_test

    # 3. Recompute duration-weighted averages for the test set
    xret_test["rx_bar"] = duration_weighted_average(xret_test, type_name="excess_return")
    y_bar_test = duration_weighted_average(yields_test, type_name="yield")
    c_bar_test = duration_weighted_average(cycles_test, type_name="cycle")
    c1_test = cycles_test["cycl1"] if "cycl1" in cycles_test.columns else pd.Series(dtype=float)

    # 4. Predict and calculate out-of-sample R2 for each model
    oos_r2 = {}
    utils_data = [xret_test, yields_test, Tau_test, y_bar_test, c_bar_test, c1_test]
    
    for key, model in results_set["models"].items():
        df_test = get_test_df(model_key=key, utils_data=utils_data, maturities=maturities)
        if df_test.empty or "rx_bar" not in df_test.columns:
            oos_r2[key] = np.nan
            continue
            
        y_true = df_test["rx_bar"]
        X_test = df_test.drop(columns=["rx_bar"])
        X_test = sm.add_constant(X_test, has_constant='add')
        X_test = X_test.reindex(columns=model.model.exog_names, fill_value=0)
        
        y_pred = model.predict(X_test)
        oos_r2[key] = r2_score(y_true, y_pred)
        
    return oos_r2, utils_data