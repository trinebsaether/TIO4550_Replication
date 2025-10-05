"""
Regression utilities for predictive regressions and model selection.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List
import warnings


def predictive_regression(y: pd.Series, X: pd.DataFrame, 
                         add_constant: bool = True, 
                         hac_lags: int = 18) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run predictive regression with Newey-West HAC standard errors.
    
    Parameters:
    -----------
    y : pd.Series
        Dependent variable.
    X : pd.DataFrame
        Independent variables.
    add_constant : bool
        Whether to add an intercept to the model.
    hac_lags : int
        Number of lags for the HAC correction.
        
    Returns:
    --------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted regression model with HAC standard errors.
    """
    # Align data and handle missing values
    data = pd.concat([y, X], axis=1).dropna()
    
    if data.empty:
        raise ValueError("No valid observations after removing NaNs. Check data alignment.")
    
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    
    # Add constant if requested
    if add_constant:
        X_clean = sm.add_constant(X_clean, prepend=True)
    
    # Fit OLS model with HAC robust standard errors
    model = sm.OLS(y_clean, X_clean).fit(
        cov_type='HAC', 
        cov_kwds={'maxlags': hac_lags}
    )
    
    return model


def model_selection_bic(y: pd.Series, candidate_factors: List[pd.DataFrame],
                       base_regressors: pd.DataFrame = None) -> Dict:
    """
    Select optimal factor specification using BIC criterion
    
    Parameters:
    -----------
    y : pd.Series
        Dependent variable
    candidate_factors : List[pd.DataFrame]
        List of candidate factor sets
    base_regressors : pd.DataFrame
        Base regressors to include in all models
        
    Returns:
    --------
    Dict
        Model selection results
    """
    results = []
    
    for i, factors in enumerate(candidate_factors):
        # Prepare regressors
        if base_regressors is not None:
            X = pd.concat([base_regressors, factors], axis=1)
        else:
            X = factors
        
        # Run regression
        try:
            model = predictive_regression(y, X, add_constant=True)

            
            # Calculate BIC
            bic = model.nobs * np.log(model.ssr / model.nobs) + model.df_model * np.log(model.nobs)
            
            results.append({
                'specification': i,
                'factors': factors.columns.tolist(),
                'bic': bic,
                'rsquared': model.rsquared,
                'rsquared_adj': model.rsquared_adj,
                'nobs': model.nobs,
                'model': model
            })
            
        except Exception as e:
            warnings.warn(f"Error fitting specification {i}: {e}")
            continue
    
    if not results:
        raise ValueError("No valid specifications could be fitted")
    
    # Find best specification (minimum BIC)
    best_idx = np.argmin([r['bic'] for r in results])
    best_spec = results[best_idx]
    
    return {
        'best_specification': best_spec,
        'all_results': results,
        'selection_criterion': 'BIC'
    }


def create_summary_factor(y: pd.Series, factors: pd.DataFrame, 
                         factor_names: List[str]) -> pd.Series:
    """
    Create single summary factor as fitted values from regression
    
    Following Cochrane-Piazzesi methodology, create a single factor
    as the fitted value from regressing returns on multiple factors.
    
    Parameters:
    -----------
    y : pd.Series
        Dependent variable (e.g., average excess returns)
    factors : pd.DataFrame
        Factor data
    factor_names : List[str]
        Names of factors to include
        
    Returns:
    --------
    pd.Series
        Summary factor (fitted values)
    """
    # Select factors
    X = factors[factor_names]
    
    # Run regression
    model = predictive_regression(y, X, add_constant=True)
    
    # Return fitted values as summary factor
    fitted_values = model.fittedvalues
    fitted_values.name = f"F{len(factor_names)}"
    
    return fitted_values