"""
Here we place functions slated for removal with the first commit to GitHub.
"""

"""First, regression.py"""

# def create_factor_combinations(factors: pd.DataFrame, max_factors: int = 6,
#                               include_nonlinear: bool = True) -> List[pd.DataFrame]:
#     """
#     Create different combinations of factors for model selection
    
#     Parameters:
#     -----------
#     factors : pd.DataFrame
#         Available factors
#     max_factors : int
#         Maximum number of factors in any combination
#     include_nonlinear : bool
#         Whether to include nonlinear terms
        
#     Returns:
#     --------
#     List[pd.DataFrame]
#         List of factor combinations
#     """
#     from itertools import combinations
    
#     factor_combinations = []
#     factor_names = factors.columns.tolist()
    
#     # Linear combinations
#     for r in range(1, min(max_factors + 1, len(factor_names) + 1)):
#         for combo in combinations(factor_names, r):
#             factor_combinations.append(factors[list(combo)])
    
#     # Add nonlinear terms if requested
#     if include_nonlinear:
#         # Add squared and cubed terms for first factor
#         if len(factor_names) > 0:
#             first_factor = factor_names[0]
            
#             # Combinations with squared term
#             for r in range(1, min(max_factors, len(factor_names) + 1)):
#                 for combo in combinations(factor_names, r):
#                     if first_factor in combo:
#                         extended_factors = factors[list(combo)].copy()
#                         extended_factors[f'{first_factor}^2'] = factors[first_factor] ** 2
#                         factor_combinations.append(extended_factors)
            
#             # Combinations with cubed term  
#             for r in range(1, min(max_factors - 1, len(factor_names) + 1)):
#                 for combo in combinations(factor_names, r):
#                     if first_factor in combo:
#                         extended_factors = factors[list(combo)].copy()
#                         extended_factors[f'{first_factor}^3'] = factors[first_factor] ** 3
#                         factor_combinations.append(extended_factors)
    
#     return factor_combinations

# def diagnostic_tests(model) -> Dict:
#     """
#     Run diagnostic tests on regression model
    
#     Parameters:
#     -----------
#     model : statsmodels result
#         Fitted regression model
        
#     Returns:
#     --------
#     Dict
#         Diagnostic test results
#     """
#     diagnostics = {}
    
#     # White test for heteroskedasticity
#     try:
#         white_lm, white_lm_pvalue, white_fvalue, white_f_pvalue = het_white(model.resid, model.model.exog)
#         diagnostics['white_test'] = {
#             'lm_statistic': white_lm,
#             'lm_pvalue': white_lm_pvalue,
#             'f_statistic': white_fvalue,
#             'f_pvalue': white_f_pvalue
#         }
#     except:
#         diagnostics['white_test'] = None
    
#     # Jarque-Bera test for normality
#     try:
#         jb_stat, jb_pvalue = stats.jarque_bera(model.resid)
#         diagnostics['jarque_bera'] = {
#             'statistic': jb_stat,
#             'pvalue': jb_pvalue
#         }
#     except:
#         diagnostics['jarque_bera'] = None
    
#     # Durbin-Watson test for serial correlation
#     try:
#         from statsmodels.stats.stattools import durbin_watson
#         dw_stat = durbin_watson(model.resid)
#         diagnostics['durbin_watson'] = dw_stat
#     except:
#         diagnostics['durbin_watson'] = None
    
#     return diagnostics

# def wald_test(model_restricted, model_unrestricted, 
#               use_hac: bool = True, hac_lags: int = 18) -> Dict:
#     """
#     Perform Wald test comparing restricted vs unrestricted models
    
#     Parameters:
#     -----------
#     model_restricted : statsmodels result
#         Restricted model
#     model_unrestricted : statsmodels result  
#         Unrestricted model
#     use_hac : bool
#         Whether to use HAC covariance matrix
#     hac_lags : int
#         Lags for HAC correction
        
#     Returns:
#     --------
#     Dict
#         Wald test results
#     """
#     # Get coefficient vectors
#     beta_r = model_restricted.params.values
#     beta_ur = model_unrestricted.params.values
    
#     # Number of restrictions
#     q = len(beta_ur) - len(beta_r)
    
#     if q <= 0:
#         raise ValueError("Unrestricted model must have more parameters than restricted")
    
#     # Get covariance matrix
#     if use_hac:
#         # Compatibility with different statsmodels versions
#         try:
#             cov_ur = cov_hac(model_unrestricted, maxlags=hac_lags, use_correction=True)
#         except TypeError:
#             cov_ur = cov_hac(model_unrestricted, nlags=hac_lags, use_correction=True)
#         except Exception:
#             cov_ur = model_unrestricted.cov_params().values
#     else:
#         cov_ur = model_unrestricted.cov_params().values
    
#     # Extract relevant part of covariance matrix (for additional parameters)
#     cov_additional = cov_ur[-q:, -q:]
#     beta_additional = beta_ur[-q:]
    
#     # Calculate Wald statistic
#     try:
#         wald_stat = beta_additional.T @ np.linalg.inv(cov_additional) @ beta_additional
#     except np.linalg.LinAlgError:
#         warnings.warn("Singular covariance matrix in Wald test")
#         wald_stat = np.nan
    
#     # Calculate p-value
#     pvalue = 1 - stats.chi2.cdf(wald_stat, df=q)
    
#     return {
#         'wald_statistic': wald_stat,
#         'df': q,
#         'pvalue': pvalue,
#         'critical_values': {
#             '10%': stats.chi2.ppf(0.9, df=q),
#             '5%': stats.chi2.ppf(0.95, df=q),
#             '1%': stats.chi2.ppf(0.99, df=q)
#         }
#     }

"""Second, regression.py"""


# def newey_west_se(model, maxlags: int = 18, use_correction: bool = True) -> np.ndarray:
#     """
#     Calculate Newey-West HAC standard errors
    
#     Parameters:
#     -----------
#     model : statsmodels regression result
#         Fitted regression model
#     maxlags : int
#         Maximum number of lags for HAC correction
#     use_correction : bool
#         Whether to use small sample correction
        
#     Returns:
#     --------
#     np.ndarray
#         HAC standard errors
#     """
#     try:
#         # Calculate HAC covariance matrix (statsmodels >=0.13)
#         cov_matrix = cov_hac(model, maxlags=maxlags, use_correction=use_correction)
#     except TypeError:
#         # Fallback for versions expecting 'nlags'
#         try:
#             cov_matrix = cov_hac(model, nlags=maxlags, use_correction=use_correction)
#         except Exception as e:
#             warnings.warn(f"Error calculating Newey-West standard errors: {e}")
#             return model.bse.values
#     except Exception as e:
#         warnings.warn(f"Error calculating Newey-West standard errors: {e}")
#         return model.bse.values

#     # Extract standard errors
#     se = np.sqrt(np.diag(cov_matrix))
#     return se


# def predictive_regression(y: pd.Series, X: pd.DataFrame, 
#                          add_constant: bool = True, 
#                          hac_lags: int = 18) -> Dict:
#     """
#     Run predictive regression with HAC standard errors
    
#     Parameters:
#     -----------
#     y : pd.Series
#         Dependent variable
#     X : pd.DataFrame
#         Independent variables
#     add_constant : bool
#         Whether to add intercept
#     hac_lags : int
#         Number of lags for HAC correction
        
#     Returns:
#     --------
#     Dict
#         Regression results
#     """
#     # Align data
#     common_index = y.index.intersection(X.index)
#     y_aligned = y.loc[common_index].dropna()
#     X_aligned = X.loc[y_aligned.index]
    
#     # Remove any rows with missing X values
#     complete_data = pd.concat([y_aligned, X_aligned], axis=1).dropna()
#     y_clean = complete_data.iloc[:, 0]
#     X_clean = complete_data.iloc[:, 1:]
    
#     if len(y_clean) == 0:
#         raise ValueError("No valid observations after cleaning data")
    
#     # Add constant if requested
#     if add_constant:
#         X_clean = sm.add_constant(X_clean)
    
#     # Fit regression
#     model = sm.OLS(y_clean, X_clean).fit()
    
#     # Calculate HAC standard errors
#     hac_se = newey_west_se(model, maxlags=hac_lags)
    
#     # Calculate HAC t-statistics
#     hac_tstat = model.params.values / hac_se
    
#     # Calculate HAC p-values (two-tailed)
#     hac_pvalues = 2 * (1 - stats.t.cdf(np.abs(hac_tstat), model.df_resid))
    
#     # Store results
#     results = {
#         'model': model,
#         'coefficients': model.params,
#         'se_ols': model.bse,
#         'se_hac': pd.Series(hac_se, index=model.params.index),
#         'tstat_ols': model.tvalues,
#         'tstat_hac': pd.Series(hac_tstat, index=model.params.index),
#         'pvalues_ols': model.pvalues,
#         'pvalues_hac': pd.Series(hac_pvalues, index=model.params.index),
#         'rsquared': model.rsquared,
#         'rsquared_adj': model.rsquared_adj,
#         'nobs': model.nobs,
#         'df_resid': model.df_resid,
#         'residuals': model.resid,
#         'fitted_values': model.fittedvalues
#     }
    
#     return results

# def predictive_regression(y: pd.Series, X: pd.DataFrame, 
#                          add_constant: bool = True, 
#                          hac_lags: int = 18) -> Dict:
#     """ (This is an update of the function above where we use built-in standard errors)
#     Run predictive regression with Newey-West HAC standard errors.
    
#     Parameters:
#     -----------
#     y : pd.Series
#         Dependent variable.
#     X : pd.DataFrame
#         Independent variables.
#     add_constant : bool
#         Whether to add an intercept to the model.
#     hac_lags : int
#         Number of lags for the HAC correction.
        
#     Returns:
#     --------
#     Dict
#         A dictionary containing detailed regression results, including
#         both OLS and HAC-corrected statistics.
#     """
#     # Align data and handle missing values
#     data = pd.concat([y, X], axis=1).dropna()
    
#     if data.empty:
#         raise ValueError("No valid observations after removing NaNs. Check data alignment.")
    
#     y_clean = data.iloc[:, 0]
#     X_clean = data.iloc[:, 1:]
    
#     # Add constant if requested
#     if add_constant:
#         X_clean = sm.add_constant(X_clean, prepend=True)
    
#     # Fit OLS model with HAC robust standard errors
#     hac_model = sm.OLS(y_clean, X_clean).fit(
#         cov_type='HAC', 
#         cov_kwds={'maxlags': hac_lags}
#     )
    
#     # Fit a standard OLS model to get the OLS-specific stats
#     ols_model = sm.OLS(y_clean, X_clean).fit()

#     # Store results in a dictionary
#     results = {
#         'model': hac_model,  # The main model object has HAC results
#         'coefficients': hac_model.params,
#         'se_ols': ols_model.bse,
#         'se_hac': hac_model.bse,
#         'tstat_ols': ols_model.tvalues,
#         'tstat_hac': hac_model.tvalues,
#         'pvalues_ols': ols_model.pvalues,
#         'pvalues_hac': hac_model.pvalues,
#         'rsquared': hac_model.rsquared,
#         'rsquared_adj': hac_model.rsquared_adj,
#         'nobs': hac_model.nobs,
#         'df_resid': hac_model.df_resid,
#         'residuals': hac_model.resid,
#         'fitted_values': hac_model.fittedvalues
#     }
    
#     return results