import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.decomposition import PCA
import warnings
from .data_processing import standardize_data, prepare_balanced_panel

def extract_factors_pca(data: pd.DataFrame, n_factors: int, 
                       standardize: bool = True) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Extract factors using Principal Components Analysis following LN methodology
    
    The factors are √T times the eigenvectors corresponding to the r largest 
    eigenvalues of the T×T matrix XX'/(TN), following LN normalization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Panel data (T x N) - should be balanced and transformed
    n_factors : int
        Number of factors to extract
    standardize : bool
        Whether to standardize data before PCA
        
    Returns:
    --------
    tuple
        - factors: DataFrame with estimated factors (T x r)
        - loadings: Array of factor loadings (N x r)  
        - results: Dict with additional results
    """
    # Ensure reasonably balanced panel and optional standardization
    balanced = prepare_balanced_panel(data)
    T, N = balanced.shape
    if standardize:
        data_std, _ = standardize_data(balanced, method='zscore')
    else:
        data_std = balanced.copy()
    # Handle missing values uniformly
    if data_std.isna().any().any():
        warnings.warn("Missing values detected. Using available data for each time period.")
        data_clean = data_std.fillna(0)
    else:
        data_clean = data_std
    
    # Use sklearn PCA; rescale scores to LN normalization
    X = data_clean.values
    pca = PCA(n_components=n_factors, svd_solver='full')
    scores = pca.fit_transform(X)  # = U @ S
    singular_values = pca.singular_values_  # length r
    # U = scores / S; LN factors = sqrt(T) * U
    U = scores / singular_values
    factors_array = U * np.sqrt(T)
    
    # Create factor DataFrame
    factor_names = [f'F{i+1}' for i in range(n_factors)]
    factors_df = pd.DataFrame(
        factors_array, 
        index=data.index, 
        columns=factor_names
    )
    
    # Calculate loadings: Λ = X'F/T
    loadings = (X.T @ factors_array) / T
    
    # Explained variance (from sklearn)
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    # Calculate common component R²
    r_squared = np.zeros(n_factors)
    for i in range(n_factors):
        factors_subset = factors_array[:, :i+1]
        loadings_subset = loadings[:, :i+1]
        fitted = factors_subset @ loadings_subset.T
        
        # R² as fraction of total variance explained
        total_variance = np.sum(np.var(X, axis=0))
        explained_variance = np.sum(np.var(fitted, axis=0))
        r_squared[i] = explained_variance / total_variance
    
    # Store results
    results = {
        'eigenvalues': pca.explained_variance_,
        'explained_variance_ratio': explained_var_ratio,
        'cumulative_variance_ratio': cumulative_var_ratio,
        'r_squared': r_squared,
        'total_variance_explained': cumulative_var_ratio[-1],
        'n_series': N,
        'n_observations': T
    }
    
    return factors_df, loadings, results


def extract_yield_factors(yields: pd.DataFrame, n_factors: int = 3) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Extract yield curve factors (level, slope, curvature) using PCA
    
    Parameters:
    -----------
    yields : pd.DataFrame
        Yield curve data with maturity columns
    n_factors : int
        Number of yield factors to extract (typically 3)
        
    Returns:
    --------
    tuple
        - yield_factors: DataFrame with yield PCs
        - loadings: Factor loadings
        - results: Additional results
    """
    # Delegate to generic PCA extractor (handles balancing, standardization, LN scaling)
    factors, loadings, results = extract_factors_pca(yields, n_factors=n_factors, standardize=True)
    # Rename factors to PC1..PCn for yield-curve convention
    factor_names = [f'PC{i+1}' for i in range(n_factors)]
    factors = factors.set_axis(factor_names, axis=1)
    # Annotate interpretation
    results = {
        **results,
        'factor_interpretation': {
            'PC1': 'Level',
            'PC2': 'Slope',
            'PC3': 'Curvature'
        }
    }
    return factors, loadings, results
