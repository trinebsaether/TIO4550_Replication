"""
Yield curve and bond return processing utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import statsmodels.api as sm


def calculate_yields_from_prices(prices: pd.DataFrame, maturities: List[int]) -> pd.DataFrame:
    """
    Calculate yields from bond prices
    
    For zero-coupon bonds: y = -log(P)/n * 100
    where P is price (as fraction of par) and n is maturity in years
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Bond prices (as percentage of par value)
    maturities : List[int]
        Maturities in years corresponding to price columns
        
    Returns:
    --------
    pd.DataFrame
        Annualized yields in percentage points
    """
    yields = pd.DataFrame(index=prices.index)
    
    for i, maturity in enumerate(maturities):
        price_col = prices.columns[i]
        yield_col = f'y{maturity}'
        
        # Convert price from percentage to fraction
        price_fraction = prices[price_col] / 100
        
        # Calculate yield: y = -log(P)/n * 100
        yields[yield_col] = -np.log(price_fraction) / maturity * 100
    
    return yields


def calculate_forward_rates(yields: pd.DataFrame, maturities: List[int]) -> pd.DataFrame:
    """
    Calculate forward rates from yields
    
    Forward rate f(n,1) = n*y(n) - (n-1)*y(n-1)
    
    Parameters:
    -----------
    yields : pd.DataFrame
        Yield data
    maturities : List[int]
        Maturities in years
        
    Returns:
    --------
    pd.DataFrame
        Forward rates
    """
    forwards = pd.DataFrame(index=yields.index)
    
    # Check which yields are actually available
    available_cols = [f'y{m}' for m in maturities if f'y{m}' in yields.columns]
    available_mats = [int(col[1:]) for col in available_cols]
    available_mats.sort()
    
    if len(available_mats) == 0:
        raise ValueError("No yield columns found matching the specified maturities")
    
    # If we have y1, f1 = y1
    if 1 in available_mats and f'y1' in yields.columns:
        forwards['f1'] = yields['y1']
    
    # Calculate forward rates for all consecutive maturity pairs
    for i in range(1, len(available_mats)):
        n = available_mats[i]
        n_prev = available_mats[i-1] 
        
        forward_col = f'f{n}'
        forwards[forward_col] = (n * yields[f'y{n}'] - n_prev * yields[f'y{n_prev}'])
    
    return forwards


def calculate_excess_returns(
    yields: pd.DataFrame,
    maturities: List[int],
    horizon: int = 12,
    risk_free_col: str = None,
    period: str = "annual",
) -> pd.DataFrame:
    """Calculate excess bond returns for annual or monthly period.

    period: "annual" uses a horizon in months (default 12). "monthly" uses a 1-month holding period.
    """
    excess_returns = pd.DataFrame(index=yields.index)
    
    # Determine risk-free rate column
    if risk_free_col is None:
        if period == "monthly" and 'y1m' in yields.columns:
            rf_col = 'y1m'
        else:
            rf_maturity = min(maturities)
            rf_col = f'y{rf_maturity}'
    else:
        rf_col = risk_free_col

    if period == "monthly":
        # 1-month holding period approximation
        for maturity in maturities:
            if maturity < 1:
                continue
            y_col = f'y{maturity}'
            xr_col = f'xr1m{maturity}'
            y_n_t = yields[y_col]
            y_n_t1 = yields[y_col].shift(-1)
            rf_t = yields[rf_col]
            remaining_maturity = maturity - 1/12
            excess_returns[xr_col] = (
                -remaining_maturity * y_n_t1 + maturity * y_n_t - (1/12) * rf_t
            ) / 12
        xr_cols = [c for c in excess_returns.columns if c.startswith('xr1m')]
        if xr_cols:
            excess_returns['xr1m_avg'] = excess_returns[xr_cols].mean(axis=1)
        return excess_returns

    # Annual (generic h-month) holding period
    horizon_years = horizon / 12
    for maturity in maturities:
        if maturity <= horizon_years:
            continue
        y_col = f'y{maturity}'
        xr_col = f'xr{maturity}'
        y_n_t = yields[y_col]
        y_h_t = yields[rf_col]
        remaining_maturity = maturity - horizon_years
        remaining_maturity_int = int(remaining_maturity) if remaining_maturity == int(remaining_maturity) else remaining_maturity
        if remaining_maturity_int in maturities:
            y_nh_col = f'y{remaining_maturity_int}'
            y_nh_th = yields[y_nh_col].shift(-horizon)
        else:
            y_nh_th = yields[y_col].shift(-horizon)
        excess_returns[xr_col] = (
            -(remaining_maturity) * y_nh_th + maturity * y_n_t - horizon_years * y_h_t
        )
    xr_cols = [col for col in excess_returns.columns if col.startswith('xr')]
    if xr_cols:
        excess_returns['xr_avg'] = excess_returns[xr_cols].mean(axis=1)
    return excess_returns


def calculate_monthly_excess_returns(yields: pd.DataFrame, maturities: List[int]) -> pd.DataFrame:
    """Wrapper for monthly excess returns (1-month holding period)."""
    return calculate_excess_returns(yields, maturities, period="monthly")


def create_cochrane_piazzesi_factor(yields: pd.DataFrame, forward_rates: pd.DataFrame = None, 
                                   method: str = 'original', verbose: bool = False,
                                   risk_free_col: str = 'y1') -> Dict:
    """
    Minimal Cochrane–Piazzesi factor.
    Returns a `pd.Series` named 'CP' with fitted values from regressing
    average 12-month excess returns (y2–y5) on selected forward rates.

    method: 'original' uses f2–f5 (canonical CP). Any other value includes f1–f5.
    """
    # Ensure forward rates
    if forward_rates is None:
        available_maturities = sorted([int(col[1:]) for col in yields.columns if col.startswith('y')])
        forward_rates = calculate_forward_rates(yields, available_maturities)

    # Target: avg excess return over maturities 2–5
    maturities = [2, 3, 4, 5]
    xr = calculate_excess_returns(yields, maturities, horizon=12, risk_free_col=risk_free_col)
    y_target = xr[[f'xr{m}' for m in maturities]].mean(axis=1)

    # Features: forward rates per method, using only available columns
    desired = ['f2','f3','f4','f5'] if method == 'original' else ['f1','f2','f3','f4','f5']
    forward_cols = [c for c in desired if c in forward_rates.columns]
    if not forward_cols:
        return {
            'cp_factor': pd.Series(index=y_target.index, name='CP'),
            'forward_rates': forward_rates,
            'excess_returns': xr,
            'avg_excess_return': y_target,
            'forward_cols': [],
            'method': method
        }

    X = forward_rates[forward_cols].dropna()
    idx = y_target.index.intersection(X.index)
    yv = y_target.loc[idx].dropna()
    Xv = X.loc[yv.index]
    if len(yv) == 0:
        return {
            'cp_factor': pd.Series(index=y_target.index, name='CP'),
            'forward_rates': forward_rates,
            'excess_returns': xr,
            'avg_excess_return': y_target,
            'forward_cols': forward_cols,
            'method': method
        }

    # OLS with intercept via statsmodels
    X_sm = sm.add_constant(Xv, has_constant='add')
    model = sm.OLS(yv, X_sm, missing='drop')
    results = model.fit()
    cp_series = pd.Series(results.fittedvalues, index=results.fittedvalues.index, name='CP')
    return {
        'cp_factor': cp_series,
        'forward_rates': forward_rates,
        'excess_returns': xr,
        'avg_excess_return': y_target,
        'forward_cols': forward_cols,
        'coefficients': results.params.values,
        'statsmodels_results': results,
        'method': method
    }


def prepare_yield_data(data: pd.DataFrame, price_cols: List[str] = None, 
                      yield_cols: List[str] = None, maturities: List[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Prepare comprehensive yield dataset with yields, forwards, and returns
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw bond data (prices or yields)
    price_cols : List[str]
        Column names for bond prices (if providing prices)
    yield_cols : List[str] 
        Column names for yields (if providing yields directly)
    maturities : List[int]
        Maturities in years
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing yields, forwards, and excess returns
    """
    if maturities is None:
        maturities = [1, 2, 3, 4, 5]
    
    # Calculate yields from prices if needed
    if price_cols is not None:
        prices = data[price_cols]
        yields = calculate_yields_from_prices(prices, maturities)
    elif yield_cols is not None:
        yields = data[yield_cols].copy()
        # Rename columns to standard format
        for i, col in enumerate(yield_cols):
            yields = yields.rename(columns={col: f'y{maturities[i]}'})
    else:
        # Assume data already has yield columns in standard format
        yields = data.copy()
    
    # Calculate forward rates
    forwards = calculate_forward_rates(yields, maturities)
    
    # Calculate excess returns (annual)
    excess_returns_annual = calculate_excess_returns(yields, maturities, horizon=12)
    
    # Calculate monthly excess returns
    excess_returns_monthly = calculate_monthly_excess_returns(yields, maturities)
    
    # Create Cochrane-Piazzesi factor
    cp_factor = create_cochrane_piazzesi_factor(yields, forwards)
    
    # Combine all data
    combined_data = pd.concat([
        yields,
        forwards, 
        excess_returns_annual,
        excess_returns_monthly,
        cp_factor
    ], axis=1)
    
    return {
        'yields': yields,
        'forwards': forwards,
        'excess_returns_annual': excess_returns_annual,
        'excess_returns_monthly': excess_returns_monthly,
        'cp_factor': cp_factor,
        'combined': combined_data
    }


