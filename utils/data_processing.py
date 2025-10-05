"""
Data processing utilities for replication
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import os


def standardize_data(data: pd.DataFrame, method: str = 'zscore') -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize data for factor analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    method : str
        Standardization method ('zscore' or 'demean')
        
    Returns:
    --------
    tuple
        Standardized data and scaling parameters
    """
    if method == 'zscore':
        means = data.mean()
        stds = data.std()
        standardized = (data - means) / stds
        scaling_params = {'means': means, 'stds': stds, 'method': 'zscore'}
    elif method == 'demean':
        means = data.mean()
        standardized = data - means
        scaling_params = {'means': means, 'method': 'demean'}
    else:
        raise ValueError("Method must be 'zscore' or 'demean'")
        
    return standardized, scaling_params


def load_wrds_famabliss(file_path: str, sample_start: str = '1964-01', 
                        sample_end: str = '2003-12') -> pd.DataFrame:
    """
    Load and process WRDS Fama-Bliss bond data
    
    Parameters:
    -----------
    file_path : str
        Path to WRDS Fama-Bliss CSV file
    sample_start : str
        Start of sample period (YYYY-MM format)
    sample_end : str
        End of sample period (YYYY-MM format)
        
    Returns:
    --------
    pd.DataFrame
        Processed bond yields with columns y1, y2, y3, y4, y5
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean data (remove rows with NaN)
    df_clean = df.dropna()
    
    # Convert date column
    df_clean = df_clean.copy()
    df_clean['MCALDT'] = pd.to_datetime(df_clean['MCALDT'])
    
    # Filter to sample period
    start_date = pd.to_datetime(sample_start + '-01')
    end_date = pd.to_datetime(sample_end + '-01') + pd.offsets.MonthEnd()
    
    df_sample = df_clean[(df_clean['MCALDT'] >= start_date) & 
                        (df_clean['MCALDT'] <= end_date)].copy()
    
    # Maturity mapping: TTERMTYPE to years
    maturity_map = {
        5001: 1,  # 1-year
        5002: 2,  # 2-year  
        5003: 3,  # 3-year
        5004: 4,  # 4-year
        5005: 5   # 5-year
    }
    
    # Pivot data to get yields by maturity
    yields_wide = df_sample.pivot_table(
        index='MCALDT', 
        columns='TTERMTYPE', 
        values='TMYTM',
        aggfunc='first'  # Take first value if multiple per date
    )
    
    # Rename columns to standard format
    yield_columns = {}
    for ttermtype, years in maturity_map.items():
        if ttermtype in yields_wide.columns:
            yield_columns[ttermtype] = f'y{years}'
    
    yields_wide = yields_wide.rename(columns=yield_columns)
    
    # Ensure we have all required maturities
    required_cols = ['y1', 'y2', 'y3', 'y4', 'y5']
    for col in required_cols:
        if col not in yields_wide.columns:
            print(f"Warning: Missing maturity {col}")
    
    # Keep only standard yield columns
    available_cols = [col for col in required_cols if col in yields_wide.columns]
    yields_final = yields_wide[available_cols].copy()
    
    # Convert to monthly frequency (take end-of-month values)
    yields_monthly = yields_final.resample('M').last()
    
    # Convert yields from percentage points to decimal form
    # WRDS data is typically in percentage points (e.g., 2.5 for 2.5%)
    # Convert to decimal form (e.g., 0.025 for 2.5%)
    yields_monthly = yields_monthly / 100
    
    print(f"Loaded WRDS Fama-Bliss data: {yields_monthly.shape}")
    print(f"Date range: {yields_monthly.index.min()} to {yields_monthly.index.max()}")
    print(f"Available maturities: {list(yields_monthly.columns)}")
    # print(f"Sample yields (first 3 observations):")
    # print(yields_monthly.head(3))
    
    return yields_monthly


def prepare_balanced_panel(data: pd.DataFrame, min_obs_ratio: float = 0.8) -> pd.DataFrame:
    """
    Prepare balanced panel by removing series with too many missing values
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input panel data
    min_obs_ratio : float
        Minimum ratio of non-missing observations required
        
    Returns:
    --------
    pd.DataFrame
        Balanced panel data
    """
    n_total = len(data)
    min_obs = int(n_total * min_obs_ratio)
    
    # Count non-missing observations per series
    obs_counts = data.count()
    
    # Keep series with sufficient observations
    valid_series = obs_counts[obs_counts >= min_obs].index
    
    # Filter data
    balanced_data = data[valid_series].copy()
    
    # Remove any remaining rows with all missing values
    balanced_data = balanced_data.dropna(how='all')
    
    print(f"Kept {len(valid_series)} out of {len(data.columns)} series")
    print(f"Final panel dimensions: {balanced_data.shape}")
    
    return balanced_data


def load_updated_fhat(path: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load updated f-hat factors from CSV or Excel and return columns F1..F8 with monthly PeriodIndex.
    Tries to auto-detect date column and factor columns.
    """
    _, ext = os.path.splitext(path.lower())
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Find date column
    date_col = None
    for cand in ['date', 'Date', 'DATES', 'date', 'month', 'Month']:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # If first column looks like date, use it
        date_col = df.columns[0]

    # Parse date flexibly
    try:
        parsed = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    except Exception:
        parsed = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    df = df.loc[~parsed.isna()].copy()
    df.index = pd.to_datetime(parsed[~parsed.isna()])
    df = df.drop(columns=[date_col], errors='ignore').sort_index()

    # Identify factor columns
    # Prefer named F1..F8 or f1..f8
    factor_cols = []
    for i in range(1, 9):
        for name in [f'F{i}', f'f{i}']:
            if name in df.columns:
                factor_cols.append(name)
                break
    if len(factor_cols) < 8:
        # Take first 8 numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        factor_cols = numeric_cols[:8]
        if len(factor_cols) == 0:
            raise ValueError("No numeric factor columns found in updated fhat file")

    factors = df[factor_cols].copy()
    # Rename to F1..F8
    rename_map = {col: f'F{i+1}' for i, col in enumerate(factor_cols)}
    factors = factors.rename(columns=rename_map)

    # Restrict window
    if start is not None:
        factors = factors[factors.index >= pd.to_datetime(start + '-01')]
    if end is not None:
        factors = factors[factors.index <= pd.to_datetime(end + '-01')]

    # PeriodIndex(M)
    factors.index = factors.index.to_period('M')
    return factors


def load_lw_daily_yields(file_path: str, start: str = None, end: str = None, max_maturity=5, include_day=False) -> pd.DataFrame:
    """
    Fetch and preprocess LW daily yields, returning monthly end-of-month data.
    Conforms to the output of load_fed_gsw_daily_yields.

    Args:
        file_path (str): Path to the CSV file containing LW yields data.
        start (str, optional): Start date for the data window (e.g., 'YYYY-MM-DD').
        end (str, optional): End date for the data window (e.g., 'YYYY-MM-DD').
        maturities (list, optional): List of maturities in years to extract.

    Returns:
        pd.DataFrame: Preprocessed LW yields data, resampled to monthly, with columns y1, y2, etc.
    """
    maturities = list(range(1, max_maturity+1))
    lw_yields = pd.read_csv(file_path, comment='%', index_col=0, parse_dates=True)
    lw_yields.columns = lw_yields.columns.str.strip()

    # Map year maturities to month-based column names (e.g., 1 -> '12 m')
    # and create the rename mapping to 'yN' (e.g., '12 m' -> 'y1')
    col_map = {f"{12 * i} m": f'y{i}' for i in maturities}
    
    # Filter for available columns and rename
    available_cols = {k: v for k, v in col_map.items() if k in lw_yields.columns}
    if not available_cols:
        raise ValueError(f"No matching maturities found for years {maturities}. Available columns: {list(lw_yields.columns)}")

    yields = lw_yields[list(available_cols.keys())].rename(columns=available_cols)

    # Convert to numeric, coercing errors
    for c in yields.columns:
        yields[c] = pd.to_numeric(yields[c], errors='coerce')

    # Resample to month-end
    monthly = yields.resample('M').last()

    # Restrict window
    if start is not None:
        monthly = monthly[monthly.index >= pd.to_datetime(start)]
    if end is not None:
        end_date = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        monthly = monthly[monthly.index <= end_date]
        
    # PeriodIndex
    if include_day:
        monthly.index = monthly.index.to_period('D')
    else:
        monthly.index = monthly.index.to_period('M')


    # to decimal
    return monthly/100


def load_fed_gsw_daily_yields(file_path: str, start: str = None, end: str = None, max_maturity=5, include_day=False) -> pd.DataFrame:
    """
    Load FED GSW daily file and extract zero-coupon yields SVENY01..SVENY05.
    Returns monthly end-of-month y1..y5 (percent) with PeriodIndex(M).
    """
    maturities = list(range(1, max_maturity+1))
    df = pd.read_csv(file_path, comment='%')
    if 'Date' not in df.columns:
        raise ValueError("Expected 'Date' column in FED_GSW_daily.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()

    # Map SVENYXX to yN
    sven_cols = {f'SVENY{str(i).zfill(2)}': f'y{i}' for i in maturities}
    available = {k: v for k, v in sven_cols.items() if k in df.columns}
    if len(available) < 3:
        raise ValueError(f"Insufficient SVENY maturities for 1-5y. Found: {list(available.keys())}")

    yields = df[list(available.keys())].rename(columns=available)
    # Convert to numeric, handle 'NA'
    for c in yields.columns:
        yields[c] = pd.to_numeric(yields[c], errors='coerce')

    # Resample to month-end
    monthly = yields.resample('M').last()

    # Restrict window
    if start is not None:
        monthly = monthly[monthly.index >= pd.to_datetime(start)]
    if end is not None:
        end_date = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        monthly = monthly[monthly.index <= end_date]
        
    # PeriodIndex
    if include_day:
        monthly.index = monthly.index.to_period('D')
    else:
        monthly.index = monthly.index.to_period('M')

    # to monthly
    return monthly/100

# =============================
# Utilities for FRED-MD dataset 
# =============================

def load_fred_md_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load FRED-MD dataset with proper handling of transformation codes
    
    Parameters:
    -----------
    filepath : str
        Path to FRED-MD CSV file
        
    Returns:
    --------
    tuple
        Raw data and transformation codes
    """
    # Read the full file
    full_data = pd.read_csv(filepath)
    
    # Extract transformation codes (second row)
    transform_codes = full_data.iloc[0, 1:].astype(int)
    transform_codes.name = 'transform_codes'
    
    # Extract data (third row onwards)
    data = full_data.iloc[1:].copy()
    data = data.reset_index(drop=True)
    
    # Convert date column
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    data = data.set_index('date')
    
    # Convert all other columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data, transform_codes


def apply_fred_transformations(data: pd.DataFrame, transform_codes: pd.Series) -> pd.DataFrame:
    """
    Apply FRED-MD transformation codes to data
    
    Transform codes:
    1 = no transformation (levels)
    2 = first difference
    3 = second difference  
    4 = log
    5 = log first difference
    6 = log second difference
    7 = delta(x_t/x_{t-1} - 1)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data with variables in columns
    transform_codes : pd.Series
        Transformation codes for each variable
        
    Returns:
    --------
    pd.DataFrame
        Transformed data
    """
    transformed = data.copy()
    
    for col in data.columns:
        if col not in transform_codes.index:
            continue
            
        code = transform_codes[col]
        series = data[col].copy()
        
        # Handle missing values
        if series.isna().all():
            continue
            
        try:
            if code == 1:  # Levels
                transformed[col] = series
            elif code == 2:  # First difference
                transformed[col] = series.diff()
            elif code == 3:  # Second difference
                transformed[col] = series.diff().diff()
            elif code == 4:  # Log
                # Only take log of positive values
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series)
                else:
                    transformed[col] = np.nan
            elif code == 5:  # Log first difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff()
                else:
                    transformed[col] = np.nan
            elif code == 6:  # Log second difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff().diff()
                else:
                    transformed[col] = np.nan
            elif code == 7:  # Delta(x_t/x_{t-1} - 1)
                transformed[col] = (series / series.shift(1) - 1).diff()
            else:
                warnings.warn(f"Unknown transformation code {code} for variable {col}")
                transformed[col] = series
                
        except Exception as e:
            warnings.warn(f"Error transforming {col} with code {code}: {e}")
            transformed[col] = np.nan
            
    return transformed

def get_ln_grouping():
    """
    Return (ordered_series, series_to_group) per updated LN Data Appendix mapping
    suitable for grouping plots. Names use LN mnemonics (lowercase) as provided.
    """
    ordered_series = [
        # Group 1: Output and Income
        'ypr','ips10','ips11','ips299','ips12','ips13','ips18','ips25','ips32','ips34','ips38','ips43','ips307','ips306','pmp','utl11',
        # Group 2: Labor Market
        'lhel','lhelx','lhem','lhnag','lhur','lhu680','lhu5','lhu14','lhu15','lhu26','lhu27','claimuii','ces002','ces003','ces006','ces011','ces015','ces017','ces033','ces046','ces048','ces049','ces053','ces088','ces140','a0m048','ces151','ces155','aom001','pmemp','ces275','ces277','ces278',
        # Group 3: Housing
        'hsfr','hsne','hsmw','hssou','hswst','hsbr','hsbne','hsbmw','hsbsou','hsbwst',
        # Group 4: Consumption, Orders and Inventories
        'pmi','pmno','pmdel','pmnv','a1m008','a0m007','a0m027','a1m092','a0m070','a0m077','cons_r','mtq','a0m059','hhsntn',
        # Group 5: Money and Credit
        'fm1','fm2','fmscu','fm2_r','fmfba','fmrra','fmrnba','fclnbw','fclbmc','ccinrv','ccipy',
        # Group 6: Bond and Exchange rates
        'fyff','cp90','fygm3','fygm6','fygt1','fygt5','fygt10','fyaaac','fybaac','scp90F','sfygm3','sfygm6','sfygt1','sfygt5','sfygt10','sfyaaac','sfybaac','exrus','exrsw','exrjan','exruk','exrcan',
        # Group 7: Prices
        'pwfsa','pwfcsa','pwimsa','pwcmsa','psccom','pw102','pmcp','punew','pu83','pu84','pu85','puc','pucd','pus','puxf','puxhs','puxm','gmdc','gmdcd','gmdcn','gmdcs',
        # Group 8: Stock Market
        'fspcom','fspin','fsdxp','fspxe'
    ]
    g = {
        'Output & Income': set(['ypr','ips10','ips11','ips299','ips12','ips13','ips18','ips25','ips32','ips34','ips38','ips43','ips307','ips306','pmp','utl11']),
        'Labor Market': set(['lhel','lhelx','lhem','lhnag','lhur','lhu680','lhu5','lhu14','lhu15','lhu26','lhu27','claimuii','ces002','ces003','ces006','ces011','ces015','ces017','ces033','ces046','ces048','ces049','ces053','ces088','ces140','a0m048','ces151','ces155','aom001','pmemp','ces275','ces277','ces278']),
        'Housing': set(['hsfr','hsne','hsmw','hssou','hswst','hsbr','hsbne','hsbmw','hsbsou','hsbwst']),
        'Consumption, Orders & Inventories': set(['pmi','pmno','pmdel','pmnv','a1m008','a0m007','a0m027','a1m092','a0m070','a0m077','cons_r','mtq','a0m059','hhsntn']),
        'Money & Credit': set(['fm1','fm2','fmscu','fm2_r','fmfba','fmrra','fmrnba','fclnbw','fclbmc','ccinrv','ccipy']),
        'Rates & FX': set(['fyff','cp90','fygm3','fygm6','fygt1','fygt5','fygt10','fyaaac','fybaac','scp90F','sfygm3','sfygm6','sfygt1','sfygt5','sfygt10','sfyaaac','sfybaac','exrus','exrsw','exrjan','exruk','exrcan']),
        'Prices': set(['pwfsa','pwfcsa','pwimsa','pwcmsa','psccom','pw102','pmcp','punew','pu83','pu84','pu85','puc','pucd','pus','puxf','puxhs','puxm','gmdc','gmdcd','gmdcn','gmdcs']),
        'Stock Market': set(['fspcom','fspin','fsdxp','fspxe'])
    }
    def label_group(name: str) -> str:
        for group, members in g.items():
            if name in members:
                return group
        return 'Other'
    series_to_group = {s: label_group(s) for s in ordered_series}
    return ordered_series, series_to_group


def get_fredmd_grouping():
    """
    Return (ordered_series, series_to_group) for the earlier FRED-MD style
    grouping used previously in LN.py (uppercase FRED-MD mnemonics).
    """
    ordered_series = [
        # Output & Income
        'RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS',
        # Labor Market
        'HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008',
        # Consumption & Housing
        'HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW',
        # Orders & Inventories
        'DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx',
        # Money & Credit
        'M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST',
        # Rates & FX
        'FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx',
        # Prices
        'PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA',
        # Stock Market
        'S&P 500','S&P: indust','S&P div yield','S&P PE ratio'
    ]
    def label_group(name: str) -> str:
        if name in ['RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS']:
            return 'Output & Income'
        if name in ['HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008']:
            return 'Labor Market'
        if name in ['HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW']:
            return 'Consumption & Housing'
        if name in ['DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx']:
            return 'Orders & Inventories'
        if name in ['M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST']:
            return 'Money & Credit'
        if name in ['FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx']:
            return 'Rates & FX'
        if name in ['PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA']:
            return 'Prices'
        if name in ['S&P 500','S&P: indust','S&P div yield','S&P PE ratio']:
            return 'Stock Market'
        return 'Other'
    series_to_group = {s: label_group(s) for s in ordered_series}
    return ordered_series, series_to_group