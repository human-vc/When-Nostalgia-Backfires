import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_ols_regression(df, control_variables=None, standardize=False, robust_se=True):
    required_cols = ['delta_nostalgia', 'delta_turnout']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if control_variables is None:
        X = df[['delta_nostalgia']].copy()
    else:
        if not isinstance(control_variables, list):
            control_variables = list(control_variables)
        missing_controls = [col for col in control_variables if col not in df.columns]
        if missing_controls:
            raise ValueError(f"Missing control variables: {missing_controls}")
        X = df[['delta_nostalgia'] + control_variables].copy()

    if X.isnull().any().any():
        raise ValueError("Feature matrix contains missing values. Please handle NaN values before regression.")

    if standardize:
        std_vals = X.std()
        if (std_vals == 0).any():
            zero_std_cols = std_vals[std_vals == 0].index.tolist()
            raise ValueError(f"Cannot standardize: columns have zero standard deviation: {zero_std_cols}")
        X = (X - X.mean()) / X.std()

    X = sm.add_constant(X)
    y = df['delta_turnout'].copy()

    if y.isnull().any():
        raise ValueError("Target variable 'delta_turnout' contains missing values.")

    if robust_se:
        model = sm.OLS(y, X).fit(cov_type='HC3')
    else:
        model = sm.OLS(y, X).fit()

    return model

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns[1:]
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(1, len(X.columns))]
    return vif_data

def run_full_robustness_analysis(df):
    results = {}
    
    model_simple = run_ols_regression(df, control_variables=None, standardize=False, robust_se=True)
    results['simple_model'] = model_simple
    
    model_controls = run_ols_regression(df, control_variables=['median_income', 'pct_college'], standardize=True, robust_se=True)
    results['controls_model'] = model_controls
    
    X_controls = df[['delta_nostalgia', 'median_income', 'pct_college']].copy()
    X_controls = (X_controls - X_controls.mean()) / X_controls.std()
    X_controls = sm.add_constant(X_controls)
    vif_data = calculate_vif(X_controls)
    results['vif'] = vif_data
    
    return results

def run_subgroup_ols(df_subgroup, control_variables=None):
    model_simple = run_ols_regression(df_subgroup, control_variables=None, standardize=False, robust_se=True)
    
    if control_variables is not None:
        model_controls = run_ols_regression(df_subgroup, control_variables=control_variables, standardize=True, robust_se=True)
        return model_simple, model_controls
    
    return model_simple

def extract_regression_results(model):
    results = {
        'coefficients': model.params,
        'std_errors': model.bse,
        'p_values': model.pvalues,
        't_statistics': model.tvalues,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'n_observations': int(model.nobs),
        'aic': model.aic,
        'bic': model.bic
    }
    return results

def create_regression_table(results_dict):
    table_data = []
    for model_name, model in results_dict.items():
        if model_name != 'vif':
            extracted = extract_regression_results(model)
            table_data.append({
                'Model': model_name,
                'N': extracted['n_observations'],
                'R²': round(extracted['r_squared'], 3),
                'Adj R²': round(extracted['adj_r_squared'], 3),
                'AIC': round(extracted['aic'], 1),
                'BIC': round(extracted['bic'], 1)
            })
    return pd.DataFrame(table_data)