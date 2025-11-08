import numpy as np
from scipy.stats import spearmanr
from scipy import stats

np.random.seed(42)

def calculate_spearman_correlation(x, y):
    rho, p_value = spearmanr(x, y)
    return rho, p_value

def bootstrap_correlation(x, y, n_iterations=5000, seed=42):
    np.random.seed(seed)
    n = len(x)
    correlations = []
    for _ in range(n_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x.iloc[indices]
        y_boot = y.iloc[indices]
        rho_boot, _ = spearmanr(x_boot, y_boot)
        correlations.append(rho_boot)
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)
    return ci_lower, ci_upper, correlations

def permutation_test(x, y, n_iterations=5000, seed=42):
    np.random.seed(seed)
    observed_rho, _ = spearmanr(x, y)
    permuted_rhos = []
    for _ in range(n_iterations):
        y_permuted = y.sample(frac=1, random_state=seed + _).reset_index(drop=True)
        rho_perm, _ = spearmanr(x, y_permuted)
        permuted_rhos.append(rho_perm)
    p_value = np.mean(np.abs(permuted_rhos) >= np.abs(observed_rho))
    return p_value, permuted_rhos

def analyze_overall_correlation(df):
    rho_overall, p_overall = calculate_spearman_correlation(df['delta_nostalgia'], df['delta_turnout'])
    ci_lower, ci_upper, boot_dist = bootstrap_correlation(df['delta_nostalgia'], df['delta_turnout'])
    p_perm, perm_dist = permutation_test(df['delta_nostalgia'], df['delta_turnout'])
    
    results = {
        'rho': rho_overall,
        'p_value': p_overall,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_permutation': p_perm,
        'bootstrap_dist': boot_dist,
        'permutation_dist': perm_dist
    }
    return results

def analyze_state_correlations(df, states):
    state_results = {}
    for state in states:
        df_state = df[df['state'] == state]
        if len(df_state) >= 10:
            rho_state, p_state = calculate_spearman_correlation(df_state['delta_nostalgia'], df_state['delta_turnout'])
            ci_lower_state, ci_upper_state, _ = bootstrap_correlation(df_state['delta_nostalgia'], df_state['delta_turnout'])
            state_results[state] = {
                'n': len(df_state),
                'rho': rho_state,
                'p_value': p_state,
                'ci_lower': ci_lower_state,
                'ci_upper': ci_upper_state
            }
    return state_results

def fisher_r_to_z_test(r1, n1, r2, n2):
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def compare_state_correlations(state_results, reference_state, comparison_states):
    comparisons = {}
    ref_rho = state_results[reference_state]['rho']
    ref_n = state_results[reference_state]['n']
    
    for state in comparison_states:
        if state in state_results:
            z_stat, p_val = fisher_r_to_z_test(
                ref_rho, ref_n,
                state_results[state]['rho'], state_results[state]['n']
            )
            comparisons[f"{reference_state}_vs_{state}"] = {
                'z_statistic': z_stat,
                'p_value': p_val
            }
    return comparisons

def analyze_demographic_subgroups(df_state, demographic_var, threshold):
    df_high = df_state[df_state[demographic_var] > threshold]
    df_low = df_state[df_state[demographic_var] <= threshold]
    
    rho_high, p_high = calculate_spearman_correlation(df_high['delta_nostalgia'], df_high['delta_turnout'])
    ci_lower_high, ci_upper_high, boot_dist_high = bootstrap_correlation(df_high['delta_nostalgia'], df_high['delta_turnout'])
    
    rho_low, p_low = calculate_spearman_correlation(df_low['delta_nostalgia'], df_low['delta_turnout'])
    ci_lower_low, ci_upper_low, boot_dist_low = bootstrap_correlation(df_low['delta_nostalgia'], df_low['delta_turnout'])
    
    z_stat, p_comparison = fisher_r_to_z_test(rho_high, len(df_high), rho_low, len(df_low))
    
    results = {
        'high_group': {
            'n': len(df_high),
            'rho': rho_high,
            'p_value': p_high,
            'ci_lower': ci_lower_high,
            'ci_upper': ci_upper_high,
            'bootstrap_dist': boot_dist_high
        },
        'low_group': {
            'n': len(df_low),
            'rho': rho_low,
            'p_value': p_low,
            'ci_lower': ci_lower_low,
            'ci_upper': ci_upper_low,
            'bootstrap_dist': boot_dist_low
        },
        'comparison': {
            'z_statistic': z_stat,
            'p_value': p_comparison
        }
    }
    return results

