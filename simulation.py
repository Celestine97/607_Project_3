import numpy as np
from tqdm import tqdm
from statistical_methods import compute_pvalues, bonferroni_method, hochberg_method, benjamini_hochberg_method
from performance_metrics import compute_power, compute_fdr
from data_generation import generate_alternative_means
from save_files import save_simulation_results

def run_simulation_with_base_data(config, base_data, show_progress=True, save_results=True):
    """
    Run simulation using pre-generated base data.
    
    This ensures the SAME random noise is used across configurations
    with the same m, implementing variance reduction.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    base_data : np.ndarray
        Pre-generated standard normal data (n_reps, m)
    show_progress : bool
        Whether to show progress bar
    
    Returns:
    --------
    results : dict
        Dictionary with all results arrays
    """
    
    n_reps, m = base_data.shape
    assert m == config['m'], f"Base data has m={m}, config has m={config['m']}"
    assert n_reps == config['n_reps'], f"Base data has {n_reps} reps, config has {config['n_reps']}"
    
    # Generate the means (same for all replications)
    m0 = config['m0']
    m1 = config['m1']
    means = np.zeros(m)
    
    if m1 > 0:
        alt_means = generate_alternative_means(m1, config['L'], config['distribution'])
        means[m0:] = alt_means
    
    # True nulls indicator (same for all replications)
    true_nulls = np.arange(m) < m0
    
    # Storage for results
    power_bonf = np.zeros(n_reps)
    power_hoch = np.zeros(n_reps)
    power_bh = np.zeros(n_reps)
    fdr_bonf = np.zeros(n_reps)
    fdr_hoch = np.zeros(n_reps)
    fdr_bh = np.zeros(n_reps)
    
    # Run replications
    iterator = range(n_reps)
    if show_progress:
        desc = f"m={config['m']}, {config['distribution']}, {config['m1']}/{config['m']} alt"
        iterator = tqdm(iterator, desc=desc)
    
    for rep in iterator:
        # Use the SAME base random data, just add the means!
        # data = means + base_noise
        data = means + base_data[rep, :]
        
        # Compute p-values
        pvalues = compute_pvalues(data)
        
        # Apply all three methods to same data
        rej_bonf = bonferroni_method(pvalues, config['alpha'])
        rej_hoch = hochberg_method(pvalues, config['alpha'])
        rej_bh = benjamini_hochberg_method(pvalues, config['alpha'])
        
        # Compute performance metrics
        power_bonf[rep] = compute_power(rej_bonf, true_nulls)
        power_hoch[rep] = compute_power(rej_hoch, true_nulls)
        power_bh[rep] = compute_power(rej_bh, true_nulls)
        
        fdr_bonf[rep] = compute_fdr(rej_bonf, true_nulls)
        fdr_hoch[rep] = compute_fdr(rej_hoch, true_nulls)
        fdr_bh[rep] = compute_fdr(rej_bh, true_nulls)
    
    # Return results
    results = {
        'config': config,
        'power_bonf': power_bonf,
        'power_hoch': power_hoch,
        'power_bh': power_bh,
        'fdr_bonf': fdr_bonf,
        'fdr_hoch': fdr_hoch,
        'fdr_bh': fdr_bh
    }

    # Save if requested
    if save_results:
        save_simulation_results(results, config)
    
    return results