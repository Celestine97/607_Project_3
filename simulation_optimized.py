import numpy as np
from tqdm import tqdm
from statistical_methods import bonferroni_method
from statistical_methods_optimized import (
    compute_pvalues, hochberg_method, benjamini_hochberg_method
)
from performance_metrics import compute_power, compute_fdr
from data_generation_optimized import generate_alternative_means
from save_files import save_simulation_results


def run_simulation_with_base_data(config, base_data, show_progress=True, save_results=True):
    """
    Run simulation using pre-generated base data. Using optimized methods.
    
    This ensures the same random noise is used across configurations
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

    # Storage for results - pre-allocated (already optimal)
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
        data = means + base_data[rep, :]

        # Compute p-values 
        pvalues = compute_pvalues(data)

        # Apply all three methods to same data
        rej_bonf = bonferroni_method(pvalues, config['alpha'])
        rej_hoch = hochberg_method(pvalues, config['alpha'])  # Now O(m log m) guaranteed
        rej_bh = benjamini_hochberg_method(pvalues, config['alpha'])  # Now O(m log m) guaranteed

        # Compute performance metrics (already optimal - vectorized)
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


from config import create_config
from data_generation import generate_base_data
import time

# run full simulation
if __name__ == "__main__":

    all_results = {}

    # Parameters
    m_values = [4, 8, 16, 32, 64]
    null_proportions = [0.75, 0.50, 0.25, 0.0]
    distributions = ['D', 'E', 'I']
    n_reps = 20000
    base_seed = 123456789
    L_setting = 5
    alpha_setting = 0.05

    data_output_dir = 'generated_data/'
    figure_output_dir = 'figures/'

    # For each m, generate base data once
    base_data_cache = {}
    total_start = time.perf_counter()
    for m in m_values:
        config = create_config(
            m=m, m0=0, distribution='E',
            L=L_setting,
            alpha=alpha_setting,
            n_reps=n_reps,
            seed= base_seed + m  # Different seed per m to ensure different data
        )
        print(f"\n Generating base data for m={m} ({n_reps} replications)...")
        base_data_cache[m] = generate_base_data(config)

    total = len(m_values) * len(null_proportions) * len(distributions)
    sim_count = 0
    
    for m in m_values:
        # Get the pre-generated base data for this m
        base_data = base_data_cache[m]
        
        for null_prop in null_proportions:
            for dist in distributions:
                sim_count += 1
                m0 = int(m * null_prop)
                
                print(f"[{sim_count}/{total}] m={m}, {null_prop*100:.0f}% null, {dist}")
                
                # Create configuration for this setting
                config = create_config(
                    m=m, m0=m0, distribution=dist,
                    L=L_setting,
                    alpha=alpha_setting,
                    n_reps=n_reps,
                    seed=base_seed + m  # Same seed for same m
                )
                
                # Run simulation with base data
                # variance reduction
                results = run_simulation_with_base_data(config, 
                    base_data,
                    show_progress=False,
                    save_results=True
                    )
                
                # Store results
                key = (m, m0, dist)
                all_results[key] = results
    
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "="*80)
    print("Simulations complete." + "\n")
    print(f"Elapsed: {total_elapsed:.10f} seconds\n")
    print("="*80 + "\n")