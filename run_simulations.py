import numpy as np
from config import create_config
from data_generation import generate_base_data
from simulation import run_simulation_with_base_data

import time

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