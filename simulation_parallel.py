"""
Parallel simulation - parallelizes across configurations.

each replication is vectorized, we parallelize at the
configuration level (different combinations of m, null_proportion, distribution).
"""

from simulation_optimized import run_simulation_with_base_data
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time


def run_single_configuration(args):
    """
    Run simulation for a single configuration.

    Parameters:
    -----------
    args : tuple
        (m, null_prop, dist, base_data, base_seed, L_setting, alpha_setting, n_reps)

    Returns:
    --------
    tuple : (key, results)
        key is (m, m0, dist), results is the simulation results dict
    """
    m, null_prop, dist, base_data, base_seed, L_setting, alpha_setting, n_reps = args

    m0 = int(m * null_prop)

    # Create configuration
    from config import create_config
    config = create_config(
        m=m, m0=m0, distribution=dist,
        L=L_setting,
        alpha=alpha_setting,
        n_reps=n_reps,
        seed=base_seed + m
    )

    results = run_simulation_with_base_data(
        config,
        base_data,
        show_progress=False,
        save_results=True
    )

    key = (m, m0, dist)
    return (key, results)


if __name__ == "__main__":
    from config import create_config
    from data_generation import generate_base_data

    all_results = {}

    # Parameters (same as simulation_optimized.py)
    m_values = [4, 8, 16, 32, 64]
    null_proportions = [0.75, 0.50, 0.25, 0.0]
    distributions = ['D', 'E', 'I']
    n_reps = 20000
    base_seed = 123456789
    L_setting = 5
    alpha_setting = 0.05

    n_cpus = cpu_count()
    print(f"Available CPUs: {n_cpus}")
    print(f"\n Parallelize across configurations")

    # Generate base data for each m
    base_data_cache = {}
    total_start = time.perf_counter()

    for m in m_values:
        config = create_config(
            m=m, m0=0, distribution='E',
            L=L_setting,
            alpha=alpha_setting,
            n_reps=n_reps,
            seed=base_seed + m
        )
        print(f"  m={m} ({n_reps} replications)...", end=" ")
        base_data_cache[m] = generate_base_data(config)

    # Prepare all configuration arguments for parallel execution
    config_args = []
    for m in m_values:
        base_data = base_data_cache[m]
        for null_prop in null_proportions:
            for dist in distributions:
                config_args.append((
                    m, null_prop, dist, base_data,
                    base_seed, L_setting, alpha_setting, n_reps
                ))

    total = len(config_args)
    print(f"\n Running {total} configurations in parallel...")

    # Run all configurations in parallel across CPUs
    with Pool(processes=n_cpus) as pool:
        results_list = list(tqdm(
            pool.imap_unordered(run_single_configuration, config_args),
            total=total,
            desc="Simulations",
            ncols=80
        ))

    # Store results
    for key, results in results_list:
        all_results[key] = results

    total_elapsed = time.perf_counter() - total_start

    print(f"Total time: {total_elapsed:.10f} seconds")
