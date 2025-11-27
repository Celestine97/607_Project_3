"""
Computational Complexity Analysis
Analyzes:
    simulation runtime for separate m;
    time for computing p-values;
    time for BH method execution
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.optimize import curve_fit

from config import create_config
from data_generation import generate_base_data
from statistical_methods import (
    compute_pvalues, benjamini_hochberg_method
)
from simulation import run_simulation_with_base_data


def time_single_replication(m, m0):
    """Time a single replication of the simulation"""
    config = create_config(m=m, m0=m0, distribution='E', n_reps=1, seed=12345)
    base_data = generate_base_data(config)
    
    start = time.perf_counter()
    _ = run_simulation_with_base_data(config, base_data, show_progress=False, save_results=False)
    elapsed = time.perf_counter() - start
    
    return elapsed


def time_component(func, *args, n_iterations=1000):
    """Time a single component function"""
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        func(*args)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_iterations


def total_time_with_m():
    """
    Examine the complete simulation runtime under different m(number of hypotheses)
    """
    m_values = [4, 8, 16, 32, 64, 128, 256, 512]
    n_reps_fixed = 1000
    times = []

    for m in m_values:
        config = create_config(m=m, m0=m//2, distribution='E', 
                              n_reps=n_reps_fixed, seed=12345)
        base_data = generate_base_data(config)
        
        start = time.perf_counter()
        _ = run_simulation_with_base_data(config, base_data, 
                                         show_progress=False, save_results=False)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        time_per_rep = elapsed / n_reps_fixed * 1000  # ms
        time_per_m_log_m = elapsed / (m * np.log2(m)) * 1e6  # microseconds
        
    
    times = np.array(times)
    m_values = np.array(m_values)

    # fit model
    def constant_plus_mlogm(m, C, a):
        return C + a * m * np.log(m)
    
    popt_full, _ = curve_fit(constant_plus_mlogm, m_values, times)
    C_full, a_full = popt_full
    
    # Calculate prediction
    pred_full = constant_plus_mlogm(m_values, C_full, a_full)
    
    # Calculate R²
    ss_tot = np.sum((times - np.mean(times))**2)
    
    ss_res_full = np.sum((times - pred_full)**2)
    r2_full = 1 - ss_res_full / ss_tot
    
    return {
        'm_values': m_values,
        'times': times,
        'constant': C_full,
        'mlogm_coeff_with_constant': a_full,
        'r2_full': r2_full
    }

def empirical_complexity_pvalue_computation():
    """
    Empirical complexity analysis of the p-value computation step
    """
    m_values = [16, 32, 64, 128, 256, 512]

    time_pvalue = []

    for m in m_values:
        # Generate test data
        test_data = np.random.randn(m)

        # Time p-value computation
        t_pval = time_component(compute_pvalues, test_data, n_iterations=1000)
        time_pvalue.append(t_pval * 1e6)  # microseconds

    # Convert to numpy arrays
    m_values = np.array(m_values)
    time_pvalue = np.array(time_pvalue)

    # Fit linear model: time = a * m + b
    def linear_model(m, a, b):
        return a * m + b

    pval_linear, _ = curve_fit(linear_model, m_values, time_pvalue)
    a_linear, b_linear = pval_linear 
    
    # Calculate predictions
    pred_linear = linear_model(m_values, a_linear, b_linear)

    # Calculate R²
    ss_tot = np.sum((time_pvalue - np.mean(time_pvalue))**2)
    ss_res_linear = np.sum((time_pvalue - pred_linear)**2)
    r2_linear = 1 - ss_res_linear / ss_tot

    return {
        'm_values': m_values,
        'time_pvalue': time_pvalue,
        'linear_coeff': a_linear,
        'intercept': b_linear,
        'r2': r2_linear,
    }

def empirical_complexity_BH_method():
    """
    empirical complexity analysis of the BH method
    """
    m_values = [16, 32, 64, 128, 256, 512]

    time_BH = []

    for m in m_values:
        # Generate test data
        test_data = np.random.randn(m)
        pvals = compute_pvalues(test_data)

        # Time BH method
        t_bh = time_component(benjamini_hochberg_method, pvals, 0.05, n_iterations=10000)
        time_BH.append(t_bh*1e6) # microseconds

    # Convert to numpy arrays
    m_values = np.array(m_values)
    time_BH = np.array(time_BH)

    # fit model
    def m_log_m_model(m, a):
        return a * m * np.log(m)

    BH_mlogm, _ = curve_fit(m_log_m_model, m_values, time_BH)
    a_mlogm = BH_mlogm[0]
    # Calculate predictions
    pred_mlogm = m_log_m_model(m_values, a_mlogm)

    # Calculate R²
    ss_tot = np.sum((time_BH - np.mean(time_BH))**2)
    ss_res_mlogm = np.sum((time_BH - pred_mlogm)**2)
    r2_mlogm = 1 - ss_res_mlogm / ss_tot

    return {
        'm_values': m_values,
        'time_BH': time_BH,
        'm_log_m_coeff': a_mlogm,
        'r2': r2_mlogm,
    }


def create_complexity_plots(output_dir='figures/'):
    """Create comprehensive visualization with 3 subplots: p-value, total runtime, and BH method"""

    # Get data from all three functions
    print("Running empirical_complexity_pvalue_computation()...")
    data_pval = empirical_complexity_pvalue_computation()
    
    print("\nRunning total_time_with_m()...")
    data_total = total_time_with_m()

    print("\nRunning empirical_complexity_BH_method()...")
    data_bh = empirical_complexity_BH_method()

    # Create figure with 3 subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ========================================================================
    # Plot 1: P-value computation (O(m))
    # ========================================================================
    m_vals_pval = data_pval['m_values']
    times_pval = data_pval['time_pvalue']
    a_pval = data_pval['linear_coeff']
    b_pval = data_pval['intercept']

    ax1.plot(m_vals_pval, times_pval, 'o', markersize=10, color='darkblue',
               label='Measured', zorder=5)

    # Fitted model: a*m + b
    m_fine_pval = np.linspace(m_vals_pval[0], m_vals_pval[-1], 200)
    fit_pval = a_pval * m_fine_pval + b_pval
    ax1.plot(m_fine_pval, fit_pval, '-', color='red', linewidth=3,
               label=f"a·m + b (R²={data_pval['r2']:.4f})")

    ax1.set_xlabel('Number of Hypotheses (m)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Runtime (microseconds)', fontweight='bold', fontsize=11)
    ax1.set_title('P-value Computation Runtime vs m', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Add text annotation
    textstr_pval = f'a = {a_pval:.4f} μs\nb = {b_pval:.2f} μs'
    ax1.text(0.05, 0.95, textstr_pval, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================================================
    # Plot 2: BH method runtime (a*m*log(m))
    # ========================================================================
    m_vals_bh = data_bh['m_values']
    times_bh = data_bh['time_BH']

    ax2.loglog(m_vals_bh, times_bh, 'o', markersize=10, color='darkblue',
               label='Measured', zorder=5)

    # Fitted model: a*m*log(m)
    m_fine_bh = np.linspace(m_vals_bh[0], m_vals_bh[-1], 200)
    fit_bh = data_bh['m_log_m_coeff'] * m_fine_bh * np.log(m_fine_bh)
    ax2.loglog(m_fine_bh, fit_bh, '-', color='red', linewidth=3,
               label=f"a·m·log(m) (R²={data_bh['r2']:.4f})")

    ax2.set_xlabel('Number of Hypotheses (m)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Runtime (microseconds)', fontweight='bold', fontsize=11)
    ax2.set_title('Benjamini-Hochberg Method Runtime vs m', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3, which='both')

    # Add text annotation
    textstr_bh = f'a = {data_bh["m_log_m_coeff"]:.4f} μs'
    ax2.text(0.05, 0.95, textstr_bh, transform=ax2.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ========================================================================
    # Plot 3: Total simulation runtime (C + a*m*log(m))
    # ========================================================================
    m_vals = data_total['m_values']
    times = data_total['times']

    ax3.loglog(m_vals, times, 'o', markersize=10, color='black',
               label='Measured', zorder=5)

    # Fitted model: C + a*m*log(m)
    m_fine = np.linspace(m_vals[0], m_vals[-1], 200)
    full_fit = data_total['constant'] + data_total['mlogm_coeff_with_constant'] * m_fine * np.log(m_fine)
    ax3.loglog(m_fine, full_fit, '-', color='green', linewidth=3,
               label=f"C + a·m·log(m) (R²={data_total['r2_full']:.4f})")

    ax3.set_xlabel('Number of Hypotheses (m)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Runtime (seconds)', fontweight='bold', fontsize=11)
    ax3.set_title('Total Simulation Runtime vs m\n(n_reps=1000)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3, which='both')

    # Add text annotation
    textstr = f'C = {data_total["constant"]:.4f} s\na = {data_total["mlogm_coeff_with_constant"]:.2e} s'
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'complexity_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("\n Plot saved as 'complexity_analysis.png'")
    plt.show()

if __name__ == "__main__":
    create_complexity_plots(output_dir='figures/')