"""
compare original vs optimized implementations.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# original implementations
from statistical_methods import (
    compute_pvalues as compute_pvalues_orig,
    hochberg_method as hochberg_orig,
    benjamini_hochberg_method as bh_orig
)
from data_generation import generate_alternative_means as gen_means_orig

# optimized implementations
from statistical_methods_optimized import (
    compute_pvalues as compute_pvalues_opt,
    hochberg_method as hochberg_opt,
    benjamini_hochberg_method as bh_opt
)
from data_generation_optimized import generate_alternative_means as gen_means_opt


def timing_function(func, *args, n_iterations=1000):
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = func(*args)
    elapsed = time.perf_counter() - start
    return elapsed / n_iterations, result

def compare_pvalue_computation(m_values):
    """Compare original vs optimized p-value computation"""

    total_times_orig = []
    total_times_opt = []
    speedups = []
    for m in m_values:
        # Generate test data
        test_stats = np.random.randn(m)

        # Benchmark original
        time_orig, result_orig = timing_function(compute_pvalues_orig, test_stats, n_iterations=10000)

        # Benchmark optimized
        time_opt, result_opt = timing_function(compute_pvalues_opt, test_stats, n_iterations=10000)

        # Verify correctness (should be very close, allowing for numerical precision)
        assert np.allclose(result_orig, result_opt, rtol=1e-10), f"Results differ for m={m}"

        speedup = time_orig / time_opt

        total_times_orig.append(time_orig)
        total_times_opt.append(time_opt)
        speedups.append(speedup)

    return total_times_orig, total_times_opt, speedups


def compare_generate_means(m1_values):
    """Compare original vs optimized generate_alternative_means"""

    total_times_orig = []
    total_times_opt = []
    speedups = []

    for m1 in m1_values:
        L = 5.0

        # Benchmark original
        time_orig, result_orig = timing_function(gen_means_orig, m1, L, 'E', n_iterations=10000)

        # Benchmark optimized
        time_opt, result_opt = timing_function(gen_means_opt, m1, L, 'E', n_iterations=10000)

        # Verify correctness
        assert np.array_equal(result_orig, result_opt)

        speedup = time_orig / time_opt

        total_times_orig.append(time_orig)
        total_times_opt.append(time_opt)
        speedups.append(speedup)

    return total_times_orig, total_times_opt, speedups


def compare_hochberg_method(m_values):
    """Compare original vs optimized Hochberg method"""

    total_times_orig = []
    total_times_opt = []
    speedups = []

    for m in m_values:
        # Generate test data
        pvals = np.random.rand(m)

        # original
        time_orig, result_orig = timing_function(hochberg_orig, pvals, 0.05, n_iterations=1000)

        # Benchmark optimized
        time_opt, result_opt = timing_function(hochberg_opt, pvals, 0.05, n_iterations=1000)

        # Verify correctness
        assert np.array_equal(result_orig, result_opt)

        speedup = time_orig / time_opt

        total_times_orig.append(time_orig)
        total_times_opt.append(time_opt)
        speedups.append(speedup)

    return total_times_orig, total_times_opt, speedups


def compare_bh_method(m_values):
    """Compare original vs optimized Benjamini-Hochberg method"""

    total_times_orig = []
    total_times_opt = []
    speedups = []

    for m in m_values:
        # Generate test data
        pvals = np.random.rand(m)

        # original
        time_orig, result_orig = timing_function(bh_orig, pvals, 0.05, n_iterations=1000)

        # optimized
        time_opt, result_opt = timing_function(bh_opt, pvals, 0.05, n_iterations=1000)

        # Verify correctness
        assert np.array_equal(result_orig, result_opt)

        speedup = time_orig / time_opt

        total_times_orig.append(time_orig)
        total_times_opt.append(time_opt)
        speedups.append(speedup)

    return total_times_orig, total_times_opt, speedups


def create_optimization_plot(times_pval_orig, times_pval_opt,
                             times_means_orig, times_means_opt,
                             times_bh_orig, times_bh_opt,
                             times_hoch_orig, times_hoch_opt,
                             m_values):
    """Create visualization of optimized results"""

    # 4 subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # P-value computation
    ax1.plot(m_values, times_pval_orig, 'o-', label='Original (stats.norm.cdf)',
             linewidth=2, markersize=8, color='purple')
    ax1.plot(m_values, times_pval_opt, 's-', label='Optimized (ndtr)',
             linewidth=2, markersize=8, color='orange')
    ax1.set_xlabel('Number of Hypotheses (m)', fontweight='bold')
    ax1.set_ylabel('Runtime (per replication)', fontweight='bold')
    ax1.set_title('P-value Computation)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Generate Alternative Means
    ax2.plot(m_values, times_means_orig, 'o-', label='Original (Python list)',
             linewidth=2, markersize=8, color='purple')
    ax2.plot(m_values, times_means_opt, 's-', label='Optimized (np.repeat)',
             linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Alternative Hypotheses (m1)', fontweight='bold')
    ax2.set_ylabel('Runtime (per replication)', fontweight='bold')
    ax2.set_title('Generate Alternative Means', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Hochberg Method
    ax3.plot(m_values, times_hoch_orig, 'o-', label='Original', linewidth=2, markersize=8)
    ax3.plot(m_values, times_hoch_opt, 's-', label='Vectorized', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Hypotheses (m)', fontweight='bold')
    ax3.set_ylabel('Runtime (per replication)', fontweight='bold')
    ax3.set_title('Hochberg Method', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # BH Method
    ax4.plot(m_values, times_bh_orig, 'o-', label='Original', linewidth=2, markersize=8)
    ax4.plot(m_values, times_bh_opt, 's-', label='Vectorized', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Hypotheses (m)', fontweight='bold')
    ax4.set_ylabel('Runtime (per replication)', fontweight='bold')
    ax4.set_title('Benjamini-Hochberg Method', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    m_values = [64, 128, 256, 512, 1024, 2048, 4096]
    m1_values = m_values  # for generate_alternative_means

    times_pval_orig, times_pval_opt, _ = compare_pvalue_computation(m_values)
    times_means_orig, times_means_opt, _ = compare_generate_means(m1_values)
    times_hoch_orig, times_hoch_opt, _ = compare_hochberg_method(m_values)
    times_bh_orig, times_bh_opt, _ = compare_bh_method(m_values)

    create_optimization_plot(times_pval_orig, times_pval_opt,
                             times_means_orig, times_means_opt,
                             times_bh_orig, times_bh_opt,
                             times_hoch_orig, times_hoch_opt,
                             m_values)
