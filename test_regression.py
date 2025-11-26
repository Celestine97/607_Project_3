"""
Regression Testing: Verify Optimized Code Produces Equivalent Results to Baseline

1. Testing components individually
2. Testing full simulation pipeline
3. Testing edge cases

"""

import numpy as np
from scipy import stats

# Import baseline implementations
from simulation import run_simulation_with_base_data as baseline_simulation
from statistical_methods import (
    compute_pvalues as baseline_compute_pvalues,
    hochberg_method as baseline_hochberg,
    benjamini_hochberg_method as baseline_bh
)
from data_generation import generate_alternative_means as baseline_alt_means

# Import optimized implementations
from simulation_optimized import run_simulation_with_base_data as optimized_simulation
from statistical_methods_optimized import (
    compute_pvalues as optimized_compute_pvalues,
    hochberg_method as optimized_hochberg,
    benjamini_hochberg_method as optimized_bh
)
from data_generation_optimized import generate_alternative_means as optimized_alt_means

from config import create_config
from data_generation import generate_base_data


class TestComponents:

    def test_p_values(self):
        """Test p-value computation with random data"""
        data = np.random.randn(100)
        baseline_pvals = baseline_compute_pvalues(data)
        optimized_pvals = optimized_compute_pvalues(data)
        assert np.allclose(baseline_pvals, optimized_pvals, atol=1e-10)
    
    def test_hochberg(self):
        """Test Hochberg with random data"""
        pvalues = np.random.rand(100)
        alpha = 0.05
        baseline_rej = baseline_hochberg(pvalues, alpha)
        optimized_rej = optimized_hochberg(pvalues, alpha)
        assert np.array_equal(baseline_rej, optimized_rej)

    def test_bh(self):
        """Test Benjamini-Hochberg with random data"""
        pvalues = np.random.rand(100)
        alpha = 0.05
        baseline_rej = baseline_bh(pvalues, alpha)
        optimized_rej = optimized_bh(pvalues, alpha)
        assert np.array_equal(baseline_rej, optimized_rej)


class TestAlternativeMeansGeneration:
    """Test that alternative means generation is identical"""

    def test_distribution_D(self):
        """Test alternative means for distribution D"""
        np.random.seed(42)
        baseline_means = baseline_alt_means(m1=10, L=5, distribution='D')

        np.random.seed(42)
        optimized_means = optimized_alt_means(m1=10, L=5, distribution='D')

        assert np.allclose(baseline_means, optimized_means, atol=1e-10)

    def test_distribution_E(self):
        """Test alternative means for distribution E"""
        np.random.seed(42)
        baseline_means = baseline_alt_means(m1=20, L=3, distribution='E')

        np.random.seed(42)
        optimized_means = optimized_alt_means(m1=20, L=3, distribution='E')

        assert np.allclose(baseline_means, optimized_means, atol=1e-10)

    def test_distribution_I(self):
        """Test alternative means for distribution I"""
        np.random.seed(42)
        baseline_means = baseline_alt_means(m1=15, L=7, distribution='I')

        np.random.seed(42)
        optimized_means = optimized_alt_means(m1=15, L=7, distribution='I')

        assert np.allclose(baseline_means, optimized_means, atol=1e-10)


class TestFullSimulation:
    """Test full simulation pipeline"""

    def test_power_and_fdr_all_methods(self):
        """Verify all power and FDR metrics match exactly"""
        config = create_config(m=8, m0=4, distribution='E', n_reps=100, seed=12345)
        base_data = generate_base_data(config)

        baseline_results = baseline_simulation(config, base_data, show_progress=False, save_results=False)
        optimized_results = optimized_simulation(config, base_data, show_progress=False, save_results=False)

        # Test all metrics
        metrics = ['power_bonf', 'power_hoch', 'power_bh', 'fdr_bonf', 'fdr_hoch', 'fdr_bh']

        for metric in metrics:
            baseline_vals = baseline_results[metric]
            optimized_vals = optimized_results[metric]

            # Handle NaN values properly
            nan_mask_baseline = np.isnan(baseline_vals)
            nan_mask_optimized = np.isnan(optimized_vals)
            assert np.array_equal(nan_mask_baseline, nan_mask_optimized), f"{metric}: NaN positions differ"

            # Compare non-NaN values
            if not np.all(nan_mask_baseline):
                assert np.allclose(baseline_vals[~nan_mask_baseline],
                                 optimized_vals[~nan_mask_optimized],
                                 atol=1e-10), f"{metric}: values differ"


class TestEdgeCases:
    """Test edge cases"""

    def test_all_nulls_m0_equals_m(self):
        """Test edge case where all hypotheses are null (m0 = m)"""
        config = create_config(m=10, m0=10, distribution='E', n_reps=50, seed=111)
        base_data = generate_base_data(config)

        baseline_results = baseline_simulation(config, base_data, show_progress=False, save_results=False)
        optimized_results = optimized_simulation(config, base_data, show_progress=False, save_results=False)

        # Test power and FDR for Bonferroni
        for metric in ['power_bonf', 'fdr_bonf']:
            baseline_vals = baseline_results[metric]
            optimized_vals = optimized_results[metric]

            # Both should have same NaN pattern (power is NaN when all nulls)
            nan_mask_baseline = np.isnan(baseline_vals)
            nan_mask_optimized = np.isnan(optimized_vals)
            assert np.array_equal(nan_mask_baseline, nan_mask_optimized), \
                f"{metric}: NaN patterns differ"

            # Compare non-NaN values
            if not np.all(nan_mask_baseline):
                assert np.allclose(baseline_vals[~nan_mask_baseline],
                                 optimized_vals[~nan_mask_optimized],
                                 atol=1e-10), f"{metric}: values differ"

    def test_no_nulls_m0_equals_0(self):
        """Test edge case where all hypotheses are alternative (m0 = 0)"""
        config = create_config(m=10, m0=0, distribution='I', n_reps=50, seed=222)
        base_data = generate_base_data(config)

        baseline_results = baseline_simulation(config, base_data, show_progress=False, save_results=False)
        optimized_results = optimized_simulation(config, base_data, show_progress=False, save_results=False)

        # Test power and FDR for BH
        for metric in ['power_bh', 'fdr_bh']:
            baseline_vals = baseline_results[metric]
            optimized_vals = optimized_results[metric]
            assert np.allclose(baseline_vals, optimized_vals, atol=1e-10), f"{metric}: values differ"
