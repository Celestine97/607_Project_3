import pytest
import numpy as np
import sys
from scipy import stats

from config import create_config
from data_generation import generate_base_data

from simulation import (
    bonferroni_method,
    benjamini_hochberg_method,
    compute_power,
    compute_fdr,
    run_simulation_with_base_data
)

class TestDataGenerationProcess:
    def test_dgp_null_case(self):
        """
        Verify that null data has mean 0 and variance 1.
        """
        
        config = create_config(m=100, m0=100, distribution='E', L=5.0, n_reps=1000, seed=12345)
        rng = np.random.default_rng(config['seed'])
        
        # Generate data multiple times
        means_observed = []
        vars_observed = []
        
        for _ in range(100):
            data = generate_base_data(config)
            means_observed.append(np.mean(data))
            vars_observed.append(np.var(data, ddof=1))
        
        mean_of_means = np.mean(means_observed)
        mean_of_vars = np.mean(vars_observed)
        
        # Check if within reasonable bounds (3 standard errors)
        se_mean = 1.0 / np.sqrt(100 * 100)  # SE of sample mean
        se_var = np.sqrt(2.0 / (100 - 1)) / np.sqrt(100)  # Approx SE of variance
        
        # Test within 3 standard errors
        assert abs(mean_of_means) < 3 * se_mean
        assert abs(mean_of_vars - 1.0) < 3 * se_var


class TestMethodCorrectness:
    def test_bonferroni_simple_case(self):
        """
        Test if Bonferroni method works correctly.
        """
        
        pvalues = np.array([0.001, 0.004, 0.010, 0.020, 0.050])
        m = len(pvalues)
        alpha = 0.05
        threshold = alpha / m
        
        rejections = bonferroni_method(pvalues, alpha=alpha)
        
        # Expected: only p-values ≤ 0.01 should be rejected
        expected = (pvalues <= threshold).sum()
        observed = rejections.sum()
        
        print(f"\nExpected rejections: {expected}")
        print(f"Observed rejections: {observed}")
        
        assert expected == observed
    def test_bh_simple_case(self):
        """
        BH method in simple case.
        """
        print("\n" + "="*70)
        print("TEST 2.1: Method Correctness - BH Simple Case")
        print("="*70)
        
        # Simple known case
        pvalues = np.array([0.001, 0.004, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 0.800, 0.900])
        m = len(pvalues)
        q = 0.05
        
        # Manual calculation
        expected_rejections = 0
        for i in range(m):
            threshold = ((i+1) / m) * q
            passes = pvalues[i] <= threshold
            if passes:
                expected_rejections = i + 1
        
        # Apply BH
        rejections = benjamini_hochberg_method(pvalues, q=q)
        observed_rejections = rejections.sum()
        
        assert observed_rejections == expected_rejections


class TestPerfomanceMetrics:
    def test_power_calculation(self):
        """
        Test if power calculation is correct.
        """
        # Known scenario   
        # Rejections: [F, F, T, T, T, F, F, T, T, T]
        rejections = np.array([False, False, True, True, True, 
                            False, False, True, True, True])
        
        # True nulls: [T, T, T, T, T, F, F, F, F, F]
        true_nulls = np.array([True, True, True, True, True,
                            False, False, False, False, False])
        
        power = compute_power(rejections, true_nulls)
        
        expected_power = 3 / 5
        
        assert abs(power - expected_power) < 0.001
    
    def test_fdr_calculation(self):
        """
        Test if FDR calculation is correct.
        """
        
        # Known scenario
        rejections = np.array([True, True, True, True, False, False, False, False])
        true_nulls = np.array([True, True, False, False, False, False, True, True])
        
        fdr = compute_fdr(rejections, true_nulls)
        
        expected_fdr = 2 / 4  # 2 false discoveries out of 4 rejections
        
        assert abs(fdr - expected_fdr) < 0.001

class TestReproducibility:
    def test_reproducibility(self):
        """
        Test if the same seed gives identical results.
        """
        
        config = create_config(m=16, m0=8, distribution='E', n_reps=100, seed=12345)
        base_data = generate_base_data(config)
        
        # Run 1
        results1 = run_simulation_with_base_data(config, base_data, show_progress=False)
        
        # Run 2 (same config, same seed)
        results2 = run_simulation_with_base_data(config, base_data, show_progress=False)
        
        # Compare
        all_identical = True
        
        for key in ['power_bonf', 'power_hoch', 'power_bh', 'fdr_bonf', 'fdr_hoch', 'fdr_bh']:
            vals1 = results1[key]
            vals2 = results2[key]
            
            # Handle NaN
            identical = np.allclose(vals1, vals2, equal_nan=True)
            
            if not identical:
                all_identical = False
                break
        
        assert all_identical