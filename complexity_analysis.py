"""
COMPREHENSIVE GUIDE: How to Analyze Computational Complexity
=============================================================

This guide covers both THEORETICAL and EMPIRICAL approaches to analyzing
the computational complexity of your code.
"""

# =============================================================================
# PART 1: THEORETICAL COMPLEXITY ANALYSIS
# =============================================================================

"""
STEP 1: Identify the key parameters
------------------------------------
What variables control how much work your code does?

For the BH simulation:
  - m: number of hypotheses
  - n_reps: number of simulation replications
  - m0: number of null hypotheses (affects logic, but not complexity)

General questions to ask:
  - Does my code process n items? → Parameter: n
  - Does my code run k iterations? → Parameter: k
  - Does my code have nested loops? → Multiple parameters
"""

def theoretical_analysis_example():
    """
    Example: Analyze the theoretical complexity of BH simulation
    """
    
    print("="*80)
    print("THEORETICAL COMPLEXITY ANALYSIS - STEP BY STEP")
    print("="*80)
    
    # Component 1: Data generation
    print("\n1. ANALYZE: generate_base_data()")
    print("-" * 40)
    print("Code: rng.standard_normal(size=(n_reps, m))")
    print("Operations:")
    print("  - Generate n_reps * m random numbers")
    print("  - One operation per number")
    print("Complexity: O(n_reps × m)")
    print("Reasoning: Linear in the total number of elements")
    
    # Component 2: P-value computation
    print("\n2. ANALYZE: compute_pvalues()")
    print("-" * 40)
    print("Code: 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))")
    print("Operations:")
    print("  - np.abs(): O(m) - one operation per element")
    print("  - stats.norm.cdf(): O(m) - vectorized, one CDF per element")
    print("  - Subtraction: O(m)")
    print("  - Multiplication: O(m)")
    print("Complexity: O(m)")
    print("Reasoning: All operations are vectorized and linear in m")
    
    # Component 3: Hochberg method
    print("\n3. ANALYZE: hochberg_method()")
    print("-" * 40)
    print("Code:")
    print("  sorted_indices = np.argsort(pvalues)  # Line 1")
    print("  for i in range(m, 0, -1):             # Line 2")
    print("      if sorted_pvalues[i-1] <= ...:    # Line 3")
    print("          break")
    print("\nOperations:")
    print("  Line 1: Sorting - O(m log m)")
    print("  Line 2: Loop up to m times - O(m)")
    print("  Line 3: Comparison - O(1) per iteration")
    print("Complexity: O(m log m) + O(m) = O(m log m)")
    print("Reasoning: Sorting dominates")
    
    # Component 4: Full simulation
    print("\n4. ANALYZE: run_simulation_with_base_data()")
    print("-" * 40)
    print("Code:")
    print("  for rep in range(n_reps):")
    print("      data = means + base_data[rep, :]      # O(m)")
    print("      pvalues = compute_pvalues(data)       # O(m)")
    print("      rej_bonf = bonferroni_method(...)     # O(m)")
    print("      rej_hoch = hochberg_method(...)       # O(m log m)")
    print("      rej_bh = benjamini_hochberg_method(...) # O(m log m)")
    print("      # ... compute metrics O(m)")
    print("\nPer iteration:")
    print("  - Most expensive: O(m log m) from Hochberg/BH")
    print("  - Total per iteration: O(m log m)")
    print("\nFull complexity:")
    print("  n_reps iterations × O(m log m) per iteration")
    print("  = O(n_reps × m log m)")
    print("\nREASONING:")
    print("  - Outer loop: n_reps times")
    print("  - Inner work: O(m log m) per iteration")
    print("  - Loops are independent → multiply complexities")


# =============================================================================
# PART 2: EMPIRICAL COMPLEXITY ANALYSIS
# =============================================================================

"""
STEP 2: Design timing experiments
----------------------------------
Measure actual runtime as you vary each parameter
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

def empirical_analysis_step_by_step():
    """
    Step-by-step guide to empirical complexity analysis
    """
    
    print("\n" + "="*80)
    print("EMPIRICAL COMPLEXITY ANALYSIS - STEP BY STEP")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # STEP 1: Choose parameter values to test
    # -------------------------------------------------------------------------
    print("\nSTEP 1: Choose parameter values")
    print("-" * 40)
    print("Guidelines:")
    print("  - Use powers of 2 for log-log plots: [2, 4, 8, 16, 32, ...]")
    print("  - Cover 2-3 orders of magnitude: [10, 100, 1000]")
    print("  - Include practical values you actually use")
    print("  - Start small to avoid long waits")
    
    # Example: varying m
    m_values = [4, 8, 16, 32, 64, 128, 256]
    print(f"\nExample: m_values = {m_values}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Write timing code
    # -------------------------------------------------------------------------
    print("\n\nSTEP 2: Write timing code")
    print("-" * 40)
    print("Template:")
    print("""
    import time
    
    times = []
    for m in m_values:
        # Setup
        setup_data = create_test_case(m)
        
        # Time the operation
        start = time.perf_counter()
        result = my_function(setup_data)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        print(f"m={m}: {elapsed:.4f} seconds")
    """)
    
    # -------------------------------------------------------------------------
    # STEP 3: Plot on log-log scale
    # -------------------------------------------------------------------------
    print("\n\nSTEP 3: Plot on log-log scale")
    print("-" * 40)
    print("Why log-log?")
    print("  If time = C × m^b, then:")
    print("    log(time) = log(C) + b × log(m)")
    print("  This is a straight line with slope = b")
    print("  → Easy to see the exponent!")
    
    # Simulate some data for demonstration
    m_vals = np.array([4, 8, 16, 32, 64, 128])
    
    # Case 1: O(m) - linear
    times_linear = 0.001 * m_vals
    
    # Case 2: O(m log m)
    times_mlogm = 0.0001 * m_vals * np.log2(m_vals)
    
    # Case 3: O(m²)
    times_quadratic = 0.00001 * m_vals ** 2
    
    print("\nExample slopes on log-log plot:")
    print("  O(1):       slope ≈ 0   (flat line)")
    print("  O(√m):      slope ≈ 0.5")
    print("  O(m):       slope ≈ 1.0")
    print("  O(m log m): slope ≈ 1.1-1.3")
    print("  O(m²):      slope ≈ 2.0")
    
    # -------------------------------------------------------------------------
    # STEP 4: Fit a model
    # -------------------------------------------------------------------------
    print("\n\nSTEP 4: Fit a power law model")
    print("-" * 40)
    print("Fit: time = a × m^b")
    print("\nMethod 1: Linear regression on log-log scale")
    print("  log_m = np.log(m_values)")
    print("  log_t = np.log(times)")
    print("  slope, intercept = np.polyfit(log_m, log_t, 1)")
    print("  b = slope")
    print("  a = np.exp(intercept)")
    
    # Demonstrate
    log_m = np.log(m_vals)
    log_t_linear = np.log(times_linear)
    slope, intercept = np.polyfit(log_m, log_t_linear, 1)
    
    print(f"\nExample (O(m) data):")
    print(f"  Fitted exponent (b): {slope:.3f}")
    print(f"  Expected: 1.0 for O(m)")
    print(f"  Coefficient (a): {np.exp(intercept):.6f}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Calculate goodness of fit
    # -------------------------------------------------------------------------
    print("\n\nSTEP 5: Calculate goodness of fit (R²)")
    print("-" * 40)
    print("R² = 1 - (SS_residual / SS_total)")
    print("where:")
    print("  SS_residual = sum of squared residuals")
    print("  SS_total = total variance")
    print("\nInterpretation:")
    print("  R² = 1.00: Perfect fit")
    print("  R² > 0.95: Excellent fit")
    print("  R² > 0.90: Good fit")
    print("  R² < 0.90: Poor fit, try different model")
    
    predicted = np.exp(intercept) * m_vals ** slope
    ss_res = np.sum((times_linear - predicted) ** 2)
    ss_tot = np.sum((times_linear - np.mean(times_linear)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\nExample R²: {r2:.6f}")


# =============================================================================
# PART 3: COMPLETE WORKING EXAMPLE
# =============================================================================

def complete_complexity_analysis_example():
    """
    Complete example: Analyze complexity of a simple function
    """
    
    print("\n" + "="*80)
    print("COMPLETE EXAMPLE: Analyzing bubble sort")
    print("="*80)
    
    # The function to analyze
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    # Theoretical analysis
    print("\n1. THEORETICAL ANALYSIS")
    print("-" * 40)
    print("Code structure:")
    print("  for i in range(n):          # Outer loop: n iterations")
    print("      for j in range(n-i-1):  # Inner loop: n-i-1 iterations")
    print("          if arr[j] > arr[j+1]: # O(1) comparison")
    print("              swap              # O(1) swap")
    print("\nComplexity calculation:")
    print("  Total comparisons = n + (n-1) + (n-2) + ... + 1")
    print("                    = n(n+1)/2")
    print("                    = O(n²)")
    print("\nTheoretical complexity: O(n²)")
    
    # Empirical analysis
    print("\n2. EMPIRICAL ANALYSIS")
    print("-" * 40)
    
    n_values = [10, 20, 50, 100, 200, 500]
    times = []
    
    print(f"{'n':>6} {'Time(s)':>12} {'Time/n²':>15}")
    print("-" * 35)
    
    for n in n_values:
        # Create test data
        arr = np.random.randint(0, 1000, n).tolist()
        
        # Time the sort
        start = time.perf_counter()
        _ = bubble_sort(arr.copy())
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        time_per_n2 = elapsed / (n ** 2) if n > 0 else 0
        
        print(f"{n:6d} {elapsed:12.6f} {time_per_n2*1e6:15.6f}")
    
    # Fit power law
    times = np.array(times)
    n_values = np.array(n_values)
    
    log_n = np.log(n_values)
    log_t = np.log(times)
    
    slope, intercept = np.polyfit(log_n, log_t, 1)
    
    print(f"\nEmpirical fit: time ∝ n^{slope:.3f}")
    print(f"Expected: n^2.0 for O(n²)")
    print(f"Match: {'✓ Yes' if 1.8 < slope < 2.2 else '✗ No'}")
    
    # Calculate R²
    predicted = np.exp(intercept) * n_values ** slope
    ss_res = np.sum((times - predicted) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"R²: {r2:.6f}")


# =============================================================================
# PART 4: COMMON COMPLEXITY PATTERNS
# =============================================================================

def common_complexity_patterns():
    """
    Reference guide to common complexity patterns
    """
    
    print("\n" + "="*80)
    print("REFERENCE: Common Complexity Patterns")
    print("="*80)
    
    patterns = {
        'O(1) - Constant': {
            'description': 'Time does not depend on input size',
            'examples': [
                'Array access: arr[i]',
                'Dictionary lookup: dict[key]',
                'Arithmetic operations: a + b'
            ],
            'log_slope': 0.0
        },
        'O(log n) - Logarithmic': {
            'description': 'Time grows slowly as n increases',
            'examples': [
                'Binary search',
                'Balanced tree operations',
                'Finding GCD'
            ],
            'log_slope': 'Not straight line (curves down)'
        },
        'O(n) - Linear': {
            'description': 'Time grows proportionally with n',
            'examples': [
                'Iterating through array once',
                'np.sum(arr)',
                'Finding minimum/maximum'
            ],
            'log_slope': 1.0
        },
        'O(n log n) - Linearithmic': {
            'description': 'Slightly worse than linear',
            'examples': [
                'Efficient sorting (merge sort, quicksort)',
                'np.sort()',
                'Building balanced tree'
            ],
            'log_slope': '1.1-1.3 (slightly > 1)'
        },
        'O(n²) - Quadratic': {
            'description': 'Time grows with square of n',
            'examples': [
                'Nested loops over same array',
                'Bubble sort',
                'All pairs comparisons'
            ],
            'log_slope': 2.0
        },
        'O(2^n) - Exponential': {
            'description': 'Time doubles with each increase in n',
            'examples': [
                'Recursive Fibonacci (naive)',
                'Trying all subsets',
                'Brute force combinatorics'
            ],
            'log_slope': 'Not straight line (curves up sharply)'
        }
    }
    
    for complexity, info in patterns.items():
        print(f"\n{complexity}")
        print("-" * 60)
        print(f"Description: {info['description']}")
        print(f"Examples:")
        for ex in info['examples']:
            print(f"  • {ex}")
        print(f"Log-log slope: {info['log_slope']}")


# =============================================================================
# PART 5: PRACTICAL TIPS
# =============================================================================

def practical_tips():
    """
    Practical tips for complexity analysis
    """
    
    print("\n" + "="*80)
    print("PRACTICAL TIPS FOR COMPLEXITY ANALYSIS")
    print("="*80)
    
    tips = [
        ("Use time.perf_counter()", 
         "More accurate than time.time() for short durations"),
        
        ("Warm up before timing",
         "Run function once before timing to avoid cold start effects:\n"
         "  for _ in range(10): my_function()  # warm up\n"
         "  start = time.perf_counter()"),
        
        ("Average multiple runs",
         "For very fast functions, run many times and average:\n"
         "  n_iterations = 1000\n"
         "  start = time.perf_counter()\n"
         "  for _ in range(n_iterations): my_function()\n"
         "  avg_time = (time.perf_counter() - start) / n_iterations"),
        
        ("Control for noise",
         "Close other programs, disable turbo boost if needed,\n"
         "  run multiple times and check consistency"),
        
        ("Test both best and worst case",
         "Some algorithms have different complexity for different inputs:\n"
         "  • Sorted vs random data\n"
         "  • Sparse vs dense matrices\n"
         "  • Small vs large values"),
        
        ("Watch for constant factors",
         "O(n) with large constant can be slower than O(n log n):\n"
         "  • time = 1000 × n might be slower than time = 0.01 × n log n\n"
         "  • This is why empirical testing is important!"),
        
        ("Use log-log plots",
         "They make power laws appear as straight lines:\n"
         "  plt.loglog(n_values, times, 'o-')\n"
         "  → Straight line with slope b means O(n^b)"),
        
        ("Check R² value",
         "R² > 0.95 means your model fits well\n"
         "  R² < 0.90 means try a different model or more data points"),
        
        ("Compare theoretical vs empirical",
         "Differences reveal:\n"
         "  • Constant factors\n"
         "  • Optimization effects\n"
         "  • Caching\n"
         "  • Vectorization benefits"),
        
        ("Profile first, optimize later",
         "Don't guess what's slow - measure it!\n"
         "  • Use cProfile to find bottlenecks\n"
         "  • Then do complexity analysis on slow parts")
    ]
    
    for i, (tip, explanation) in enumerate(tips, 1):
        print(f"\n{i}. {tip}")
        print("   " + explanation.replace('\n', '\n   '))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        COMPREHENSIVE GUIDE TO COMPUTATIONAL COMPLEXITY ANALYSIS              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Part 1: Theoretical
    theoretical_analysis_example()
    
    # Part 2: Empirical
    empirical_analysis_step_by_step()
    
    # Part 3: Complete example
    complete_complexity_analysis_example()
    
    # Part 4: Common patterns
    common_complexity_patterns()
    
    # Part 5: Practical tips
    practical_tips()
    
    print("\n" + "="*80)
    print("SUMMARY: Two-Step Process")
    print("="*80)
    print("""
STEP 1: THEORETICAL ANALYSIS
  → Read code, count operations
  → Identify nested loops, recursion
  → Express as Big-O notation
  → Predict how time scales with input size

STEP 2: EMPIRICAL ANALYSIS  
  → Write timing code
  → Test with various input sizes
  → Plot results (use log-log scale)
  → Fit model and calculate R²
  → Compare with theoretical prediction

WHY DO BOTH?
  • Theory: Tells you asymptotic behavior
  • Empirical: Reveals constant factors and real performance
  • Together: Complete understanding!
""")
    
    print("="*80)
    print("For your BH simulation:")
    print("  Run: python complexity_analysis.py")
    print("="*80) 