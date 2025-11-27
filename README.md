# Benjamini-Hochberg Simulation Study - High Performance

## Project Structure
```
project3/
├── README.md
├── BASELINE.md                  # Profiling for the original code
├── OPTIMIZATION.md              # Optimization documentation
├── requirements.txt
├── Makefile
│
├── data_generation_optimized.py # Optimized data generation
├── statistical_methods_optimized.py  # Optimized statistical methods
├── simulation_optimized.py      # Optimized Simulation
├── simulation_parallel.py       # Paralleled simulation
│
├── baseline_profile.py          # Profiling (baseline vs optimized)
├── visualize_profile.py         # Profile visualization
├── optimizations_comparison.py  # Benchmark comparisons
├── empirical_complexity.py      # Empirical complexity analysis
|
├── numerical_stability.py       # Stability issues
|
├── test_regression.py           # Test if the results are identical
|
├── figures/                     # Visualizations
│   ├── profile_visualization.png      # baseline profile visualization
│   ├── profile_visualization_opt.png  # optimized profile visualization
│   ├── complexity_analysis.png.       # empirical complexity analysis result
│   └── optimization_comparison.png.   # benchmark comparison
└── profiles/                    # Profiling data
    ├── profile_stats.prof
    └── profile_stats_opt.prof

project2(original)/
├── ADEMP.md
│
├── config.py                    # Configuration management
├── data_generation.py           # Data generating functions

├── simulation.py                # Simulation functions
├── performance_metrics.py       # Power and FDR computation
├── visualization.py             # Plotting functions
├── save_files.py                # Result saving and loading utilities
├── run_simulations.py           # Simulation script
├── run_analysis.py              # Analysis script
├── run_visualization.py         # Visualization script
│
├── test.py                      # Test functions
│
├── generated_data/              # Saved simulation raw results
│   └── sim_*.pkl
├── simulation_summary/          # Summary statistics
│   ├── power_comparison_*.csv
│   ├── fdr_control_*.csv
│   └── complete_summary_*.csv
├── figures/                     # Visualizations
│   ├── figure1_reproduction.png
│   ├── power_heatmap.png
│   └── fdr_diagnostic.png
```


## New Makefile Commands

| Command | Description |
|---------|-------------|
| `make profile` | Run profiling analysis (baseline + optimized) |
| `make complexity` | Run empirical complexity analysis |
| `make benchmark` | Run optimization benchmarks and create visualizations |
| `make parallel` | Run parallel simulation demonstration |
| `make stability_check` | Discuss stability issues |
| `make test_regression` | Results validation  |

## Testing

**test_regression.py** - Correctness validation
- Component tests (p-values, Hochberg, BH, alternative means)
- Full simulation pipeline tests
- Edge case tests (all nulls, no nulls)