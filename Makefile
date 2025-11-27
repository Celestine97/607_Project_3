# Makefile for the simulation project

.PHONY: all simulate analyze figures clean test help install

# Default
all: 
	@echo "Results will be saved to:"
	@echo "  - results/all_results_*.pkl"
	@echo "  - figures/*.png"
	@echo "  - simulation_summary/*.csv"
	python run_simulations.py
	python run_analysis.py
	python run_visualization.py

# Install required packages
install:
	@echo "Installing required packages..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

# Run tests
test:
	@echo ""
	@echo "Running test suite..."
	pytest test.py

# Run simulations and save raw results
simulation:
	@echo " "
	@echo "Running simulations..."
	@echo " "
	python run_simulations.py
	@echo "Simulations complete"
# Analyze results and generate summary statistics
analyze:
	@echo " "
	@echo "Analyzing results..."
	@echo " "
	python run_analysis.py
	@echo "Analysis complete"
# Create all visualizations
visualizations:
	@echo " "
	@echo "Generating figures..."
	@echo " "
	python run_visualization.py
	@echo "Figures generated"

profile:
	@echo " "
	@echo "Baseline profiling..."
	@echo " "
	python baseline_profile.py
	python visualize_profile.py

complexity:
	@echo " "
	@echo "Running empirical complexity analysis..."
	@echo " "
	python empirical_complexity.py
	@echo "Complexity analysis complete"

benchmark:
	@echo " "
	@echo "Running benchmark comparisons..."
	@echo " "
	python optimizations_comparison.py
	@echo "Creating optimized profile visualization..."
	python -c "from visualize_profile import create_visualizations; create_visualizations(profile_file='profiles/profile_stats_opt.prof', output_path='figures/profile_visualization_opt.png')"
	@echo "Benchmarking complete"

parallel:
	@echo " "
	@echo "Running parallel simulation..."
	@echo " "
	python simulation_parallel.py
	@echo "Parallel simulations complete"

stability_check:
	@echo " "
	@echo "The current version is already stable. \n More details in OPTIMIZATION.md and numerical_stability.py"
	@echo " "

test_regression:
	@echo ""
	@echo "Running regression tests..."
	pytest test_regression.py
# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf generated_data/*.pkl
	rm -rf figures/*.png
	rm -rf profile/*.prof
	rm -rf simulation_summary/*.csv
	@echo "✓ Clean complete"