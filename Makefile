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

profiling:
	@echo " "
	@echo "Baseline profiling..."
	@echo " "
	python baseline_profile.py
	snakeviz profile_stats.prof
	python visualize _profile.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf generated_data/*.pkl
	rm -rf figures/*.png
	rm -rf simulation_summary/*.csv
	rm -rf profile_stats.prof
	@echo "✓ Clean complete"