# run_analysis.py
# get summary statistics and create summary tables

from save_files import load_all_simulation_results
from analysis_functions import compute_summary_statistics, create_summary_tables

if __name__ == "__main__":
    

    print("\n" + "="*80)
    print("loading all simulation raw results...")
    print("="*80 + "\n")
    all_results = load_all_simulation_results('generated_data/')
    
    summary_df = compute_summary_statistics(all_results)
    
    # Create summary tables
    create_summary_tables(summary_df, output_dir='simulation_summary/')
    
    
    print("All complete.")