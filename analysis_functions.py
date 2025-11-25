import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from save_files import load_all_simulation_results


def compute_summary_statistics(all_results):
    """
    Compute comprehensive summary statistics from simulation results.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with keys (m, m0, dist) and values as result dicts
    
    Returns:
    --------
    summary_df : pd.DataFrame
        DataFrame with summary statistics for each configuration
    """
    print("\n" + "="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    summary_data = []
    
    for key in sorted(all_results.keys()):
        m, m0, dist = key
        results = all_results[key]
        config = results['config']
        
        m1 = m - m0
        null_prop = m0 / m
        
        # Compute statistics for each method
        stats = {}
        
        for method in ['bonf', 'hoch', 'bh']:
            # Power statistics
            power = results[f'power_{method}']
            power_clean = power[~np.isnan(power)]  # Remove NaN values
            
            if len(power_clean) > 0:
                stats[f'power_{method}_mean'] = np.mean(power_clean)
                stats[f'power_{method}_median'] = np.median(power_clean)
                stats[f'power_{method}_sd'] = np.std(power_clean)
                stats[f'power_{method}_min'] = np.min(power_clean)
                stats[f'power_{method}_max'] = np.max(power_clean)
                stats[f'power_{method}_q25'] = np.percentile(power_clean, 25)
                stats[f'power_{method}_q75'] = np.percentile(power_clean, 75)
            else:
                stats[f'power_{method}_mean'] = np.nan
                stats[f'power_{method}_median'] = np.nan
                stats[f'power_{method}_sd'] = np.nan
                stats[f'power_{method}_min'] = np.nan
                stats[f'power_{method}_max'] = np.nan
                stats[f'power_{method}_q25'] = np.nan
                stats[f'power_{method}_q75'] = np.nan
            
            # FDR statistics
            fdr = results[f'fdr_{method}']
            stats[f'fdr_{method}_mean'] = np.mean(fdr)
            stats[f'fdr_{method}_median'] = np.median(fdr)
            stats[f'fdr_{method}_sd'] = np.std(fdr)
            stats[f'fdr_{method}_max'] = np.max(fdr)
        
        # Power gains (relative to Bonferroni)
        if stats['power_bonf_mean'] > 0:
            stats['power_gain_hoch'] = stats['power_hoch_mean'] / stats['power_bonf_mean']
            stats['power_gain_bh'] = stats['power_bh_mean'] / stats['power_bonf_mean']
        else:
            stats['power_gain_hoch'] = np.nan
            stats['power_gain_bh'] = np.nan
        
        # Absolute power gains
        stats['power_diff_hoch'] = stats['power_hoch_mean'] - stats['power_bonf_mean']
        stats['power_diff_bh'] = stats['power_bh_mean'] - stats['power_bonf_mean']
        
        # FDR control verification
        expected_fdr_bound = null_prop * 0.05
        stats['fdr_bh_bound'] = expected_fdr_bound
        stats['fdr_bh_controlled'] = stats['fdr_bh_mean'] <= expected_fdr_bound * 1.1  # 10% tolerance
        
        # Compile into row
        row = {
            # Configuration
            'm': m,
            'm0': m0,
            'm1': m1,
            'null_prop': null_prop,
            'distribution': dist,
            'L': config['L'],
            'alpha': config['alpha'],
            'n_reps': config['n_reps'],
            'seed': config['seed'],
            # Statistics
            **stats
        }
        
        summary_data.append(row)
        
        # Print progress
        print(f" Processed: m={m:2d}, {int(null_prop*100):3d}% null, {dist}")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['m', 'null_prop', 'distribution'])
    
    print(f"\n Computed statistics for {len(summary_df)} configurations")
    
    return summary_df


def create_summary_tables(summary_df, output_dir='results'):
    """
    Create various summary tables and save to CSV.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Full summary statistics DataFrame
    output_dir : str
        Directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLES")
    print("="*80 + "\n")
    
    # Table 1: Power comparison (main results)
    print("1. Power Comparison Table")
    power_table = summary_df[[
        'm', 'm0', 'm1', 'null_prop', 'distribution',
        'power_bonf_mean', 
        'power_hoch_mean', 
        'power_bh_mean', 
        'power_gain_bh'
    ]].copy()
    
    power_table.columns = [
        'm', 'm0', 'm1', 'null_%', 'dist',
        'bonf_power', 
        'hoch_power', 
        'bh_power',
        'bh_gain'
    ]
    
    power_path = os.path.join(output_dir, f'power_comparison_{timestamp}.csv')
    power_table.to_csv(power_path, index=False, float_format='%.4f')
    print(f"  Saved to: {power_path}")
    
    # Table 2: FDR control verification
    print("\n2. FDR Control Verification Table")
    fdr_table = summary_df[[
        'm', 'm0', 'null_prop', 'distribution',
        'fdr_bonf_mean', 'fdr_hoch_mean', 'fdr_bh_mean',
        'fdr_bh_bound', 'fdr_bh_controlled'
    ]].copy()
    
    fdr_table.columns = [
        'm', 'm0', 'null_%', 'dist',
        'bonf_fdr', 'hoch_fdr', 'bh_fdr',
        'bh_bound', 'controlled'
    ]
    
    fdr_path = os.path.join(output_dir, f'fdr_control_{timestamp}.csv')
    fdr_table.to_csv(fdr_path, index=False, float_format='%.5f')
    print(f"  Saved to: {fdr_path}")
    
    # Table 3: Complete summary (all statistics)
    print("\n3. Complete Summary Table")
    complete_path = os.path.join(output_dir, f'complete_summary_{timestamp}.csv')
    summary_df.to_csv(complete_path, index=False, float_format='%.6f')
    print(f"   ✓ Saved to: {complete_path}")
    
    print("\n All tables saved")
    
    return power_path, fdr_path, complete_path
