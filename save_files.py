import pickle
import glob
import os
from datetime import datetime

def save_simulation_results(results, config, output_dir='generated_data/'):
    """
    Save simulation results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_simulation
    config : dict
        Configuration dictionary
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    filepath : str
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename
    m = config['m']
    m0 = config['m0']
    dist = config['distribution']
    L = config['L']
    n_reps = config['n_reps']
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"sim_m{m}_m0{m0}_{dist}_L{L}_n{n_reps}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save as pickle (preserves all data)
    with open(filepath, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': results
        }, f)
    
    print(f"Saved results to: {filepath}")
    return filepath


def load_simulation_results(filepath):
    """
    Load single simulation results.
    
    Parameters:
    -----------
    filepath : str
        Path to saved results file
    
    Returns:
    --------
    config : dict
        Configuration dictionary
    results : dict
        Results dictionary
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['config'], data['results']

def load_all_simulation_results(output_dir='generated_data'):
    """
    Load all simulation results from a directory.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing saved results
    
    Returns:
    --------
    all_results : dict
        Dictionary with keys (m, m0, dist) and values as result dicts
    """
    
    # Find all .pkl files
    pattern = os.path.join(output_dir, 'sim_*.pkl')
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No simulation files found in {output_dir}/")
    
    print(f"\nLoading {len(files)} simulation results...")
    
    all_results = {}
    
    for filepath in sorted(files):
        # Load individual result
        config, results = load_simulation_results(filepath)
        
        # Create key
        m = config['m']
        m0 = config['m0']
        dist = config['distribution']
        key = (m, m0, dist)
        
        # Store
        all_results[key] = results
    
    print(f"\n Loaded {len(all_results)} configurations")
    
    return all_results