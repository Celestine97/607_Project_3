import numpy as np

def generate_alternative_means(m1, L, distribution):
    """
    Generate means for false null hypotheses.
    
    Returns sorted means to ensure monotonic relationship
    across different distributions with same m.
    """
    if m1 == 0:
        return np.array([])
    
    levels = np.array([L/4, L/2, 3*L/4, L])
    
    if distribution == 'D':
        weights = np.array([4, 3, 2, 1])
    elif distribution == 'E':
        weights = np.array([1, 1, 1, 1])
    elif distribution == 'I':
        weights = np.array([1, 2, 3, 4])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    counts = np.round(weights / weights.sum() * m1).astype(int)
    diff = m1 - counts.sum()
    if diff != 0:
        counts[-1] += diff
    
    counts = np.maximum(counts, 0)
    
    # Create means
    means = []
    for level, count in zip(levels, counts):
        means.extend([level] * count)
    
    means = np.array(means)
    
    means = np.sort(means)
    
    return means

def generate_base_data(config):
    """
    Generate base random normal data that will be reused across configurations.
    
    Parameters:
    -----------
    m : int
        Number of hypotheses
    n_reps : int
        Number of replications
    seed : int
        Random seed
    
    Returns:
    --------
    base_data : np.ndarray
        Shape (n_reps, m) - standard normal data to be reused
    """
    m = config['m']
    n_reps = config['n_reps']
    seed = config['seed']
    rng = np.random.default_rng(seed)
    base_data = rng.standard_normal(size=(n_reps, m))
    return base_data