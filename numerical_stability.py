
import numpy as np

# Optional: Use log-space for extreme p-values
def compute_pvalues_stable(test_statistic):
    """Numerically stable p-value computation using log-space"""
    from scipy.special import log_ndtr
    
    abs_z = np.abs(test_statistic)
    
    # For large z, use log-space to avoid underflow
    log_p_upper = log_ndtr(-abs_z)  # log(P(Z > |z|))
    log_pvalues = log_p_upper + np.log(2)  # log(2 * P(Z > |z|))
    
    pvalues = np.exp(log_pvalues)
    return pvalues

def compute_power(rejections, true_nulls):
    """
    Compute average power.
    
    Returns NaN when n_false_nulls=0 (all hypotheses are null),
    which is mathematically correct since power is undefined
    in this case.
    """
    false_nulls = ~true_nulls
    n_false_nulls = false_nulls.sum()
    
    if n_false_nulls == 0:
        return np.nan  # Undefined when no alternatives exist
    
    true_positives = (rejections & false_nulls).sum()
    power = true_positives / n_false_nulls
    return power