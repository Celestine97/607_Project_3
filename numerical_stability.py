
import numpy as np

# log-space for extreme p-values
def compute_pvalues_stable(test_statistic):
    """Numerically stable p-value computation using log-space"""
    from scipy.special import log_ndtr
    
    abs_z = np.abs(test_statistic)
    
    # For large z, use log-space to avoid underflow
    log_p_upper = log_ndtr(-abs_z)  # log(P(Z > |z|))
    log_pvalues = log_p_upper + np.log(2)  # log(2 * P(Z > |z|))
    
    pvalues = np.exp(log_pvalues)
    return pvalues