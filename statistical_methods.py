import numpy as np
from scipy import stats

def compute_pvalues(test_statistic):
    """
    Compute two-sided p-values for z-tests.
    
    Parameters:
    -----------
    test_statistic : np.ndarray
        Test statistics for the hypotheses
    
    Returns:
    --------
    pvalues : np.ndarray
        Two-sided p-values
    """
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))
    return pvalues


def bonferroni_method(pvalues, alpha=0.05):
    """
    Bonferroni multiple testing procedure.
    
    Parameters:
    -----------
    pvalues : np.ndarray
        P-values for all hypotheses
    alpha : float
        Significance level
    
    Returns:
    --------
    rejections : np.ndarray
        Boolean array indicating rejections
    """
    m = len(pvalues)
    threshold = alpha / m
    rejections = pvalues <= threshold
    return rejections


def hochberg_method(pvalues, alpha=0.05):
    """
    Hochberg (1988) step-up multiple testing procedure.
    
    Parameters:
    -----------
    pvalues : np.ndarray
        P-values for all hypotheses
    alpha : float
        Significance level
    
    Returns:
    --------
    rejections : np.ndarray
        Boolean array indicating rejections
    """
    m = len(pvalues)
    
    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    
    # Find largest i where P_(i) <= i*alpha/(m+1-i)
    k = 0
    for i in range(m, 0, -1):
        if sorted_pvalues[i-1] <= 1 * alpha / (m + 1 - i):
            k = i
            break
    
    # Reject all hypotheses up to k
    rejections = np.zeros(m, dtype=bool)
    if k > 0:
        rejections[sorted_indices[:k]] = True
    
    return rejections


def benjamini_hochberg_method(pvalues, q=0.05):
    """
    Benjamini-Hochberg (1995) FDR controlling procedure.
    
    Parameters:
    -----------
    pvalues : np.ndarray
        P-values for all hypotheses
    q : float
        FDR level to control
    
    Returns:
    --------
    rejections : np.ndarray
        Boolean array indicating rejections
    """
    m = len(pvalues)
    
    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]
    
    # Find largest i where P_(i) <= (i/m)*q
    k = 0
    for i in range(m, 0, -1):
        if sorted_pvalues[i-1] <= (i / m) * q:
            k = i
            break
    
    # Reject all hypotheses up to k
    rejections = np.zeros(m, dtype=bool)
    if k > 0:
        rejections[sorted_indices[:k]] = True
    
    return rejections