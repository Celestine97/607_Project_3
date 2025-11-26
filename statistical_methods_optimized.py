import numpy as np
from scipy.special import ndtr  # Faster than stats.norm.cdf

def compute_pvalues(test_statistic):
    """
    Compute two-sided p-values for z-tests using analytical formula.

    Parameters:
    -----------
    test_statistic : np.ndarray
        Test statistics for the hypotheses

    Returns:
    --------
    pvalues : np.ndarray
        Two-sided p-values
    """
    # Use ndtr (faster C implementation) instead of stats.norm.cdf
    # ndtr(x): the standard normal CDF
    pvalues = 2 * (1 - ndtr(np.abs(test_statistic)))
    return pvalues


def hochberg_method(pvalues, alpha=0.05):
    """
    Hochberg (1988) step-up multiple testing procedure.

    Removed O(m) backwards loop, replaced with vectorized comparison.

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

    # Vectorized comparison instead of loop
    i_values = np.arange(1, m + 1)
    thresholds = alpha / (m + 1 - i_values)

    # Vectorized comparison
    valid = sorted_pvalues <= thresholds

    # Find the largest i (if any exist)
    if np.any(valid):
        k = np.max(np.where(valid)[0]) + 1  # +1 because we want count, not index
    else:
        k = 0

    # Reject all hypotheses up to k
    rejections = np.zeros(m, dtype=bool)
    if k > 0:
        rejections[sorted_indices[:k]] = True

    return rejections


def benjamini_hochberg_method(pvalues, q=0.05):
    """
    Benjamini-Hochberg (1995) FDR controlling procedure.

    Removed O(m) backwards loop, replaced with vectorized comparison.

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

    # Vectorized comparison instead of loop
    i_values = np.arange(1, m + 1)
    thresholds = (i_values / m) * q

    # Vectorized comparison
    valid = sorted_pvalues <= thresholds

    # Find the largest i (if any exist)
    if np.any(valid):
        k = np.max(np.where(valid)[0]) + 1  # +1 because we want count, not index
    else:
        k = 0

    # Reject all hypotheses up to k
    rejections = np.zeros(m, dtype=bool)
    if k > 0:
        rejections[sorted_indices[:k]] = True

    return rejections
