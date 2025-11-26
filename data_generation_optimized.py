import numpy as np

def generate_alternative_means(m1, L, distribution):
    """
    Generate means for false null hypotheses.
    New: Direct NumPy repeat + concatenate
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

    means = np.repeat(levels, counts)
    # could potentially be unsorted due to rounding adjustments
    means = np.sort(means)

    return means
