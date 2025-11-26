# Optimization Report


---

## 1. Algorithmic Improvements

### 1.1 P-value Computation: (50% of runtime before)
**Original implementation:**
```python
from scipy import stats
pvalues = 2 * (1 - stats.norm.cdf(np.abs(test_statistic)))
```

The `stats.norm.cdf()` function is a high-level interface, which takes more time.

**Improvement:**
```python
# Lower-level C implementation
from scipy.special import ndtr  # Low-level C function
pvalues = 2 * (1 - ndtr(np.abs(test_statistic)))
```

**Trade-offs:**
- less readability
- limited for standard normal distribution: only works for $N(0,1)$, cannot handle non-standard situations
- no input check, less stable than `stats.norm.cdf()` 

---

### 1.2 Generate Alternative Means: Eliminated Python List Operations

**Original implementation:**
```python
# Python list operations
means = []
for level, count in zip(levels, counts):
    means.extend([level] * count)
means = np.array(means)
```

**Solution:**
```python
# Direct NumPy operation
means = np.repeat(levels, counts)
```

**Benefits:**
- **Eliminated list overhead:** No Python list allocation/extension
- **Less memory fragmentation:** Direct array creation

**Trade-offs:**
- less readability(?)

---

## 2. Array Programming
### 2.1 Hochberg Method: Backward Loop

**Original implementation:**
```python
for i in range(m, 0, -1):
    if sorted_pvalues[i-1] <= alpha / (m + 1 - i):
        k = i
        break
```

The backwards loop iterates through all m hypotheses, making this O(m) after the O(m log m) sort.

**Improvement:**
```python

i_values = np.arange(1, m + 1)
thresholds = alpha / (m + 1 - i_values)
valid = sorted_pvalues <= thresholds  # Vectorized comparison

if np.any(valid):
    k = np.max(np.where(valid)[0]) + 1
else:
    k = 0
```



### 2.2 Benjamini-Hochberg Method: Backward Loop


**Original implementation:**
```python
for i in range(m, 0, -1):
    if sorted_pvalues[i-1] <= (i / m) * q:
        k = i
        break
```

**Improvement:**
```python
i_values = np.arange(1, m + 1)
thresholds = (i_values / m) * q
valid = sorted_pvalues <= thresholds  # Vectorized comparison

if np.any(valid):
    k = np.max(np.where(valid)[0]) + 1
else:
    k = 0
```


---

## 3. Numerical Stability

### 3.1 Extreme p value computation - use log scale
**Improved Version:**
```python
# Lower-level C implementation
from scipy.special import ndtr  # Low-level C function
pvalues = 2 * (1 - ndtr(np.abs(test_statistic)))
```

**Improved Version with Better Numerical Stability:**
```python
# use log-version
from scipy.special import log_ndtr
abs_z = np.abs(test_statistic)
# use log-space to avoid underflow
log_p_upper = log_ndtr(-abs_z)  # log(P(Z > |z|))
log_pvalues = log_p_upper + np.log(2)  # log(2 * P(Z > |z|))
pvalues = np.exp(log_pvalues) # or we can just use log-scale p-value for further computation
```
---
### 3.2 Computational Stability

Generally we can implement LogSumExp trick to make key steps more stable; but in this implementation, it's mainly about comparison and simple additive operations, thus not implemented.

---

## 4. Performance Impact


