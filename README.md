# Multifractal Analysis of Clouds

This repository accompanies the paper "[Multifractal Analysis for Evaluating the Representation of Clouds in Global Kilometre-Scale Models](https://doi.org/10.22541/au.171813202.26984086/v1)" by Lilli J. Freischem, Philipp Weiss, Hannah M. Christensen, and Philip Stier, submitted to Geophysical Research Letters. Here, we provide Python scripts and Jupyter Notebook examples for conducting multifractal analysis.




## Theoretical Overview

For some one-dimensional signal $\theta (x)$, we compute the $q\text{th}$ order structure function as:

$S_q (r)=\langle|\theta(x+r)-\theta(x)|^q \rangle$,

where $r$ is a finite distance in $x$, and $\langle ... \rangle$ indicates a spatial average over all $x$.


Within the scaling range of the signal $\theta(x)$, the magnitude of the structure function is proportional to the distance $r$, raised to some exponent $\zeta_q$:

$S_q (r)\propto r^{\zeta_q}$,

such that $\log S_q (r)$ is approximately linear in $\log r$.

We parameterise the calculated $\zeta_q$ exponents as

$\zeta_q=\frac{aq}{1+aq / \zeta_\infty}$,

using the two-parameter fit introduced by [Pierrehumbert, 1996](https://doi.org/10.1029/96GL01121).

## Key Functions

`multifractalanalysis.py` contains the `multifractals` function which computes structure functions and multifractal scaling parameters from a 2-dimensional cloudfield.

```python
def multifractals(cloudfield, R, Q = np.arange(1, 11)):
    """
    Calculate multifractal scaling exponents of a given cloudfield.

    Parameters:
    - cloudfield: numpy.ndarray
        The cloudfield array to calculate moments for.
    - R: numpy.ndarray
        An array of separation distances to use for calculating moments.
    - Q: numpy.ndarray
        An array of exponents q to use for calculating moments.
    
    Returns:
    - moments: numpy.ndarray
        A 2D array containing the calculated moments for all separation distances R and exponents Q.
    - zetas: numpy.ndarray
        A 1D array of scaling exponents (zeta_q) of the cloudfield for orders Q.
    """
```

## Examples

`compute_multifractals.ipynb` contains a notebook demonstrating how to use `multifractalanalysis.py` for computing structure functions, scaling exponents, and multifractal parameters.

## Citing

```bibtex
@software{freischem2024,
  author       = {Freischem, Lilli Johanna},
  title        = {Multifractal Analysis of Clouds},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.13832882},
  url          = {https://doi.org/10.5281/zenodo.13832882}
}
```


<!-- ### DOI

[![DOI](https://zenodo.org/badge/DOI/freischem/zenodo.multifractals.svg)](https://zenodo.org/badge/latestdoi/freischem-multifractals) -->
