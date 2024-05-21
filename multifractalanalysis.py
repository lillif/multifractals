import numpy as np
from scipy.stats import linregress as linreg

import warnings

import numpy as np

def calc_moments(cloudfield, R, Q):
    """
    Calculate the moments of a given cloudfield.
    
    Parameters:
    - cloudfield: numpy.ndarray
        The cloudfield array to calculate moments for.
    - R: numpy.ndarray
        An array of radii / distances to use for calculating moments.
    - Q: numpy.ndarray
        An array of q values to use for calculating moments.
    
    Returns:
    - moments: numpy.ndarray
        A 2D array containing the calculated moments for each combination of R and Q values.
    """
    moments = np.zeros([len(R), len(Q)])

    for r, r_idx in zip(R, range(len(R))):
        # ignore edge as it contains pixels from the other side of the array
        shifted = np.roll(cloudfield, -r, axis=1)[:,:-r] # [:,-r:] are the pixels that wrapped around
        deltas = np.abs(cloudfield[:,:-r] - shifted) # take same size cloudfield containing pixels we want to compare to shifted 
        moments[r_idx, :] = [np.nanmean(deltas**q) for q in Q]

    return moments


def multifractals(cloudfield, R, Q = np.arange(1, 11)):
    """
    Calculate the multifractal analysis of a given cloudfield.

    Parameters:
    - cloudfield (numpy.ndarray):
        The 2-dimensional cloudfield array.
    - R (numpy.ndarray):
        The range of distances used for computing structure functions.
    - Q (numpy.ndarray):
        The orders of the structure functions to compute. Default is 1 to 10.
    
    Returns:
    - moments (numpy.ndarray):
        The moments (or structure functions, S_q) of the cloudfield of orders Q for distances R.
    - zetas (numpy.ndarray):
        The scaling exponents (zeta_q) of the cloudfield for orders Q.

    """
    
    moments = calc_moments(cloudfield, R, Q)

    zetas = np.zeros(len(Q))
    logR = np.log(R)

    fittingRange = np.where((R >= 16) & (R <= 64))[0]

    warned=False

    for q in Q:
        normQthMoment = moments[:, q] / moments[0, q]
        logNormMoment = np.log(normQthMoment)

        a = linreg(logR[fittingRange], logNormMoment[fittingRange])[0]
        zetas[q] = a
        
        # check if fitting range is suitable (it is approx. linear in fitting range)
        end = logNormMoment[fittingRange[-1]]
        start = logNormMoment[fittingRange[0]]
        fitted_end = start + zetas[q] * (logR[fittingRange[-1]] - logR[fittingRange[0]])
        
        if not np.isclose(end, fitted_end, rtol=0.01) and not warned:
            # issues at most one warning per cloudfield
            warnings.warn(f'Warning: log(Moment) is not linear in the chosen fitting range\nlog(Moment) ends at {end}, the fitted line ends at {fitted_end}\n')
            warned=True

    return moments, zetas
