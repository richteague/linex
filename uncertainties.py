"""
Calculate the flux calibration uncertainties for the values. We first calculate
the uncertainties in {T, N} from flux calibration (as this only affects the
peak of the line). Then from dM we are able to constrain the maximum dT.
"""

import numpy as np
import scipy.constants as sc


# Physical constants for CS.
B0 = 2.449913e10
mu = 44.
nu = {2: 146.9690287e9, 4: 244.9355565e9, 6: 342.8828503}
gu = {2: 7., 4: 11., 6: 15.}


def get_uncertainties(T, M, dT, dM, fluxcal):
    """Returns all the uncertainties for [T, N, M] from flux calibration."""
    dT_fc = np.amax([dT, fluxcal_temperature(fluxcal, T)], axis=0)
    dT_fc = np.amin([dT_fc, maxfluxcal_temperature(T, M, dT, dM)], axis=0)
    dN_fc = fluxcal_columndensity(fluxcal)
    dN_fc = np.array([dN_fc for _ in range(dT_fc.size)])
    dM_fc = maxfluxcal_turbulence(T, M, dT, dM)
    return dT_fc, dN_fc, dM_fc


def fluxcal_uncertainties(fluxcal, T):
    """Calculates the uncertainties from the flux calibration."""
    dT = fluxcal_temperature(fluxcal, T)
    dN = fluxcal_columndensity(fluxcal)
    return dT, dN


def fluxcal_temperature(fluxcal, T):
    """Uncertainty in temperature from flux calibration uncertainties."""
    dT = sc.k / sc.h / B0 / partition_function(T)
    dT = [dT - sc.h * nu[j] / sc.k / T**2 for j in nu.keys()]
    return fluxcal / np.amin(dT, axis=0)


def fluxcal_columndensity(fluxcal):
    """Estimate the error on the log column density from flux calibration."""
    return 0.434 * fluxcal


def partition_function(T):
    """Partition function for CS. Values from JPL."""
    return sc.k * T / sc.h / B0 + 1. / 3.


def maxfluxcal_temperature(T, M, dT, dM):
    """Max uncertainty in temperature from flux calibration uncertainties."""
    return 2 * uncertainty_linewidth(T, M, dT, dM) * T / linewidth(T, M)


def maxfluxcal_turbulence(T, M, dT, dM):
    """Max uncertainty in turbulence from flux calibration uncertainties."""
    return (uncertainty_linewidth(T, M, dT, dM) * linewidth(T, M) / M /
            soundspeed(T)**2)


def uncertainty_linewidth(T, M, dT, dM):
    """Combined unceratinty on the linewidth."""
    dV = dM * M * soundspeed(T)**2 / linewidth(T, M)
    return dV + dT * linewidth(T, M) / 2. / T


def linewidth(T, M):
    """Doppler width of line, dV [m/s]."""
    return np.hypot(thermalwidth(T), nonthermalwidth(M, T))


def nonthermalwidth(M, T):
    """Non-thermal Doppler width [m/s]."""
    return M * soundspeed(T)


def uncertainty_nonthermalwidth(M, dM, T, dT):
    """Uncertainty in non-thermal Doppler width [m/s]."""
    
    return


def thermalwidth(T):
    """Thermal Doppler width [m/s]."""
    return np.sqrt(2. * sc.k * T / mu / sc.m_p)


def uncertainty_thermalwidth(T, dT):
    """Uncertainty in thermal Doppler width [m/s]."""
    return 0.5 * dT * np.sqrt(2 * sc.k / mu / sc.m_p / T)


def soundspeed(T):
    """Soundspeed of gas [m/s]."""
    return np.sqrt(sc.k * T / 2.34 / sc.m_p)


def uncertainty_soundspeed(T, dT):
    """Uncertainty in soundspeed of the gas [m/s]."""
    return 0.5 * dT * np.sqrt(sc.k / 2.34 / sc.m_p / T)
