"""
Simple dictionary class to pass to fitter.py. The velocity axis is assumed to
be in km/s and the spectrum is in K (brightness temperature).
"""

import numpy as np
from scipy.optimize import curve_fit


class spectrum:

    def __init__(self, velax, spectrum, trans, **kwargs):
        """Initialise the spectrum class."""
        self.velax = velax
        self.spectrum = np.nan_to_num(spectrum)
        self.trans = trans
        self.mu = kwargs.get('mu', 44.)

        # Include a recaling to mimic flux calibration.
        if kwargs.get('rescale', False):
            self.spectrum *= kwargs.get('rescale', 1.0)

        # Shift and rescale the velocity axis if appropriate.
        if kwargs.get('vunit', 'm/s') == 'm/s':
            self.velax /= 1e3
        self.vlsr = kwargs.pop('vlsr', 0.0)
        if self.vlsr != 0.0:
            self.velax -= self.vlsr

        # Fit a Gaussian to estimate parameters.
        self.x0, self.dx, self.Tb = self.fitGaussian()
        self.flux = np.trapz(self.spectrum, self.velax)

        # Estimate the RMS of the spectrum using line-free regions.
        self.rms = np.nanstd(self.spectrum[abs(self.velax) > 4. * self.dx])
        return

    def fitGaussian(self):
        """Fit a Gaussian profile to the spectrum."""
        x, y = self.velax, self.spectrum
        x0 = self.velax[self.spectrum.argmax()]
        dx = 0.3
        Tb = self.spectrum.max()
        return curve_fit(self.Gaussian, x, y, p0=[x0, dx, Tb])[0]

    def Gaussian(self, x, x0, dx, Tb):
        """Gaussian function, dx is the standard deviation."""
        return Tb * np.exp(-0.5 * np.power((x0 - x) / dx, 2))
