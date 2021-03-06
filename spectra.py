"""
Simple dictionary class to pass to fitter.py. The velocity axis is assumed to
be in km/s and the spectrum is in K (brightness temperature).
"""

import numpy as np
from scipy.optimize import curve_fit


class spectrum:

    def __init__(self, velax, spectrum, trans, **kwargs):
        """Initialise the spectrum class."""

        # Local parameters.
        self.trans = trans
        self.mu = kwargs.get('mu', 44.)

        # Shift and rescale the velocity axis if appropriate.
        self.velax = velax
        if kwargs.get('vunit', 'm/s') == 'm/s':
            self.velax /= 1e3
        self.vlsr = kwargs.pop('vlsr', 0.0)
        if self.vlsr != 0.0:
            self.velax -= self.vlsr

        # Read in the spectrum and apply corrections.
        self.spectrum = np.nan_to_num(spectrum)
        self.spectrum -= self.estimateOffset()
        if kwargs.get('rescale', False):
            self.spectrum *= kwargs.get('rescale', 1.0)

        # Fit a Gaussian to estimate parameters and the noise.
        self.x0, self.dx, self.Tb = self.fitGaussian()
        self.flux = np.trapz(self.spectrum, self.velax)
        self.rms = self.estimateNoise()
        return

    def fitGaussian(self):
        """Fit a Gaussian profile to the spectrum."""
        x, y = self.velax, self.spectrum
        x0 = self.velax[self.spectrum.argmax()]
        dx = 0.3
        Tb = self.spectrum.max()
        return curve_fit(self.Gaussian, x, y, p0=[x0, dx, Tb], maxfev=10000)[0]

    def Gaussian(self, x, x0, dx, Tb):
        """Gaussian function, dx is the standard deviation."""
        return Tb * np.exp(-0.5 * np.power((x0 - x) / dx, 2))

    def estimateOffset(self):
        """Estimate the offset and correct for it."""
        center = self.velax[self.spectrum.argmax()]
        masked = abs(self.velax - center) > 0.5
        return np.nanmean(self.spectrum[masked])

    def estimateNoise(self):
        """Estimate the noise."""
        center = self.velax[self.spectrum.argmax()]
        masked = abs(self.velax - center) > 0.5
        return np.nanstd(self.spectrum[masked])
