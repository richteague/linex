"""
A class to read in and average the slab models run from LIME. It will convert
all the units to [K] for easy fitting and remove all the spurious pixels (if
any) from an averaged spectrum. The transition which is returned should be the
index for use with the RADEX grids generated with getradexgrid.py.

It will also find p0 values for an assumed Gaussian profile. This is not
correct but will give a good starting point for fits to the profile. The
fitting can be achived with slabfitter.py.
"""

import os
import warnings
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from limepy.analysis.collisionalrates import ratefile
warnings.simplefilter("ignore")


class slabmodel:

    def __init__(self, path, molecule='cs'):
        """Slab model class."""
        self.path = path
        self.molecule = molecule
        self.nu = self.readfrequency()
        self.trans = self.readtransition()
        self.velax = self.readvelocityaxis()
        self.data = fits.getdata(path)
        if fits.getval(self.path, 'bunit') != 'K':
            self.data *= self.converttobrightness()
        self.spectrum = self.averagespectra()
        self.p0 = self.fitGaussian()
        self.x0, self.dx, self.Tb = self.p0
        self.mu = self.rates.mu
        return

    def fitGaussian(self):
        """Pseudo Gaussian parameters."""
        Tb = self.spectrum.max()
        x0 = self.velax[self.spectrum.argmax()]
        dx = np.trapz(self.spectrum, self.velax) / np.sqrt(2. * np.pi) / Tb
        return x0, dx, Tb

    def readfrequency(self):
        """Read the rest frequency [Hz]."""
        try:
            nu = fits.getval(self.path, 'restfreq')
        except KeyError:
            nu = fits.getval(self.path, 'restfrq')
        return nu

    def averagespectra(self):
        """Average all the pixels to one spectrum."""
        masked = self.data * self.maskpixels()
        return np.squeeze(np.apply_over_axes(np.nanmean, masked, (1, 2)))

    def maskpixels(self, clip=2.0):
        """Mask bad pixels with NaNs."""
        peak = np.amax(self.data, axis=0)
        medp = np.median(peak)
        return np.where(peak / medp > clip, np.nan, 1.0)[None, :, :]

    def readtransition(self):
        """Infer the LAMDA transition number from frequency."""
        path = os.getenv('PYTHONPATH') + '/limepy/aux/'
        self.rates = ratefile(path + self.molecule + '.dat')
        for j in self.rates.lines.keys():
            if self.rates.lines[j].freq == self.nu:
                return j-1
        raise ValueError('No transition found. Check molecule is correct.')

    def readvelocityaxis(self):
        """Return velocity axis in [km/s]."""
        a_len = fits.getval(self.path, 'naxis3')
        a_del = fits.getval(self.path, 'cdelt3')
        a_pix = fits.getval(self.path, 'crpix3')
        a_ref = fits.getval(self.path, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 1) * a_del) / 1e3

    def converttobrightness(self):
        """Convert [Jy/pix] units to [K]."""
        dpix = np.radians(abs(fits.getval(self.path, 'cdelt2')))
        T = 2. * np.log(2.) * sc.c**2 / sc.k / self.nu**2. * 1e-26
        return T / np.pi / dpix / dpix


def slabdictionary(identifier='', dir='./'):
    """Returns a dictionary of slab models for the fitter."""
    files = sorted([fn for fn in os.listdir(dir)
                    if fn.endswith('.fits') and identifier in fn])
    print('Selecting the following files:')
    for fn in files:
        print(r'%s%s' % (dir, fn))
    return {slabmodel(dir+fn).trans: slabmodel(dir+fn) for fn in files}
