"""
A similar class to the slabclass, however for LIME models of disks. This
assumes that they have been run with limepy to easily extract the spectra.

The disk is azimuthally averaged, taking into account the Keplerian shift of
the lines (needs to be tested) and then binned radially. A dictionary suitable
for fitter.py can be retrieved through the getdic(r) function where r is the
radius the spectra are taken from.

"""

import os
import warnings
import numpy as np
from astropy.io import fits
from limepy.analysis.analysecube import cube
from limepy.analysis.collisionalrates import ratefile
warnings.simplefilter("ignore")


class diskglobal:

    def __init__(self, name, dir='./', bins=None, molecule='cs', noise=None):
        """Read in and prepare the disk model dictionary."""
        self.dir = dir
        self.name = name
        self.molecule = molecule
        self.files = self.findfiles()
        self.cubes = [cube(self.dir+fn) for fn in self.files]
        self.trans = [self.readtransition(self.dir+fn) for fn in self.files]
        self.velaxs = [self.readvelocityaxis(self.dir+fn) for fn in self.files]
        if bins is None:
            bins = np.linspace(10., 180., 18)
        self.rvals, self.spectra = self.averagespectra(bins=bins, nbins=None)
        self.mu = self.rates.mu
        self.noise = noise
        return

    def readfrequency(self, path):
        """Read the rest frequency [Hz]."""
        try:
            nu = fits.getval(path, 'restfreq')
        except KeyError:
            nu = fits.getval(path, 'restfrq')
        return nu

    def findfiles(self):
        """Read in the cube files."""
        files = sorted([fn for fn in os.listdir(self.dir)
                        if fn.endswith('.fits') and self.name in fn])
        print('Selecting the following files:')
        for fn in files:
            print(r'%s%s' % (self.dir, fn))
        return files
        return [cube(self.dir+fn) for fn in files]

    def averagespectra(self, bins, nbins):
        """Return the azimuthally averaged spectra."""
        rawout = [c.averagespectra(bins=bins, nbins=nbins)
                  for c in self.cubes]
        radius = np.unique([s[0] for s in rawout])
        spectra = np.squeeze([s[1] for s in rawout])
        return radius, spectra

    def readtransition(self, path):
        """Infer the LAMDA transition number from frequency."""
        nu = self.readfrequency(path)
        path = os.getenv('PYTHONPATH') + '/limepy/aux/'
        self.rates = ratefile(path + self.molecule + '.dat')
        for j in self.rates.lines.keys():
            if self.rates.lines[j].freq == nu:
                return j-1
        raise ValueError('No transition found. Check molecule is correct.')

    def readvelocityaxis(self, path):
        """Return velocity axis in [km/s]."""
        a_len = fits.getval(path, 'naxis3')
        a_del = fits.getval(path, 'cdelt3')
        a_pix = fits.getval(path, 'crpix3')
        a_ref = fits.getval(path, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 1) * a_del) / 1e3

    def getdic(self, radius):
        """Returns a dictionary of the spectra at the given radius."""
        ridx = abs(self.rvals-radius).argmin()
        return {t: diskmodel(self.spectra[i][ridx], self.velaxs[i], self.mu,
                             self.noise) for i, t in enumerate(self.trans)}


class diskmodel:

    def __init__(self, spectrum, velax, mu=44., noise=None):
        self.velax = velax
        self.spectrum = spectrum
        self.p0 = self.fitGaussian()
        self.x0, self.dx, self.Tb = self.p0
        self.mu = mu
        if noise is not None:
            if noise > 0.1:
                print("Noise is greater than 10%, are you sure?")
            noise *= self.Tb * np.random.randn(self.spectrum.size)
        self.spectrum += noise
        return

    def fitGaussian(self):
        """Pseudo Gaussian parameters."""
        Tb = self.spectrum.max()
        x0 = self.velax[self.spectrum.argmax()]
        dx = np.trapz(self.spectrum, self.velax) / np.sqrt(2. * np.pi) / Tb
        return x0, dx, Tb


def diskdictionary(identifier, radius, dir='./', molecule='cs',
                   bins=None, noise=None):
    """Return a dictionary for fitter.py."""
    dg = diskglobal(name=identifier, dir=dir, bins=bins,
                    molecule=molecule, noise=noise)
    return dg.getdic(radius)
