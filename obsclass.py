"""
A class, like diskclass and slabclass, to read in the shifted data from ALMA
and link it to fitter.py. Similar to diskclass, a specific radius (in arcsec)
can be specified through getdic(r).

The format of the input is self.datas[i][j]
    [i] = [0,1,2]; which of the transitions.
    [j] = [0,1,2,3]; velax, radii, spectra, noise.

"""


import os
import warnings
import numpy as np
warnings.simplefilter("ignore")


class obsglobal:

    def __init__(self, name, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.dir = kwargs.get('dir', './')
        self.name = name
        self.molecule = kwargs.get('molecule', 'cs')
        self.files = self.findfiles()
        self.datas = [np.load(fn) for fn in self.files]
        self.velaxs = np.squeeze([d[0] for d in self.datas])
        self.radii = np.squeeze([d[1] for d in self.datas])
        if not (self.radii == self.radii[0]).all():
            raise ValueError('Mismatched radial sampling.')
        else:
            self.radii = np.unique(self.radii)
        self.spectra = [d[2] for d in self.datas]
        self.trans = [self.findtransitions(fn) for fn in self.files]
        self.mu = kwargs.get('mu', 44.)
        return

    def findfiles(self):
        """Find the approrpriate files."""
        files = sorted([self.dir + fn for fn in os.listdir(self.dir)
                        if fn.endswith('average.npy')
                        and self.name in fn])
        if self.verbose:
            print('Selecting the following files:')
            for fn in files:
                print(r'%s' % (fn))
        return files

    def findtransitions(self, path):
        """Read in the CS transitions."""
        if '76' in path:
            return 6
        elif '54' in path:
            return 4
        else:
            return 2

    def getdict(self, radius):
        """Returns a dictionary of the spectra at a given (nearest) radius."""
        ridx = abs(self.radii - radius).argmin()
        if self.verbose:
            print('Requested %.3f", found %.3f".' % (radius, self.radii[ridx]))
        return {t: obsmodel(self.spectra[i][ridx], self.velaxs[i], self.mu)
                for i, t in enumerate(self.trans)}


class obsmodel:

    def __init__(self, spectrum, velax, mu=44.):
        self.velax = velax
        self.spectrum = spectrum
        self.p0 = self.fitGaussian()
        self.x0, self.dx, self.Tb = self.p0
        self.mu = mu
        return

    def fitGaussian(self):
        """Pseudo Gaussian parameters."""
        Tb = self.spectrum.max()
        x0 = self.velax[self.spectrum.argmax()]
        dx = np.trapz(self.spectrum, self.velax) / np.sqrt(2. * np.pi) / Tb
        return x0, dx, Tb


def obsdictionary(identifier, radius, dir='./', molecule='cs'):
    """Return a dictionary for fitter.py."""
    og = obsglobal(name=identifier, dir=dir, molecule=molecule)
    return og.getdict(radius)
