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


class averagedspectra:
    """Read in a .npy of a spectrum."""

    def __init__(self, path, radius, **kwargs):

        self.data = np.load(path)
        self.verbose = kwargs.get('verbose', True)
        self.trans = readtrans(path)

        self.velax = self.data[0]
        self.spectrum = self.data[2][abs(self.data[1] - radius).argmin()]
        self.p0 = self.fitGaussian()
        self.Tb, self.x0, self.dx, self.iTb = self.p0

        if self.verbose:
            rpnt = self.data[1][abs(self.data[1] - radius).argmin()]
            print("Asked for %.2f, found %.2f." % (radius, rpnt))
            p0 = ', '.join(['%.3f' % p for p in self.p0])
            print('p0 = (' + p0 + ').')

        self.molecule = kwargs.get('molecule', 'cs')
        self.mu = kwargs.get('mu', 44.)

        return

    def fitGaussian(self):
        """Estimate Gaussian parameters for the spectrum."""
        Tb = self.spectrum.max()
        x0 = self.velax[self.spectrum.argmax()]
        iTb = np.trapz(self.spectrum, self.velax)
        dx = iTb / np.sqrt(2 * np.pi) / Tb
        return Tb, x0, dx, iTb


def averagedictionary(name, radius, dir='./', trans=None, **kwargs):
    """Returns a dictionary of averaged spectra for the fitter."""
    files = [f for f in os.listdir(dir) if f.endswith('npy') and name in f]
    if trans is not None:
        files = [f for f in files if readtrans(f) in trans]
    files = sorted(files)
    if kwargs.get('verbose', True):
        print("Selecting the following files:\n" + "\n".join(files) + "\n")
    return {readtrans(f): averagedspectra(f, radius, **kwargs) for f in files}


def readtrans(path):
    """Read the transition, return Jlower."""
    if '_76_' in path:
        return 6
    elif '_54_' in path:
        return 4
    elif '_32_' in path:
        return 2
    else:
        raise ValueError("Cannot parse filename.")

class obsglobal:

    def __init__(self, name, trans=None, **kwargs):
        self.verbose = kwargs.get('verbose', True)
        self.dir = kwargs.get('dir', './')
        self.name = name
        self.molecule = kwargs.get('molecule', 'cs')
        self.files = self.findfiles(trans)
        self.datas = [np.load(fn) for fn in self.files]
        self.velaxs = [d[0] for d in self.datas]
        self.radii = [d[1] for d in self.datas]

        # Check that the radial points of each transition are the same.
        if len(self.files) > 1:
            if not all([np.array_equal(r, self.radii[0]) for r in self.radii]):
                raise ValueError('Mismatched radial sampling.')
            else:
                self.radii = np.unique(self.radii)
            self.spectra = [d[2] for d in self.datas]
            self.trans = [self.findtransitions(fn) for fn in self.files]
        else:
            self.spectra = self.datas[0][2]
            self.trans = [self.findtransitions(self.files[0])]

        self.mu = kwargs.get('mu', 44.)
        return

    def findfiles(self, trans):
        """Find the approrpriate files."""
        files = sorted([self.dir + fn for fn in os.listdir(self.dir)
                        if fn.endswith('.npy') and self.name in fn])
        if trans is not None:
            files = sorted([fn for fn in files
                            if self.findtransitions(fn) in trans])
        if self.verbose:
            print('Selecting the following files:')
            for fn in files:
                print(r'%s' % (fn))
        return files

    def findtransitions(self, path):
        """Read in the CS transitions."""
        if '_76_' in path:
            return 6
        elif '_54_' in path:
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


def obsdictionary(identifier, radius, dir='./', molecule='cs', trans=None):
    """Return a dictionary for fitter.py."""
    og = obsglobal(name=identifier, dir=dir, molecule=molecule, trans=trans)
    return og.getdict(radius)
