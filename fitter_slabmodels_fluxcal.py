"""
Class to fit slab models with a single temperature and density. Assume that
the free parameers are {Tkin, nH2, NCS, M, x0, x1, x2}.
As the slab models are relatively simple we do not consider flux calibration
uncertainties, correlated noise or different Mach values for each line.

Input:
    dic - slab dictionary of spectra from slabclass.py.
    gridpath - path to the pre-calculated intensities.
    rms - the variance of the noise values used in corrupting the spectra [%].

"""

import emcee
import george
import numpy as np
import scipy.constants as sc
from linex.radexgridclass import radexgrid
import matplotlib.pyplot as plt
from george.kernels import ExpSquaredKernel as ExpSq


class fitdict:

    def __init__(self, dic, gridpath, rms=1e-10, **kwargs):
        """Fit the slab models."""
        self.dic = dic
        self.trans = sorted([k for k in self.dic.keys()])
        self.peaks = [self.dic[k].Tb for k in self.trans]
        self.widths = [self.dic[k].dx for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.grid = radexgrid(gridpath)
        self.mu = self.dic[self.trans[0]].mu
        self.rms = np.array([rms * Tb for Tb in self.peaks])
        if kwargs.get('plot', False):
            self.plotlines()
        return

    def _lnprob(self, theta, fc=0.0):
        """Log-probability function."""
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta, fc)

    def _lnprior(self, theta):
        """
        Log-prior function. Temperature, density and column density must be
        in the RADEX grid. Line centre must be on the velocity axis. The Mach
        number can be any positive value less than one. The range of widths
        allowed by the radexgrid should cover everything up to M = 1.
        """
        temp, dens, sigma, mach, x0s, vamp, vcorr = self._parse(theta)
        if not self.grid.in_grid(temp, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dens, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf
        for x0, velax in zip(x0s, self.velaxs):
            if x0 < velax[0] or x0 > velax[-1]:
                return -np.inf
        if abs(mach) > 0.5:
            return -np.inf

        # Hyper-parameters for the noise model.

        if not 0. < vamp < 0.1:
            return -np.inf
        if not -5. < vcorr < 0.:
            return -np.inf
        return 0.0

    def _parse(self, t):
        """Parses theta values given self.params."""
        x0s = [t[j] for j in -(np.arange(len(self.trans)) + 3)[::-1]]
        return t[0], t[1], t[2], t[3], x0s, t[-2], t[-1]

    def _lnlike(self, theta, fc=0.0):
        """Log-likelihood for fitting peak brightness."""
        temp, dens, sigma, mach, x0s, vamp, vcorr = self._parse(theta)
        toiter = zip(self.trans, self.velaxs, x0s)
        models = [self._spectrum(j, temp, dens, sigma, v, x0, mach)
                  for j, v, x0 in toiter]
        lnx2 = 0.0
        noises = [george.GP(vamp * ExpSq(10**vcorr)) for dy in self.rms]
        for k, velax, dy in zip(noises, self.velaxs, self.rms):
            k.compute(velax, dy)
        for k, mod, obs in zip(noises, models, self.spectra):
            lnx2 += k.lnlikelihood(mod * (1. + fc * np.random.randn()) - obs)
        return lnx2

    def _spectrum(self, j, t, d, s, x, x0, mach):
        """Returns a spectrum on the provided velocity axis."""
        dV = self.linewidth(t, mach)
        A = self.grid.intensity(j, dV, t, d, s)
        return self.gaussian(x, x0, dV, A)

    def emcee(self, **kwargs):
        """Run emcee fitting just the profile peaks."""

        # Default values for the MCMC fitting.

        nwalkers = kwargs.get('nwalkers', 200)
        nburnin = kwargs.get('nburnin', 500)
        nsteps = kwargs.get('nsteps', 500)

        # Set up the parameters and the sampler. Should allow for any multiple
        # of spectra to be simultaneously fit.

        self.params = ['temp', 'dens', 'sigma', 'mach']
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i]
        self.params += ['var', 'logR']
        ndim = len(self.params)

        # For the sampling, we first sample the whole parameter space. After
        # the burn-in phase, we recenter the walkers around the median value in
        # each parameter and start from there.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        pos = [self.grid.random_samples(p.split('_')[0], nwalkers)
               for p in self.params if p.split('_')[0] in self.grid.parameters]
        pos += [np.random.uniform(0.0, 1.0, nwalkers)]
        for i in range(len(self.trans)):
            pos += [np.zeros(nwalkers)]

        # Hyper-parameters for the noise model. 'vamp' is the variance of the
        # noise while 'vcorr' is the correlation length.

        pos += [np.mean(self.rms)**2 + np.random.randn(nwalkers)]
        pos += [np.random.uniform(-4, -1, nwalkers)]

        print("Running first burn-in...")
        pos, lp, _ = sampler.run_mcmc(np.squeeze(pos).T, nburnin)
        pos = pos[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler.reset()

        # To speed up the initial burn-in, we only apply flux calibration
        # uncertainties in the second call of the sampler. This rescales the
        # model spectra by some fraction before calling the chi-squared
        # likelihood.

        print("Running second burn-in...")
        sampler.args = (kwargs.get('fluxcal', 0.0))
        pos, _, _ = sampler.run_mcmc(pos, nburnin+nsteps)
        return sampler, sampler.flatchain

    # General Functions.

    def plotlines(self, ax=None):
        """Plot the emission lines."""
        if ax is None:
            fig, ax = plt.subplots()
        for i, t in enumerate(self.trans):
            ax.plot(self.velaxs[i], self.spectra[i], lw=1.25,
                    label=r'J = %d - %d' % (t+1, t))
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Brightness [K]')
        ax.legend(frameon=False)
        return

    def thermalwidth(self, T):
        """Thermal Doppler width [km/s]."""
        return np.sqrt(2. * sc.k * T / self.mu / sc.m_p) / 1e3

    def soundspeed(self, T):
        """Soundspeed of gas [km/s]."""
        return np.sqrt(sc.k * T / 2.34 / sc.m_p) / 1e3

    def linewidth(self, T, Mach):
        """Standard deviation of line, dV [km/s]."""
        v_therm = self.thermalwidth(T)
        v_turb = Mach * self.soundspeed(T)
        return np.hypot(v_therm, v_turb) / np.sqrt(2)

    def gaussian(self, x, x0, dx, A):
        """Gaussian function. dx is the standard deviation."""
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2))

    def samples2percentiles(self, samples):
        """Returns the [16th, 50th, 84th] percentiles of the samples."""
        return np.array([np.percentile(s, [16, 50, 84]) for s in samples.T])

    def samples2uncertainties(self, samples):
        """Returns the percentiles in a [<y>, -dy, +dy] format."""
        pcnts = self.samples2percentiles(samples)
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts])
