"""
Class to fit slab models with a single temperature and density. Assume that
the free parameers are {Tkin, nH2, NCS, M, x0, x1, x2}. We can account for flux
calibration uncertainties and different noises for each spectral line. In the
disk model version we do not account for correlated noise though.

Input:
    dic - dictionary of spectra.
    gridpath - path to the pre-calculated intensities.
    rms - list of the RMS values for the spectra in [K].
    fluxcal - list of the flux calibration uncertainties in [%].
"""

import emcee
import numpy as np
import scipy.constants as sc
from linex.radexgridclass import radexgrid
import matplotlib.pyplot as plt


class fitdict:

    def __init__(self, dic, gridpath, rms=None, fluxcal=0.0, **kwargs):
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
        self.verbose = False
        self.centers = [vel[spec.argmax()] for
                        vel, spec in zip(self.velaxs, self.spectra)]
        try:
            iter(fluxcal)
            self.fluxcal = fluxcal
        except TypeError:
            self.fluxcal = [fluxcal for p in self.peaks]
        print self.fluxcal
        self.applyfc = any(self.fluxcal)
        self.fluxcal = [fc if fc is not None else fc for fc in self.fluxcal]
        self.fluxcal = np.squeeze(self.fluxcal)
        if kwargs.get('plot', False):
            self.plotlines()
            self.verbose = True
        self.taulim = kwargs.get('taulim', False)
        return

    # MCMC Functions.

    def _lnprob(self, theta):
        """Log-probability function."""
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta)

    def _lnx2(self, modelspectra):
        """Log-Chi-squared function."""
        lnx2 = []
        for i, mod in enumerate(modelspectra):
            s = self.spectra[i]
            dy = np.hypot(s * self.fluxcal[i], self.rms[i])
            dlnx2 = ((s - mod) / dy)**2 + np.log(dy**2 * np.sqrt(2. * np.pi))
            lnx2.append(-0.5 * np.nansum(dlnx2))
        return np.nansum(lnx2)

    def _lnprior(self, theta):
        """
        Log-prior function. Temperature, density and column density must be
        in the RADEX grid. Line centre must be on the velocity axis. The Mach
        number can be any positive value less than one. The range of widths
        allowed by the radexgrid should cover everything up to M = 1.

        Include an optional check on the optical depth of the line. Set this
        through the 'taulim' kwarg, any lines with optical depths higher than
        this is disaollowed.
        """
        temp, dens, sigma, mach, x0s = self._parse(theta)
        if not self.grid.in_grid(temp, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dens, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf
        for x0, center in zip(x0s, self.centers):
            if abs(x0 - center) > 1.0:
                return -np.inf
        if not 0 <= mach <= 1:
            return -np.inf
        if self.taulim:
            t = [self.grid.tau(j, 0.2, temp, dens, sigma) for j in self.trans]
            if any([tau > self.taulim for tau in t]):
                return -np.inf
        return 0.0

    def _parse(self, theta):
        """Parses theta values given self.params."""
        x0s = [theta[j] for j in -(np.arange(len(self.trans)) + 1)[::-1]]
        return theta[0], theta[1], theta[2], theta[3], x0s

    def _lnlike(self, theta):
        """Log-likelihood for fitting peak brightness."""
        temp, dens, sigma, mach, x0s = self._parse(theta)
        toiter = zip(self.trans, self.velaxs, x0s)
        spectra = [self._spectrum(j, temp, dens, sigma, v, x0, mach)
                   for j, v, x0 in toiter]
        return self._lnx2(self.addfluxcal(spectra))

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

        # Set up the parameters and the sampler.
        self.params = ['temp', 'dens', 'sigma', 'mach']
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i]
        ndim = len(self.params)

        # For the sampling, we first sample the whole parameter space. After
        # the burn-in phase, we recenter the walkers around the median value in
        # each parameter and start from there.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        pos = [self.grid.random_samples(p, nwalkers) for p in self.params
               if p in self.grid.parameters]
        pos += [np.random.uniform(0.0, 0.3, nwalkers)]
        for i in range(len(self.trans)):
            pos += [0.3 * np.random.randn(nwalkers)]

        print("Running first burn-in...")
        sampler.run_mcmc(np.squeeze(pos).T, nburnin)

        if self.verbose:
            for param in sampler.chain.T:
                fig, ax = plt.subplots()
                for walker in param.T:
                    ax.plot(walker, alpha=0.1, c='k')

        pos = np.median(sampler.flatchain[-int(0.2*nburnin):], axis=0)

        if self.verbose:
            print pos

        pos = pos[None, :] + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler.reset()

        print("Running second burn-in...")
        pos, _, _ = sampler.run_mcmc(pos, nburnin+nsteps)

        # We return the full sampler and the flattened chains with the burn-in
        # period removed. The former is useful to check if the burn-in was
        # sufficient, while the latter is useful for quickly plotting corner
        # plots and the like.
        return sampler, sampler.chain[:, nburnin:].reshape((-1, ndim))

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

    def addfluxcal(self, spectra):
        """Add flux calibration uncertainty to the spectra."""
        if self.applyfc:
            return spectra
        toiter = zip(spectra, self.fluxcal)
        fcspec = [s * (np.random.randn() * fc + 1.) for s, fc in toiter]
        return np.squeeze(fcspec)

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
