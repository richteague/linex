"""
Class to fit slab models with a single temperature and density. Assume that
the excitation conditions are {Tkin, nH2, NCS, M}. In addition, each line has
the parameters {x0_i, var_i, vcorr_i} which specify the line centre, the
variance of the noise and the correlation of the noise. Thus for N lines we
have 4 + 3N free parameters.

Is there a way to implement this for N lines? Or do I have to hardcode it all?

Input:
    dic - slab dictionary of spectra from slabclass.py.
    gridpath - path to the pre-calculated intensities.

"""

import time
import emcee
import george
import numpy as np
import scipy.constants as sc
from linex.radexgridclass import radexgrid
from george.kernels import ExpSquaredKernel as ExpSq
import matplotlib.pyplot as plt


class fitdict:

    B0 = 2.449913e10
    nu = {2: 146.9690287e9,
          4: 244.9355565e9,
          6: 342.8828503}
    gu = {2: 7.,
          4: 11.,
          6: 15.}

    def __init__(self, dic, gridpath, **kwargs):
        """
        Fit the slab models.
        """
        self.dic = dic
        self.grid = radexgrid(gridpath)
        self.trans = sorted([k for k in self.dic.keys()])
        self.ntrans = len(self.trans)
        self.peaks = [self.dic[k].Tb for k in self.trans]
        self.widths = [self.dic[k].dx for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.mu = self.dic[self.trans[0]].mu

        # Estimate the noise from the channels far from the line centre. Note
        # that these values are only used as starting values for the fitting.
        self.rms = [np.nanstd(self.spectra[k][abs(self.velaxs[k]) > 0.5])
                    for k in range(len(self.trans))]
        self.rms = np.array(self.rms)
        print("Estimated RMS of each line:")
        for j, sig in zip(self.trans, self.rms):
            print("  J = %d - %d: %.1f mK" % (j+1, j, 1e3 * sig))

        # Plot the read in lines to check everything looks OK.
        if kwargs.get('plot', False):
            self.plotlines()
        return

    def _lnprob(self, theta, fluxcal=0.0):
        """
        Log-probability function.
        """
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta, fluxcal)

    def _lnprior(self, theta):
        """
        Log-prior function.
        """
        temp, dens, sigma, mach, x0s, sig2, corr = self._parse(theta)

        # Excitation conditions.
        if not self.grid.in_grid(temp, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dens, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf
        if not 0.0 <= mach < 0.5:
            return -np.inf

        # Line profiles and noise.
        if not all([x > v[0] or x < v[-1] for x, v in zip(x0s, self.velaxs)]):
            return -np.inf
        if not all([0. < s < 0.1 for s in sig2]):
            return -np.inf
        if not all([-15. < c < 0. for c in corr]):
            return -np.inf

        return 0.0

    def _parse(self, t):
        """Parses theta values given self.params."""
        idxs = 4 + np.arange(self.ntrans) * 3
        x0s = [t[i] for i in idxs]
        idxs += 1
        sig = [t[i] for i in idxs]
        idxs += 1
        cor = [t[i] for i in idxs]
        return t[0], t[1], t[2], t[3], x0s, sig, cor

    def _lnlike(self, theta, fluxcal=0.0):
        """
        Log-likelihood for fitting peak brightness.
        """
        t, d, s, m, x0s, sig2, corr = self._parse(theta)
        toiter = zip(self.trans, self.velaxs, x0s)
        models = [self._spectrum(j, t, d, s, v, x, m) for j, v, x in toiter]
        models = [mod * (1. + fluxcal * np.random.randn()) for mod in models]
        toiter = zip(sig2, corr, self.spectra)
        noises = [george.GP(s * ExpSq(10**c)) for sig, c, y in toiter]
        for k, velax, dy in zip(noises, self.velaxs, self.rms):
            k.compute(velax, dy)
        lnx2 = 0.0
        for k, mod, obs in zip(noises, models, self.spectra):
            lnx2 += k.lnlikelihood(mod - obs)
        return lnx2

    def _spectrum(self, j, t, d, s, x, x0, mach):
        """
        Returns a spectrum on the provided velocity axis.
        """
        dV = self.linewidth(t, mach)
        A = self.grid.intensity(j, dV, t, d, s)
        return self.gaussian(x, x0, dV, A)

    def emcee(self, **kwargs):
        """
        Run emcee.
        """

        # Default values for the MCMC fitting.
        # Timing values:
        #   3 lines: 200 walkers, 500 steps takes:  2m34s.
        #   3 lines: 500 walkers, 500 steps takes:  8m34s.
        #   3 lines: 200 walkers, 1000 steps takes: 7m49s.

        nwalkers = kwargs.get('nwalkers', 200)
        nburnin = kwargs.get('nburnin', 500)
        nsteps = kwargs.get('nsteps', 500)

        # Set up the parameters and the sampler. Should allow for any multiple
        # of spectra to be simultaneously fit.

        self.params = ['temp', 'dens', 'sigma', 'mach']
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i, 'sig_%d' % i, 'vcorr_%d' % i]
        ndim = len(self.params)

        # For the sampling, we first sample the whole parameter space. After
        # the burn-in phase, we recenter the walkers around the median value in
        # each parameter and start from there. For noise parameters we sample
        # around the estimated RMS value and assume a correlation ranging
        # between 1 m/s and 100 m/s.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        pos = [self.grid.random_samples(p.split('_')[0], nwalkers)
               for p in self.params if p.split('_')[0] in self.grid.parameters]
        pos += [np.random.uniform(0.0, 0.5, nwalkers)]
        for i in range(len(self.trans)):
            pos += [np.zeros(nwalkers) + 1e-2 * np.random.randn(nwalkers)]
            pos += [self.rms[i]**2 + 1e-4 * np.random.randn(nwalkers)]
            pos += [np.random.uniform(-3, -1, nwalkers)]

        # Run the two burn-in periods and then one major one.

        print("Running first burn-in...")
        pos, lp, _ = sampler.run_mcmc(np.squeeze(pos).T, nburnin)
        pos = pos[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler.reset()

        print("Running second burn-in...")
        pos, _, _ = sampler.run_mcmc(pos, nburnin)

        print("Running productions...")
        t0 = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob,
                                        args=[kwargs.get('fluxcal', 0.0)])
        pos, _, _ = sampler.run_mcmc(pos, nsteps)
        print("Production complete in %s." % self.hmsformat(time.time()-t0))

        # Plot the samples if requested.

        if kwargs.get('plotsamples', True):
            self.plotsampling(sampler, kwargs.get('color', 'dodgerblue'))
        return sampler, sampler.flatchain

    # General Functions.

    def fluxcal_uncertainties(self, fluxcal, samples):
        """
        Calculates the uncertainties from the flux calibration.
        """
        dT = self.fluxcal_temperature(fluxcal, np.median(samples[0]))
        dN = self.fluxcal_columndensity(fluxcal)
        return dT, dN

    def fluxcal_temperature(self, fluxcal, T):
        """
        Estimate the error on temperature from flux calibration uncertainties.
        """
        dT = sc.k / sc.h / self.B0 / self.partition_function(T)
        dT = [dT - sc.h * self.nu[j] / sc.k / T**2 for j in self.trans]
        return fluxcal / max(dT)

    def fluxcal_columndensity(self, fluxcal):
        """
        Estimate the error on the log column density from flux calibration.
        """
        return 0.434 * fluxcal

    def partition_function(self, T):
        """
        Partition function for CS. Values from JPL.
        """
        return sc.k * T / sc.h / self.B0 + 1. / 3.

    def plotlines(self, ax=None):
        """
        Plot the emission lines which are to be fit.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for i, t in enumerate(self.trans):
            l = ax.plot(self.velaxs[i], self.spectra[i], lw=1.25,
                        label=r'J = %d - %d' % (t+1, t))
            ax.fill_between(self.velaxs[i],
                            self.spectra[i] - 3.0 * self.rms[i],
                            self.spectra[i] + 3.0 * self.rms[i],
                            lw=0.0, alpha=0.2, color=l[0].get_color(),
                            zorder=-3)
        ax.set_xlabel(r'${\rm Velocity \quad (km s^{-1})}$')
        ax.set_ylabel(r'${\rm Brightness \quad (K)}$')
        ax.legend(frameon=False, markerfirst=False)
        return

    def plotsampling(self, sampler, color='dodgerblue'):
        """Plot the sampling."""

        # Loop through each of the parameters and plot their sampling.
        for param, samples in zip(self.params, sampler.chain.T):
            fig, ax = plt.subplots()

            # Plot the individual walkers.
            for walker in samples.T:
                ax.plot(walker, alpha=0.075, color='k')

            # Plot the percentiles.
            l, m, h = np.percentile(samples, [16, 50, 84], axis=1)
            mm = np.mean(m)
            ax.axhline(mm, color='w', ls='--')
            ax.plot(l, color=color, lw=1.0)
            ax.plot(m, color=color, lw=1.0)
            ax.plot(h, color=color, lw=0.5)

            # Rescale the axes to make everything visible.
            ll = mm - l[-1]
            hh = mm - h[-1]
            yy = max(ll, hh)
            ax.set_ylim(mm - 3.5 * yy, mm + 3.5 * yy)
            ax.set_xlim(0, m.size)

            # Axis labels.
            ax.set_xlabel(r'$N_{\rm steps}$')
            ax.set_ylabel(r'${\rm %s}$' % param)
        return

    def linewidth_uncertainty(self, samples):
        """
        Returns the linewidth uncertaity [km/s].
        """
        unc = self.samples2uncertainties(samples)
        T, dT = unc[0, 0], np.average(unc[0, 1:])
        M, dM = unc[3, 0], np.average(unc[3, 1:])
        dV = self.linewidth(T, M) * np.sqrt(2) * 1e3
        cs = self.soundspeed(T)
        return ((dT * dV / 2. / T) + (dM * M * cs**2 / dV)) / 1e3

    def thermalwidth(self, T):
        """Thermal Doppler width [m/s]."""
        return np.sqrt(2. * sc.k * T / self.mu / sc.m_p)

    def soundspeed(self, T):
        """Soundspeed of gas [m/s]."""
        return np.sqrt(sc.k * T / 2.34 / sc.m_p)

    def linewidth(self, T, Mach):
        """Standard deviation of line, dV [km/s]."""
        v_therm = self.thermalwidth(T)
        v_turb = Mach * self.soundspeed(T)
        return np.hypot(v_therm, v_turb) / np.sqrt(2) / 1e3

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

    def hmsformat(self, time):
        """Returns nicely formatted time."""
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
