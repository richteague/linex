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
import corner
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
        """Fit the slab models."""

        self.dic = dic
        self.grid = radexgrid(gridpath)

        # If verbose, print what's going on. If diagnostics is true, plot
        # several diagnostic plots: burn-in plots, samples, posterior corner
        # plots and best-fit models overlaid on observations.

        self.verbose = kwargs.get('verbose', True)
        self.diagnostics = kwargs.get('diagnostics', True)

        # Read in the models and their simple-fit values for estimating noise.
        # Everything should be ordered in the lower energy transitions first.

        self.trans = sorted([k for k in self.dic.keys()])
        self.ntrans = len(self.trans)
        self.Tbs = [self.dic[k].Tb for k in self.trans]
        self.dxs = [self.dic[k].dx for k in self.trans]
        self.x0s = [self.dic[k].x0 for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.mu = self.dic[self.trans[0]].mu

        # Control the type of turbulence we want to model.
        # If laminar, don't assume any turbulent broadening.
        # If logmach, fit log(M) rather than M. This is better for limits.
        # If singlemach, assume a common non-thermal broadening term.
        # If vbeam > 0.0, include an additional broadening term to the profile.

        self.laminar = kwargs.get('laminar', False)
        if self.laminar and self.verbose:
            print("Assuming only thermal broadening.")

        self.logmach = kwargs.get('logmach', False)
        if self.logmach and self.verbose:
            print("Fitting for log-Mach.")

        if self.logmach and self.laminar:
            self.logmach = False
            if self.verbose:
                print("Model is laminar, reverting from log-Mach.")

        self.vbeam = kwargs.get('vbeam', 0.0)
        if self.vbeam > 0.0 and self.verbose:
            print("Assuming v_beam = %.0f m/s." % self.vbeam)
            print("Change this at any time through 'self.vbeam'.")

        self.singlemach = kwargs.get('singlemach', True)
        if self.laminar:
            self.singlemach = True
        if not self.singlemach and self.verbose:
            print("Allowing individual non-thermal components.")

        self.lte = kwargs.get('lte', kwargs.get('LTE', False))
        if self.lte and self.verbose:
            print("Assuming LTE.")

        # Estimate the noise from the channels far from the line centre. Note
        # that these values are only used as starting values for the fitting.

        self.rms = self._estimatenoise()

        # Check if we want to include Gaussian processes to model the noise.
        # If we don't use it, the fitting is much quicker as we just use a more
        # simple chi-squared approach.

        self.GP = kwargs.get('GP', False)
        if self.GP and self.verbose:
                print("Using Gaussian processes to model noise.")

        # Writ a list of the free parameters which we will fit.
        self._popparams()

        return

    def _estimatenoise(self):
        """Estimate the noise in each spectra."""
        rms = []
        for k in range(self.ntrans):
            mask = abs(self.velaxs[k] - self.x0s[k]) > 0.5
            rms += [np.nanstd(self.spectra[k][mask])]
        rms = np.array(rms)
        if self.verbose:
            print("Estimated RMS of each line:")
            for j, sig in zip(self.trans, rms):
                print("  J = %d - %d: %.1f mK" % (j+1, j, 1e3 * sig))
        return rms

    def _popparams(self):
        """Populates the free variables and the number of dimensions."""
        if self.lte:
            self.params = ['temp', 'sigma']
        else:
            self.params = ['temp', 'dens', 'sigma']
        if self.singlemach:
            self.params += ['mach']
        else:
            if self.verbose:
                print("Fitting individual non-thermal width components.")
            for i in range(len(self.trans)):
                self.params += ['mach_%d' % i]
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i]
            if self.GP:
                self.params += ['sig_%d' % i, 'vcorr_%d' % i]
        if self.laminar:
            self.params = [p for p in self.params if 'mach' not in p]
        self.ndim = len(self.params)
        return

    def _parse(self, theta):
        """Parses theta into {temp, dens, sigma, mach, x0s, sigs, corrs}."""
        zipped = zip(theta, self.params)
        temp = theta[0]
        if self.lte:
            dens = self.grid.maxdens
            sigma = theta[1]
        else:
            dens = theta[1]
            sigma = theta[2]
        if not self.laminar:
            mach = [t for t, n in zipped if 'mach' in n]
        else:
            mach = [0.0]
        x0s = [t for t, n in zipped if 'x0_' in n]
        if self.GP:
            sigs = [t for t, n in zipped if 'sig_' in n]
            corrs = [t for t, n in zipped if 'vcorr_' in n]
        else:
            sigs = [np.nan for _ in range(self.ntrans)]
            corrs = [np.nan for _ in range(self.ntrans)]
        return temp, dens, sigma, mach, x0s, sigs, corrs

    # Fitting with standard chi-squared likelihood function.

    def _lnprob(self, theta):
        """Log-probability function with simple chi-squared likelihood."""
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta)

    def _lnprior(self, theta):
        """Log-prior function. Uninformative priors."""
        temp, dens, sigma, mach, x0s, sigs, corrs = self._parse(theta)

        # Excitation parameters.
        if not self.grid.in_grid(temp, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dens, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf

        # Non-thermal broadening.
        if self.logmach:
            if not all([-5.0 < m < -0.3 for m in mach]):
                return -np.inf
        else:
            if not all([0.0 <= m < 0.5 for m in mach]):
                return -np.inf

        # Line centres.
        if not all([min(v) < x < max(v) for x, v in zip(x0s, self.velaxs)]):
            return -np.inf

        # Noise properties.
        if self.GP:
            if not all([0. < s < 0.1 for s in sigs]):
                return -np.inf
            if not all([-15. < c < 0. for c in corrs]):
                return -np.inf
        return 0.0

    def _calculatemodels(self, theta):
        """Calculates the appropriate models."""
        t, d, s, m, x, _, _ = self._parse(theta)
        return [self._spectrum(j, t, d, s, v, x[i], m[i % len(m)])
                for i, (j, v) in enumerate(zip(self.trans, self.velaxs))]

    def _spectrum(self, j, t, d, s, x, x0, mach):
        """Returns a spectrum on the provided velocity axis."""
        vbeam = self.vbeam / self.soundspeed(t)
        if self.logmach:
            dV_int = self.linewidth(t, np.power(10, mach))
            dV_obs = self.linewidth(t, np.power(10, mach) + vbeam)
        else:
            dV_int = self.linewidth(t, mach)
            dV_obs = self.linewidth(t, mach + vbeam)
        A = self.grid.intensity(j, dV_int, t, d, s)
        return self.gaussian(x, x0, dV_obs, A)

    def _lnlike(self, theta):
        """Log-likelihood with chi-squared likelihood function."""
        models = self._calculatemodels(theta)
        if not self.GP:
            return self._chisquared(models)
        noises = self._calculatenoises(theta)
        for k, x, dy in zip(noises, self.velaxs, self.rms):
            k.compute(x, dy)
        lnx2 = 0.0
        for k, mod, obs in zip(noises, models, self.spectra):
            lnx2 += k.lnlikelihood(mod - obs)
        return lnx2

    def _calculatenoises(self, theta):
        """Return the noise models."""
        _, _, _, _, _, sigs, corrs = self._parse(theta)
        return [george.GP(s**2 * ExpSq(10**c)) for s, c in zip(sigs, corrs)]

    def _chisquared(self, models):
        """Chi-squared likelihoo function."""
        lnx2 = []
        for o, dy, y in zip(self.spectra, self.rms, models):
            dlnx2 = ((o - y) / dy)**2 + np.log(dy**2 * np.sqrt(2. * np.pi))
            lnx2.append(-0.5 * np.nansum(dlnx2))
        return np.nansum(lnx2)

    def _startingpositions(self, nwalkers):
        """Return random starting positions."""

        # Excitation conditions.
        pos = [self.grid.random_samples('temp', nwalkers)]
        if not self.lte:
            pos += [self.grid.random_samples('dens', nwalkers)]
        pos += [self.grid.random_samples('sigma', nwalkers)]

        # Non-thermal broadening.
        if not self.laminar:
            if self.logmach:
                pos += [np.random.uniform(-5, -1, nwalkers)
                        for p in self.params if 'mach' in p]
            else:
                pos += [np.random.uniform(0.0, 0.5, nwalkers)
                        for p in self.params if 'mach' in p]

        # Individual line properties.
        for i in range(self.ntrans):
            pos += [self.x0s[i] + 1e-2 * np.random.randn(nwalkers)]
            if self.GP:
                pos += [self.rms[i]**2 + 1e-4 * np.random.randn(nwalkers)]
                pos += [np.random.uniform(-3, -1, nwalkers)]
        return pos

    def emcee(self, **kwargs):
        """Run emcee."""

        nwalkers = kwargs.get('nwalkers', 200)
        nburnin = kwargs.get('nburnin', 500)
        nsteps = kwargs.get('nsteps', 500)

        # For the sampling, we first sample the whole parameter space. After
        # the burn-in phase, we recenter the walkers around the median value in
        # each parameter and start from there. For noise parameters we sample
        # around the estimated RMS value and assume a correlation ranging
        # between 1 m/s and 100 m/s.

        pos = self._startingpositions(nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)

        t0 = time.time()

        if self.verbose:
            print("Running first burn-in...")
        pos, lp, _ = sampler.run_mcmc(np.squeeze(pos).T, nburnin)
        pos = pos[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, self.ndim)
        if self.diagnostics:
            self.plotsampling(sampler, title='First Burn-In')
        sampler.reset()

        if self.verbose:
            print("Running second burn-in...")
        pos, _, _ = sampler.run_mcmc(pos, nburnin)
        if self.diagnostics:
            self.plotsampling(sampler, title='Second Burn-In')

        if self.verbose:
            print("Running productions...")
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)
        pos, _, _ = sampler.run_mcmc(pos, nsteps)

        if self.diagnostics:
            self.plotsampling(sampler, title='Production')
            ax = self.plotobservations()
            self.plotbestfit(sampler, ax=ax)
            self.plotcorner(sampler)
        if self.verbose:
            t = self.hmsformat(time.time()-t0)
            print("Production complete in %s." % t)

        return sampler, sampler.flatchain

    def _calculatetheta(self, sampler, method='median'):
        """Returns the best fit parameters given the provided method."""
        if method.lower() == 'median':
            return np.median(sampler.flatchain, axis=0)
        elif method.lower() == 'mean':
            return np.mean(sampler.flatchain, axis=0)
        else:
            raise ValueError("Method must be 'median' or 'mean'.")

    def plotbestfit(self, sampler, ax=None):
        """Plot the best-fit spectra."""
        if ax is None:
            fig, ax = plt.subplots()
        models = self._calculatemodels(self._calculatetheta(sampler))
        for x, y, J in zip(self.velaxs, models, self.trans):
            ax.scatter(x, y, color='r')
        ax.legend(fontsize=6)
        return ax

    def plotobservations(self, ax=None):
        """Plot the emission lines which are to be fit."""
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
        return ax

    def plotsampling(self, sampler, color='dodgerblue', title=None):
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
            if title is not None:
                ax.set_title(title)
        return

    def plotcorner(self, sampler):
        """Plot the corner plot."""
        corner.corner(sampler.flatchain, labels=self.params,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True)
        return

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
