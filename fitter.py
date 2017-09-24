"""
Class to fit spectra with a Gaussian line profile. It allows the simultaneous
fitting of multiple transitions, accounts for both non-LTE and LTE excitation,
the inclusion of non-thermal broadening components and the option to model the
noise with Gaussian Processes with george.

-- Input:

    dic - A dictionary containing the spectra to fit. The keys are the lower
        energy level of the transition. Each dictionary must contain estimates
        of the peak brightness, line centre and line width.

    grid - Path to the pre-calculated line brightnesses from RADEX.

-- Optional Input:

    verbose [bool]      - Writes messages of what is going on.
    diagnostics [bool]  - Plots the sampling, best fit models and corner plots.

    GP [bool]           - Model the noise with Gaussian proccesses with Geroge.
                          This will vastly slow down the computation.
    laminar [bool]      - Force non-thermal broadening to be zero.
    singlemach [bool]   - All lines share the same non-thermal broadening.
    logmach [bool]      - Use logarithmic sampling for non-thermal broadening.
    lte [bool]          - Assume LTE (assume this is the largest density in the
                          attached RADEX grid).
    vbeam [float]       - Additional non-thermal term to account for beamsize.
    vdeproj [float]     - Broadening factor induced by the azimuthal averaging.
                          It must be equal to or larger than 1.
    tdeproj [float]     - Rescale the model spectra by this factor to account
                          for the loss in peak from azimuthal averaging.
    thick [bool]        - Use a line profile assuming tau is Gaussian. Very
                          much work in progress!

The MCMC is set up to run relatively well with the default settings. Burn-in is
performed in two steps, the first starts from random positions allowed by the
attached grid running for 500 steps. The walkers are restarted with a small
scatter around the median position of the walkers. This should be closer to the
correct value and allow for the removal of walkers trapped in bad locations.
This then runs for an additional 1000 steps. The last 500 of these are taken as
the posterior and used to plot the corner plot.

-- emcee Input:

    nwalkers [int]      - Number of walkers. More walkers is much, much better
                          than more steps.
    nburnin1 [int]      - Number of steps for the intial stage of burn-in.
    nburnin2 [int]      - Number of steps for the second stage of burn-in.
    nsteps [int]        - Number of steps used for the posterior sampling.
    p0 [list]           - Starting positions for the emcee. If not specified
                          then random positions across the entire grid will be
                          chosen. While this allows one to explore the whole
                          space, it may result in the global minima not being
                          found.
-- TODO:

    1) Update this for a single spectral axis but multiple components.

"""

import time
import emcee
import george
import numpy as np
import scipy.constants as sc
from linex.radexgridclass import radexgrid
from george.kernels import ExpSquaredKernel as ExpSq
from plotting import plotsampling
from plotting import plotcorner
from plotting import plotobservations
from plotting import plotbestfit


class fitdict:

    def __init__(self, dic, grid, **kwargs):
        """Fit the slab models."""

        self.dic = dic
        self.grid = radexgrid(grid)
        self.verbose = kwargs.get('verbose', True)

        # Read in the observations and their best fit values.
        # TODO: There must be a more efficient way of doing this...

        self.trans = sorted([k for k in self.dic.keys()])
        self.trans = np.array(self.trans)
        self.ntrans = self.trans.size
        self.Tbs = [self.dic[k].Tb for k in self.trans]
        self.dxs = [self.dic[k].dx for k in self.trans]
        self.x0s = [self.dic[k].x0 for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.mu = self.dic[self.trans[0]].mu

        # Set up the type of density we will model.

        self.lte = kwargs.get('lte', kwargs.get('LTE', False))

        # Control the type of turbulence we want to model.

        self.laminar = kwargs.get('laminar', False)
        self.logmach = kwargs.get('logmach', False)
        self.singlemach = kwargs.get('singlemach', True)
        if self.logmach and self.laminar:
            self.logmach = False
        if self.laminar:
            self.singlemach = True

        # Include addtional terms to model the observational effects.
        # With these variables, check to see if they're iterable, otherwise
        # extend them into a list with length self.ntrans.

        self.vbeam = self._makeiterable('vbeam', 0.0, **kwargs)
        self.vdeproj = self._makeiterable('vdeproj', 1.0, **kwargs)
        self.tdeproj = self._makeiterable('tdeproj', 1.0, **kwargs)

        self.thick = kwargs.get('thick', False)
        if self.thick and not self.grid.hastau:
            raise ValueError("Attached grid does not have tau values.")

        # Set up the noise, estimating it in the spectrum and controlling the
        # noise model if requested.

        self.rms = self._estimatenoise()
        self.GP = kwargs.get('GP', kwargs.get('gp', False))

        # Populate the parameters to fit.

        self._popparams()

        # Output messages.

        if self.verbose:
            print("\n")
            print("Estimated RMS of each line:")
            for j, sig in zip(self.trans, self.rms):
                print("  J = %d - %d: %.1f mK" % (j+1, j, 1e3 * sig))
            if any(self.vbeam > 0.0):
                print("Including a beam broadening term.")
            if any(self.vdeproj != 1.0):
                print("Including deprojection broadening term.")
            if any(self.tdeproj != 1.0):
                print("Including a brightness temperature scaling factor.")
            if self.laminar:
                print("Assuming only thermal broadening.")
            if self.singlemach:
                print("Fitting individual non-thermal width components.")
            if self.logmach:
                print("Fitting for log-Mach.")
            if self.lte:
                print("Assuming LTE.")
            if self.GP:
                print("Using Gaussian processes to model noise.")
            if self.thick:
                print("Including opacity in line profile calculation.")
            print("\n")
        self.diagnostics = kwargs.get('diagnostics', True)

        return

    def _makeiterable(self, name, default, **kwargs):
        """Return a kwarg argument that's iterable."""
        val = np.array([kwargs.get(name, default)]).flatten()
        if len(val) == self.ntrans:
            return np.squeeze(val)
        elif len(val) == 1:
            return np.squeeze([val for _ in range(self.ntrans)])
        else:
            raise ValueError("Must be single or one for each spectrum.")

    def _estimatenoise(self):
        """Estimate the noise in each spectra."""
        rms = []
        for k in range(self.ntrans):
            mask = abs(self.velaxs[k] - self.x0s[k]) > 0.5
            rms += [np.nanstd(self.spectra[k][mask])]
        return np.array(rms)

    def _popparams(self):
        """Populates the free variables and the number of dimensions."""
        if self.lte:
            self.params = ['temp', 'sigma']
        else:
            self.params = ['temp', 'dens', 'sigma']
        if self.singlemach:
            self.params += ['mach']
        else:
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
            if not all([-5.0 < m < 0.0 for m in mach]):
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
        vbeam = self.vbeam[abs(self.trans - j).argmin()] / self.soundspeed(t)
        if self.logmach:
            dV_int = self.linewidth(t, np.power(10, mach))
            dV_obs = self.linewidth(t, np.power(10, mach) + vbeam)
        else:
            dV_int = self.linewidth(t, mach)
            dV_obs = self.linewidth(t, mach + vbeam)
        Tb = self.grid.intensity(j, dV_int, t, d, s)
        dV_obs *= self.vdeproj[abs(self.trans - j).argmin()]
        Tb *= self.tdeproj[abs(self.trans - j).argmin()]
        if not self.thick:
            return self.gaussian(x, x0, dV_obs, Tb)
        tau = self.grid.tau(j, dV_int, t, d, s)
        return self.thickline(x, x0, dV_obs, Tb, tau)

    def thickline(self, x, x0, dx, Tb, tau):
        """Returns an optically thick line profile."""
        return Tb * (1. - np.exp(-self.gaussian(x, x0, dx, tau)))

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
        nburnin1 = kwargs.get('nburnin1', 500)
        nburnin2 = kwargs.get('nburnin2', nburnin1)
        nsteps = kwargs.get('nsteps', 500)
        p0 = kwargs.get('p0', None)

        # For the initial positions, unless they are provided through the p0
        # kwarg, run a burn-in cycle with positions starting randomly within
        # the entire grid.

        t0 = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)

        if p0 is None:
            pos = self._startingpositions(nwalkers)
            if self.verbose:
                print("Finding p0...")
            pos, lp, _ = sampler.run_mcmc(np.squeeze(pos).T, nburnin1)
            pos = pos[np.argmax(lp)]
            if self.diagnostics:
                plotsampling(sampler, self.params, title='Estimating p0')
            sampler.reset()
        else:
            pos = p0
            if len(pos) != len(self.params):
                raise ValueError("Wrong number of starting positions.")

        pos = pos + 1e-4 * np.random.randn(nwalkers, self.ndim)
        if self.verbose:
            print("Running burn-in...")
        pos, _, _ = sampler.run_mcmc(pos, nburnin2)
        if self.diagnostics:
            plotsampling(sampler, self.params, title='Burn-In')

        if self.verbose:
            print("Running productions...")
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)
        pos, _, _ = sampler.run_mcmc(pos, nsteps)

        if self.diagnostics:
            plotsampling(sampler, self.params, title='Production')
            ax = plotobservations(self.trans, self.velaxs,
                                  self.spectra, self.rms)
            plotbestfit(self.trans, self.velaxs,
                        self.bestfitmodels(sampler), ax=ax)
            plotcorner(sampler, self.params)
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

    def bestfitmodels(self, sampler):
        """Return the best-fit models."""
        return self._calculatemodels(self._calculatetheta(sampler))

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
