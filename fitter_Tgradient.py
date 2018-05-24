"""
Class to fit the Gaussian line profile by including a temperature gradient.
Assumes: LTE, an average over N slabs
"""

import time
import emcee
import celerite
import numpy as np
import scipy.constants as sc
from linex.radexgridclass import radexgrid
from celerite.terms import Matern32Term as M32
from plotting import plotsampling
from plotting import plotcorner
from plotting import plotobservations
from plotting import plotbestfit

linefreq = {2: 146.9690287e9, 4: 244.9355565e9, 6: 342.8828503e9}


class fitdict_gradient:

    def __init__(self, dic, grid, **kwargs):
        """Initialise the class."""

        self.dic = dic
        self.grid = radexgrid(grid)
        self.verbose = kwargs.get('verbose', True)
        self.diagnostics = kwargs.get('diagnostics', True)

        # Read in the observations.

        self.trans = sorted([k for k in self.dic.keys()])
        self.trans = np.array(self.trans)
        self.ntrans = self.trans.size
        self.Tbs = [self.dic[k].Tb for k in self.trans]
        self.dxs = [self.dic[k].dx for k in self.trans]
        self.x0s = [self.dic[k].x0 for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.mu = self.dic[self.trans[0]].mu
        self.freq = [linefreq[J] for J in self.trans]
        self.rms = self._estimatenoise()

        # Define the model - very stripped down verion of linex.

        self.nslabs = int(kwargs.get('nslabs', 5))
        self.singlemach = kwargs.get('singlemach', True)
        self.nslabs = int(kwargs.get('nslabs', 100))
        self.oversample = kwargs.get('oversample', True)
        self.hanning = kwargs.get('hanning', True)
        self.thick = kwargs.get('thick', True)
        self.GP = kwargs.get('GP', kwargs.get('gp', True))

        # Populate the free parameters.

        self.params, self.ndim = self._populate_params()

    def emcee(self, **kwargs):
        """Run the MCMC sampler and return the posterior samples."""

        nwalkers = kwargs.get('nwalkers', 300)
        nburns = kwargs.get('nburns', 1)
        burnin_steps = kwargs.get('burnin_steps', 300)
        burnin_sample = kwargs.get('burnin_sample', 50)
        sample_steps = kwargs.get('sample_steps', 50)
        p0 = kwargs.get('p0', None)

        # For the initial positions, unless they are provided through the p0
        # kwarg, run a burn-in cycle with positions starting randomly within
        # the entire grid.

        t0 = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)

        # Initial burn in to find the parameters.
        if p0 is None:
            pos = self._startingpositions(nwalkers)
        else:
            pos = p0
            if len(pos) != len(self.params):
                raise ValueError("Wrong number of starting positions.")
            pos = (pos + 1e-4 * np.random.randn(nwalkers, self.ndim)).T

        # Run the number of burn-ins requested. After each burn-in, resample
        # the walkers around the percentiles of the last 20 samples.

        for burn in range(nburns):
            if self.verbose:
                print("Running burn-in part %d of %d..." % (burn + 1, nburns))
            _, _, _ = sampler.run_mcmc(np.squeeze(pos).T, burnin_steps)
            pos = self._resample_posterior(sampler, nwalkers, N=burnin_sample)
            if self.diagnostics:
                plotsampling(sampler, self.params,
                             title='Burn-In %d' % (burn + 1))
            sampler.reset()

        # Run the production samples.
        if self.verbose:
            print("Running productions...")
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob)
        pos, _, _ = sampler.run_mcmc(np.squeeze(pos).T, sample_steps)

        if self.diagnostics:
            plotsampling(sampler, self.params, title='Production')
            ax = plotobservations(self.trans, self.velaxs,
                                  self.spectra, self.rms)
            plotbestfit(self.trans, self.velaxs,
                        self.bestfitmodels(sampler), ax=ax)
            plotcorner(sampler, self.params)
        if self.verbose:
            t = self.hmsformat(time.time() - t0)
            print("Production complete in %s." % t)

        return sampler, sampler.flatchain

    def _estimatenoise(self):
        """Estimate the noise in each spectra."""
        rms = []
        for k in range(self.ntrans):
            mask = abs(self.velaxs[k] - self.x0s[k]) > 0.5
            rms += [np.nanstd(self.spectra[k][mask])]
        return np.array(rms)

    def _populate_params(self):
        """Returns the free variables and the number of dimensions."""
        params = ['Tmin', 'Tmax', 'sigma']
        if self.singlemach:
            params += ['mach']
        else:
            for i in range(len(self.trans)):
                params += ['mach_%d' % i]
        for i in range(len(self.trans)):
            params += ['x0_%d' % i]
            if self.GP:
                params += ['sig_%d' % i, 'vcorr_%d' % i]
        return params, len(params)

    def _parse(self, theta):
        """Parse theta into the correct free parameters."""
        zipped = zip(theta, self.params)
        Tmin = theta[self.params.index('Tmin')]
        Tmax = theta[self.params.index('Tmax')]
        sigma = theta[self.params.index('sigma')]
        if self.singlemach:
            mach = [theta[self.params.index('mach')]]
        else:
            mach = [t for t, n in zipped if 'mach_' in n]
        x0s = [t for t, n in zipped if 'x0_' in n]
        if self.GP:
            sigs = [t for t, n in zipped if 'sig_' in n]
            corrs = [t for t, n in zipped if 'vcorr_' in n]
        else:
            sigs = [np.nan for _ in range(self.ntrans)]
            corrs = [np.nan for _ in range(self.ntrans)]
        return Tmin, Tmax, sigma, mach, x0s, sigs, corrs

    def _lnlike(self, theta):
        """Log-likelihood with chi-squared likelihood function."""
        models = self._calculatemodels(theta)
        if not self.GP:
            return self._chisquared(models)
        lnx2 = 0.0
        _, _, _, _, _, sigs, corrs = self._parse(theta)
        for i in range(self.ntrans):
            rho = M32(log_sigma=sigs[i], log_rho=corrs[i])
            gp = celerite.GP(rho)
            gp.compute(self.velaxs[i], self.rms[i])
            lnx2 += gp.log_likelihood(models[i] - self.spectra[i])
        return np.nansum(lnx2)

    def _chisquared(self, models):
        """Chi-squared likelihoo function."""
        lnx2 = []
        for o, dy, y in zip(self.spectra, self.rms, models):
            dlnx2 = ((o - y) / dy)**2 + np.log(dy**2 * np.sqrt(2. * np.pi))
            lnx2.append(-0.5 * np.nansum(dlnx2))
        return np.nansum(lnx2)

    def _lnprob(self, theta):
        """Log-probability function."""
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta)

    def _lnprior(self, theta):
        """Log-prior function."""

        # Unpack the variables.
        Tmin, Tmax, sigma, mach, x0s, sigs, corrs = self._parse(theta)

        # Excitation parameters.
        if not self.grid.temp[0] <= Tmin < Tmax:
            return -np.inf
        if Tmax > self.grid.temp[-1]:
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf

        # Non-thermal broadening.
        if not all([-5.0 < m < 0.0 for m in mach]):
            return -np.inf

        # Line centres.
        if not all([min(v) < x < max(v) for x, v in zip(x0s, self.velaxs)]):
            return -np.inf

        # Noise properties.
        if self.GP:
            if not all([-5. < s < 2. for s in sigs]):
                return -np.inf
            if not all([-5. < c < 2. for c in corrs]):
                return -np.inf

        return 0.0

    def _spectrum(self, j, t, d, s, x, x0, mach, N=100):
        """Return a spectrum."""
        if self.oversample:
            xx = np.linspace(x[0], x[-1], x.size * N)
        else:
            xx = x
        j_idx = abs(self.trans - j).argmin()
        dV = self.linewidth(t, np.power(10, mach))
        if not self.thick:
            Tb = self.grid.intensity(j, dV, t, d, s)
            s = self.gaussian(xx, x0, dV, Tb)
        else:
            Tex = self.grid.Tex(j, dV, t, d, s)
            tau = self.grid.tau(j, dV, t, d, s)
            nu = self.freq[j_idx]
            s = self.thickline(xx, x0, dV, Tex, tau, nu)
        if self.oversample:
            s = np.interp(x, x, s[N/2::N])
        if self.hanning:
            s = np.convolve(s, [0.25, 0.5, 0.25], mode='same')
        return s

    def _calculatemodels(self, theta):
        """Calculate the slab models."""
        Tmin, Tmax, s, m, x, _, _ = self._parse(theta)
        spectra = []
        for T in np.linspace(Tmin, Tmax, self.nslabs):
            lines = []
            for i, (j, v) in enumerate(zip(self.trans, self.velaxs)):
                line = self._spectrum(j, T, 10, s, v, x[i], m[i % len(m)])
                lines += [line]
            spectra += [lines]
        return np.average(spectra, axis=0)

    def _sourcefunction(self, T, nu):
        """Source function of the line."""
        J = np.exp(sc.h * nu / sc.k / T) - 1.
        return sc.h * nu / sc.k / J

    def _calculateTb(self, Tex, nu, Tbg=2.73):
        """Calculate the brightness temperature from Tex."""
        Jbg = self._sourcefunction(Tbg, nu)
        hnk = sc.h * nu / sc.k
        Tb = hnk / (np.exp(hnk / Tex) - 1.)
        return Tb - Jbg

    def thickline(self, x, x0, dx, Tex, tau, nu):
        """Returns an optically thick line profile."""
        Tb = self._calculateTb(Tex, nu)
        return Tb * (1. - np.exp(-self.gaussian(x, x0, dx, tau)))

    def linewidth(self, T, Mach):
        """Standard deviation of line, dV [km/s]."""
        v_therm = self.thermalwidth(T)
        v_turb = Mach * self.soundspeed(T)
        return np.hypot(v_therm, v_turb) / np.sqrt(2) / 1e3

    def thermalwidth(self, T):
        """Thermal Doppler width [m/s]."""
        return np.sqrt(2. * sc.k * T / self.mu / sc.m_p)

    def soundspeed(self, T):
        """Soundspeed of gas [m/s]."""
        return np.sqrt(sc.k * T / 2.34 / sc.m_p)

    def samples2percentiles(self, samples):
        """Returns the [16th, 50th, 84th] percentiles of the samples."""
        return np.percentile(samples, [16, 50, 84], axis=0).T

    def samples2uncertainties(self, samples):
        """Returns the percentiles in a [<y>, -dy, +dy] format."""
        pcnts = self.samples2percentiles(samples)
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts])

    def hmsformat(self, time):
        """Returns nicely formatted time."""
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def random_from_percentiles(self, pcnts):
        """Draw from a distribution described by [16, 50, 84] percentiles."""
        rnd = np.random.randn()
        if rnd < 0.0:
            return pcnts[1] + rnd * (pcnts[1] - pcnts[0])
        else:
            return pcnts[1] + rnd * (pcnts[2] - pcnts[1])

    def _startingpositions(self, nwalkers):
        """Return random starting positions."""
        pos = []
        pos += [np.random.uniform(20, 30, nwalkers)]
        pos += [np.random.uniform(30, 40, nwalkers)]
        pos += [self.grid.random_samples('sigma', nwalkers)]
        pos += [np.random.uniform(-5, -1, nwalkers)
                for p in self.params if 'mach' in p]
        for i in range(self.ntrans):
            pos += [self.x0s[i] + 1e-2 * np.random.randn(nwalkers)]
            if self.GP:
                pos += [np.random.uniform(-3, -2, nwalkers)]
                pos += [np.random.uniform(-3, -2, nwalkers)]
        return pos

    def gaussian(self, x, x0, dx, A):
        """Gaussian function. dx is the standard deviation."""
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2))

    def _getpercentiles(self, sampler, N=50):
        """Returns the perncentiles of the final N steps."""
        samples = sampler.chain[:, -50:]
        samples = samples.reshape((-1, samples.shape[-1]))
        return np.percentile(samples, [16, 50, 84], axis=0).T

    def _resample_posterior(self, sampler, nwalkers, N=20):
        """Resample the walkers given the sampler."""
        percentiles = self._getpercentiles(sampler, N=N)
        pos = [[self.random_from_percentiles(param)
                for _ in range(nwalkers)]
               for param in percentiles]
        return np.array(pos)

    def _calculatetheta(self, sampler, method='random'):
        """Returns the best fit parameters given the provided method."""
        if method.lower() == 'median':
            return np.median(sampler.flatchain, axis=0)
        elif method.lower() == 'mean':
            return np.mean(sampler.flatchain, axis=0)
        elif method.lower() == 'random':
            idx = np.random.randint(0, sampler.flatchain.shape[0])
            return sampler.flatchain[idx]
        else:
            raise ValueError("Method must be 'median', 'mean' or 'random'.")

    def bestfitmodels(self, sampler):
        """Return the best-fit models."""
        return self._calculatemodels(self._calculatetheta(sampler))
