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
    fixT [float]        - Fix the kinetic temperature to that value.
    fixn [float]        - Fix the volume density to that value. Note that LTE
                          will override this.
    fixN [float]        - Fix the column density to that value.
    Wkern [floats]      - [16, 50, 84]th percentiles of the kernel width used
                          for broadening the line to account for the beam.
    Hkern [floats]      - [16, 50, 84]th percentiles of the kernel height used
                          for broadening the line to account for the beam.
    thick [bool]        - Use a line profile assuming tau is Gaussian.
    oversample [bool]   - Oversample the profile and include Hanning smoothing.
    fluxval [float]     - If specified, the estimated flux calibration
                          uncertainty as a fraction. This value will be used as
                          the prior.

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
    nburns [int]        - Number of burn-ins before running the production. If
                          nburns > 1, the for the second and further burn-ins,
                          the starting position is sampled from the ending of
                          the last burn-in. This aims to remove walkers stuck
                          in local minima.
    burnin_sample [int] - Number of steps to sample in order to re-run the next
                          burn-in period if nburns > 1.
    burnin_steps [int]  - Number of steps taken in each burn-in.
    sample_steps [int]  - Number of steps taken for the production run.
    p0 [list]           - Starting positions for the emcee. If not specified
                          then random positions across the entire grid will be
                          chosen. While this allows one to explore the whole
                          space, it may result in the global minima not being
                          found.
-- TODO:

    1) Tidy up.
    2) Update this for a single spectral axis but multiple components.

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


# CS line frequencies. TODO: Read this from the grid file or something.
linefreq = {2: 146.9690287e9, 4: 244.9355565e9, 6: 342.8828503e9}


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
        self.freq = [linefreq[J] for J in self.trans]

        # Set up the type of model we want and if we want to hold any of the
        # values fixed during the fitting. Can include uncertainties.

        self.lte = kwargs.get('lte', kwargs.get('LTE', False))
        self.fixT = kwargs.get('fixT', False)
        self.fixN = kwargs.get('fixN', False)
        self.fixn = kwargs.get('fixn', False)
        self.dT = kwargs.get('dT', 0.0)
        self.dN = kwargs.get('dN', 0.0)
        self.dn = kwargs.get('dn', 0.0)

        # Control the type of turbulence we want to model.

        self.laminar = kwargs.get('laminar', False)
        self.logmach = kwargs.get('logmach', True)
        self.singlemach = kwargs.get('singlemach', True)
        if self.logmach and self.laminar:
            self.logmach = False
        if self.laminar:
            self.singlemach = True

        # Include addtional terms to model the observational effects.
        # With these variables, check to see if they're iterable, otherwise
        # extend them into a list with length self.ntrans. We can also include
        # an uncertainty such that at each call a random value is sampled.

        self._Wkern = kwargs.get('Wkern', None)
        self._Hkern = kwargs.get('Hkern', None)
        if (self._Wkern is None) != (self._Hkern is None):
            raise ValueError("Specify both or neither of Wkern / Hkern.")
        if self._Wkern is None:
            self.beamsmear = False
        else:
            self.beamsmear = True
            self._Wkern = np.squeeze(self._Wkern)
            self._Hkern = np.squeeze(self._Hkern)
            if self._Wkern.shape != self._Hkern.shape:
                raise ValueError("Wrong shape kernel percentiles.")
            if self._Wkern.ndim != 2:
                raise ValueError("Wrong shape kernel percentiles.")
            if self._Wkern.shape[0] != self.ntrans:
                raise ValueError("Not enough transitions specified.")

        # Specify the flux calibration uncertainty. If True will default to 10%
        # otherwise any value can be used. This will be used in the priors.

        self.fluxcal = kwargs.get('fluxcal', False)
        if self.fluxcal and type(self.fluxcal) is bool:
            self.fluxcal = 0.1

        self.oversample = kwargs.get('oversample', True)
        self.hanning = kwargs.get('hanning', True)

        # Decide whether to include beam dilution as a free parameter.
        # This will be applied to all lines homogeneously (assumes they have
        # the same beam sizes and underlying emission structure).

        self.beam_dilution = kwargs.get('beam_dilution', False)

        self.thick = kwargs.get('thick', True)
        if self.thick and not self.grid.hastau:
            raise ValueError("Attached grid does not have tau values.")

        # Set up the noise, estimating it in the spectrum and controlling the
        # noise model if requested.

        self.rms = self._estimatenoise()
        zipped = zip(self.spectra, self.rms)
        self.rms = [np.hypot(0.1 * Tb, rms) for Tb, rms in zipped]
        self.rms = np.squeeze(self.rms)

        self.GP = kwargs.get('GP', kwargs.get('gp', False))

        # Populate the parameters to fit.

        self._popparams()

        # Output messages.

        if self.verbose:
            print("Estimated RMS of each line:")
            for j, sig in zip(self.trans, self.rms):
                print("J = %d - %d: %.1f mK" % (j+1, j, 1e3 * np.nanmean(sig)))
            print("\n")
            if self.beamsmear:
                print("Including a beam broadening term.")
            if self.laminar:
                print("Assuming only thermal broadening.")
            if not self.singlemach:
                print("Fitting individual non-thermal width components.")
            if self.logmach:
                print("Fitting for the logarithm of Mach.")
            if self.lte:
                print("Assuming local thermodynamic equilibrium.")
            if self.GP:
                print("Using Gaussian processes to model noise.")
            if self.thick:
                print("Including opacity in line profile calculation.")
            if self.oversample:
                print("Oversampling the line profile.")
            if self.hanning:
                print("Including a Hanning smoothing of the data.")
            if self.fluxcal:
                print("Fitting for flux calibration.")
            if self.beam_dilution:
                print("Fitting for a beam filling factor.")
            print("\n")
        self.diagnostics = kwargs.get('diagnostics', False)

        return

    def _makeiterable(self, name, default, **kwargs):
        """Return a kwarg argument that's iterable."""
        val = np.array([kwargs.get(name, default)]).flatten()
        if len(val) == self.ntrans:
            return val
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
        self.params = []

        # Physical properties.
        if not self.fixT:
            self.params += ['temp']
        if not self.fixn and not self.lte:
            self.params += ['dens']
        if not self.fixN:
            self.params += ['sigma']
        if self.singlemach:
            self.params += ['mach']
        else:
            for i in range(len(self.trans)):
                self.params += ['mach_%d' % i]
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i]

            # Hyper-parameters for GPs.
            if self.GP:
                self.params += ['sig_%d' % i, 'vcorr_%d' % i]

        # Flux calibration.
        for i in range(len(self.trans)):
            if self.fluxcal:
                self.params += ['fc_%d' % i]
        if self.laminar:
            self.params = [p for p in self.params if 'mach' not in p]

        # Beam dilution.
        if self.beam_dilution:
            self.params += ['ff']
        self.ndim = len(self.params)
        return

    def _parse(self, theta):
        """Parses theta into properties."""
        zipped = zip(theta, self.params)

        # Physical parameters.
        if self.fixT:
            temp = self.fixT + self.dT * np.random.randn()
        else:
            temp = [t for t, p in zipped if 'temp' in p][0]
        if self.fixn:
            dens = self.fixn + self.dn * np.random.randn()
        elif self.lte:
            dens = self.grid.maxdens
        else:
            dens = [t for t, p in zipped if 'dens' in p][0]
        if self.fixN:
            sigma = self.fixN + self.dN * np.random.randn()
        else:
            sigma = [t for t, p in zipped if 'sigma' in p][0]
        if not self.laminar:
            mach = [t for t, p in zipped if 'mach' in p]
        else:
            mach = [0.0]
        x0s = [t for t, n in zipped if 'x0_' in n]

        # Hyper-parameters for noise.
        if self.GP:
            sigs = [t for t, n in zipped if 'sig_' in n]
            corrs = [t for t, n in zipped if 'vcorr_' in n]
        else:
            sigs = [np.nan for _ in range(self.ntrans)]
            corrs = [np.nan for _ in range(self.ntrans)]

        # Flux calibration.
        if self.fluxcal:
            fc = [t for t, n in zipped if 'fc_' in n]
        else:
            fc = [1.0 for _ in range(self.ntrans)]

        # Beam dilution.
        if self.beam_dilution:
            ff = [t for t, n in zipped if 'ff' in n][0]
        else:
            ff = 1.0
        return temp, dens, sigma, mach, x0s, sigs, corrs, fc, ff

    def _lnprob(self, theta):
        """Log-probability function with simple chi-squared likelihood."""
        lnp = self._lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf
        return self._lnlike(theta)

    def _lnprior(self, theta):
        """Log-prior function. Uninformative priors."""
        temp, dens, sigma, mach, x0s, sigs, corrs, fc, ff = self._parse(theta)

        # Excitation parameters.
        if not self.grid.in_grid(temp, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dens, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf

        # Non-thermal broadening.
        if self.logmach:
            if not all([-5.0 < m < 5.0 for m in mach]):
                return -np.inf
        else:
            if not all([0.0 <= m < 5.0 for m in mach]):
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

        # Flux calibration uncertainty.
        if self.fluxcal:
            if not all([abs(1. - f) < self.fluxcal for f in fc]):
                return -np.inf

        # Beam filling factor.
        if self.beam_dilution:
            if not 0.0 < ff <= 1.0:
                return -np.inf

        return 0.0

    def _calculatemodels(self, theta):
        """Calculates the appropriate models."""
        t, d, s, m, x, _, _, f, ff = self._parse(theta)
        return [self._spectrum(j, t, d, s, v, x[i], m[i % len(m)], f[i], ff)
                for i, (j, v) in enumerate(zip(self.trans, self.velaxs))]

    def _spectrum(self, j, t, d, s, x, x0, mach, f, ff, N=100):
        """Returns a spectrum on the provided velocity axis."""

        # Choose the correct velocity axis to calculate the profile on.

        if self.oversample:
            xx = np.linspace(x[0], x[-1], x.size * N)
        else:
            xx = x

        j_idx = abs(self.trans - j).argmin()

        if self.logmach:
            dV = self.linewidth(t, np.power(10, mach))
        else:
            dV = self.linewidth(t, mach)

        # INCLUDE AN OPTICAL DEPTH PRIOR.
        tau = self.grid.tau(j, dV, t, d, s)
        if any(tau > 1.0):
            return np.ones(x.size) * np.nan

        # Generate the initial spectrum, either as a simple Gaussian or with
        # the more accurate profile accounting for the optical depth.

        if not self.thick:
            Tb = self.grid.intensity(j, dV, t, d, s)
            s = self.gaussian(xx, x0, dV, Tb * ff)
        else:
            Tex = self.grid.Tex(j, dV, t, d, s)
            nu = self.freq[j_idx]
            s = self.thickline(xx, x0, dV, Tex, tau, nu, ff)

        if self.oversample:
            s = np.interp(x, x, s[N/2::N])

        if self.hanning:
            s = np.convolve(s, [0.5, 0.5], mode='same')
            # s = np.convolve(s, [0.25, 0.5, 0.25], mode='same')

        if self.fluxcal:
            s *= f

        if self.beamsmear:
            W = self.random_from_percentiles(self._Wkern[j_idx])
            H = self.random_from_percentiles(self._Hkern[j_idx])
            if W > 0.0 and H > 0.0:
                s = self.convolve_spectrum(s, W, H)
        return s

    def thickline(self, x, x0, dx, Tex, tau, nu, ff):
        """Returns an optically thick line profile."""
        Tb = self._calculateTb(Tex, nu)
        return Tb * (1. - np.exp(-self.gaussian(x, x0, dx, tau))) * ff

    def convolve_spectrum(self, x, W, H):
        """Convolve array x with Hanning kernel of width W and height H."""
        Wa, Wb = int(np.floor(W)), int(np.ceil(W))
        ya, yb = self._convolve(x, Wa, H), self._convolve(x, Wb, H)
        if np.isclose(Wa, W):
            return ya
        elif np.isclose(Wb, W):
            return yb
        weight = [W - float(Wa), float(Wb) - W]
        return np.average([ya, yb], weights=weight, axis=0)

    def _convolve(self, x, W, H):
        """Single convolution function."""
        K = np.hanning(W)
        K *= H / np.sum(K)
        K = K[K > 0.0]
        if len(K) <= 1:
            return x
        y = [np.convolve(x, K, mode='same'),
             np.convolve(x[::-1], K, mode='same')[::-1]]
        return np.average(y, axis=0)

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

    def _lnlike(self, theta):
        """Log-likelihood with chi-squared likelihood function."""
        models = self._calculatemodels(theta)

        for model in models:
            if any(np.isnan(model)):
                return -np.inf

        if not self.GP:
            return self._chisquared(models)
        lnx2 = 0.0
        _, _, _, _, _, sigs, corrs, _, _ = self._parse(theta)
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

    def _startingpositions(self, nwalkers):
        """Return random starting positions."""

        pos = []

        # Excitation conditions.
        if not self.fixT:
            pos += [np.random.uniform(20, 40, nwalkers)]
            # pos += [self.grid.random_samples('temp', nwalkers)]
        if not self.fixn and not self.lte:
            pos += [self.grid.random_samples('dens', nwalkers)]
        if not self.fixN:
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
                pos += [np.random.uniform(-3, -2, nwalkers)]
                pos += [np.random.uniform(-3, -2, nwalkers)]

        # Flux calibration.
        if self.fluxcal:
            fc = self.fluxcal
            for i in range(self.ntrans):
                pos += [np.random.uniform(1 - fc, 1. + fc, nwalkers)]

        # Beam dilution.
        if self.beam_dilution:
            pos += [np.random.uniform(0.1, 1.0, nwalkers)]

        return pos

    def emcee(self, **kwargs):
        """Run emcee with multiple runs to make the final nice."""

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

    def _resample_posterior(self, sampler, nwalkers, N=20):
        """Resample the walkers given the sampler."""
        percentiles = self._getpercentiles(sampler, N=N)
        pos = [[self.random_from_percentiles(param)
                for _ in range(nwalkers)]
               for param in percentiles]
        return np.array(pos)

    def _getpercentiles(self, sampler, N=50):
        """Returns the perncentiles of the final N steps."""
        samples = sampler.chain[:, -50:]
        samples = samples.reshape((-1, samples.shape[-1]))
        return np.percentile(samples, [16, 50, 84], axis=0).T

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

    def samples2percentiles(self, samples, percentiles=[16, 50, 84]):
        """Returns the [16th, 50th, 84th] percentiles of the samples."""
        return np.percentile(samples, percentiles, axis=0).T

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
