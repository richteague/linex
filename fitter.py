"""
Class to fit slab models with a range in temperature and density.

Input widths are non-thermal widths. As there are now temperature gradients
the linewidth will change as a function of temperature.

The sampling of the temperature and density gradients is set by three
parameters: 'maxnslab', the maximum number of samples along each gradient;
and 'mindX' (where X is temp or dens), which specifies the minimum spacing
between samples. This is in a bid to stop calls sampling gradients that are
essential non-existant. This needs to be tested. The resulting spectra will be
an average over all the spectra sampling the range in parameter space. This
results in the maximum of 100 spectra.

Notation, dx is the turbulent width, dV is the total line width. Both have
standard units of [km/s].
"""

import emcee
import numpy as np
import scipy.constants as sc
from radexcee.radexgridclass import radexgrid
import matplotlib.pyplot as plt


class fitdict:

    def __init__(self, slabdic, radexgridpath, **kwargs):
        """Fit the slab models."""
        self.dic = slabdic
        self.trans = sorted([k for k in self.dic.keys()])
        self.peaks = [self.dic[k].Tb for k in self.trans]
        self.widths = [self.dic[k].dx for k in self.trans]
        self.velaxs = [self.dic[k].velax for k in self.trans]
        self.spectra = [self.dic[k].spectrum for k in self.trans]
        self.grid = radexgrid(radexgridpath)
        self.mu = self.dic[self.trans[0]].mu
        self.params = []

        # Setting up the slab parameters. The number of samples will be the
        # shorter of a maximum of `mindslab` and 'maxval - minval / minsample'.
        # TODO: Create a better handling of this.

        self.maxnslab = 10
        self.mindtemp = 0.01
        self.minddens = 0.01

        # Flux calibration can be included by rescaling the spectra and peaks
        # by a random number within the specified amount.

        self.fluxcal = np.squeeze(kwargs.get('fluxcal', 0.0))
        if self.fluxcal.size == 1:
            self.fluxcal = np.squeeze([self.fluxcal for p in self.peaks])
        self.spectra = self.addfluxcal(self.spectra, self.fluxcal)
        self.peaks = [np.nanmax(s) for s in self.spectra]

        # The noise, a fraction of the peak value, can be specified either
        # individually as a list for the lines. Random noise is then added to
        # each of the spectra, including the peaks after converting the RMS
        # value to units of [K].

        self.rms = np.squeeze(kwargs.get('rms', None))
        if self.rms is not None:
            rms = []
            for i, t in enumerate(self.trans):
                dx = self.widths[i]
                velo = self.velaxs[i]
                spec = self.spectra[i]
                x0 = velo[abs(spec).argmax()]
                rms.append(np.nanstd(spec[abs(velo-x0) > 3. * dx]))
            self.rms = np.squeeze(rms)
        else:
            if self.rms.size == 1:
                self.rms = np.squeeze([self.rms for p in self.peaks])
            self.rms *= self.peaks

        self.spectra = self.addnoise(self.spectra, self.rms)
        self.peaks = [np.nanmax(s) for s in self.spectra]

        # Plot the lines to show what we are fitting. This allows a quick check
        # that everything is running OK.

        if kwargs.get('plot', False):
            self.plotlines()

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
        """Log-prior function."""

        tmin, tmax, dmin, dmax, sigma, x0s, dxs, dVs = self._parse(theta)

        # Minimum and maximum gradients must be in order.

        if tmin > tmax:
            return -np.inf
        if dmin > dmax:
            return -np.inf

        # Temperature, density and line widths must be within the RADEX grid.
        # Note that this means that the grid must be sufficiently broad to
        # encompass all possible scenarios.

        if not self.grid.in_grid(tmin, 'temp'):
            return -np.inf
        if not self.grid.in_grid(tmax, 'temp'):
            return -np.inf
        if not self.grid.in_grid(dmin, 'dens'):
            return -np.inf
        if not self.grid.in_grid(dmax, 'dens'):
            return -np.inf
        if not self.grid.in_grid(sigma, 'sigma'):
            return -np.inf
        for dV in dVs:
            if not self.grid.in_grid(dV, 'width'):
                return -np.inf

        # Line centre must be on the velocity axis while the non-thermal width
        # must be positive.

        for x0, velax in zip(x0s, self.velaxs):
            if x0 < velax[0] or x0 > velax[-1]:
                return -np.inf
        for dx in dxs:
            if dx < 0:
                return -np.inf
        return 0.0

    def _parse(self, theta):
        """Parses theta values given self.params."""

        # Slab properties.
        tidx = int('temp_max' in self.params)
        didx = int('dens_max' in self.params)
        tmin = theta[0]
        tmax = theta[tidx]
        dmin = theta[1+tidx]
        dmax = theta[1+tidx+didx]
        sigma = theta[2+tidx+didx]

        # Line properties.
        # TODO: fill this in backwards, might be quicker than zipping.

        x0s = [t for t, p in zip(theta, self.params) if 'x0' in p]
        dxs = [t for t, p in zip(theta, self.params) if 'dx' in p]
        dVs = [min([self.linewidth(dx, tmin) for dx in dxs]),
               max([self.linewidth(dx, tmax) for dx in dxs])]

        return tmin, tmax, dmin, dmax, sigma, x0s, dxs, dVs

    def _lnlike(self, theta):
        """Log-likelihood for fitting peak brightness."""

        # Parse all the free parameters.
        tmin, tmax, dmin, dmax, s, x0s, dxs, _ = self._parse(theta)

        # Create the gradients in temperature and density.
        tgrid = np.arange(tmin, tmax+1e-4, self.mindtemp)
        if len(tgrid) > self.maxnslab:
            tgrid = np.linspace(tmin, tmax, self.maxnslab)
        dgrid = np.arange(dmin, dmax+1e-4, self.minddens)
        if len(dgrid) > self.maxnslab:
            dgrid = np.linspace(dmin, dmax, self.maxnslab)

        # Build the models. Need to loop through each kinetic temperature and
        # n(H2) value. Each time making sure to calculate the correct width.

        toiter = zip(self.trans, self.velaxs, x0s, dxs)
        spectra = [[self._spectrum(j, t, d, s, v, x0, dx)
                    for d in dgrid for t in tgrid] for j, v, x0, dx in toiter]
        spectra = np.squeeze(spectra)

        if any('_min' in p for p in self.params):
            spectra = np.average(np.squeeze(spectra), axis=1)
        return self._lnx2(self.addfluxcal(spectra, self.fluxcal))

    def _spectrum(self, j, t, d, s, x, x0, dx):
        """Returns a spectrum on the provided velocity axis."""
        dV = self.linewidth(dx, t)
        A = self.grid.intensity(j, dV, t, d, s)
        return self.gaussian(x, x0, dV, A)

    def emcee(self, **kwargs):
        """Run emcee fitting just the profile peaks."""

        # Default values for the MCMC fitting.
        nwalkers = kwargs.get('nwalkers', 200)
        nburnin = kwargs.get('nburnin', 500)
        nsteps = kwargs.get('nsteps', 500)
        tgrad = kwargs.get('tgrad', True)
        dgrad = kwargs.get('dgrad', True)

        # Set up the parameters and the sampler.
        self.params = []
        if tgrad:
            self.params += ['temp_min', 'temp_max']
        else:
            self.params += ['temp']
        if dgrad:
            self.params += ['dens_min', 'dens_max']
        else:
            self.params += ['dens']
        self.params += ['sigma']
        for i in range(len(self.trans)):
            self.params += ['x0_%d' % i, 'dx_%d' % i]
        ndim = len(self.params)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)

        # If no p0 is specifed, we can run one with totally random starting
        # positions. This will double the time it takes to run MCMC. When
        # populating the initial starting positions, we can use the
        # grid.random_samples() functionalilty for slab properties. For the
        # line properties, we also assume small (< 200 m/s) non-thermal widths.

        p0 = kwargs.get('p0', None)
        if p0 is None:
            print('Estimating p0...')
            pos = [self.grid.random_samples(p.split('_')[0], nwalkers)
                   for p in self.params
                   if p.split('_')[0] in self.grid.parameters]
            for i in range(len(self.trans)):
                pos += [np.zeros(nwalkers)]
                pos += [np.random.uniform(0.0, 0.2, nwalkers)]
            sampler.run_mcmc(self.orderpos(pos), nburnin+nsteps)
            samples = sampler.chain[:, nburnin:].reshape((-1, ndim)).T
            p0 = [np.nanmedian(s) for s in samples]
            print('Starting values are:')
            for param, pp in zip(self.params, p0):
                print('%s: %.2e' % (param, pp))
        if len(p0) != ndim:
            raise ValueError('Wrong number of starting positions.')

        # Run the proper MCMC routine.

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        pos = [p0 + kwargs.get('scatter', 1e-2)*np.random.randn(ndim)
               for i in range(nwalkers)]
        sampler.run_mcmc(self.orderpos(pos), nburnin+nsteps)
        samples = sampler.chain[:, nburnin:].reshape((-1, ndim))
        return sampler, samples

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

    def orderpos(self, pos):
        """Orders the minimum and maximum values as appropriate."""
        pos = np.squeeze(pos)
        if 'temp_min' in self.params:
            tidx = 2
            pos_a = np.amin(pos[:2], axis=0)
            pos_b = np.amax(pos[:2], axis=0)
            pos = np.vstack((pos_a, pos_b, pos[2:]))
        else:
            tidx = 1
        if 'dens_min' in self.params:
            pos_a = np.amin(pos[tidx:2+tidx], axis=0)
            pos_b = np.amax(pos[tidx:2+tidx], axis=0)
            pos = np.vstack((pos[:tidx], pos_a, pos_b, pos[2+tidx:]))
        return self.rotatepos(pos)

    def rotatepos(self, pos):
        """Make sure the starting positions are the correct orientation."""
        if pos.shape[0] < pos.shape[1]:
            return pos.T
        return pos

    def linewidth(self, dx, temp):
        """Total linewidth, dV [km/s]."""
        return np.hypot(dx, np.sqrt(2. * sc.k * temp / self.mu / sc.m_p) / 1e3)

    def kinetictemp(self, dV, dx=None):
        """Kinetic temperature, temp [K]."""
        V = (dV * 1e3)**2
        if dx is not None:
            V -= (dx * 1e3)**2
        return V * self.mu * sc.m_p / sc.k / 2.

    def addfluxcal(self, spectra, fluxcal):
        """Add flux calibration uncertainty to the spectra."""
        if all(self.fluxcal == 0.0):
            return spectra
        toiter = zip(spectra, fluxcal)
        fcspec = [s * (np.random.randn() * fc + 1.) for s, fc in toiter]
        return np.squeeze(fcspec)

    def addnoise(self, spectra, rms):
        """Add noise to the spectra."""
        if any(self.rms == 0.0):
            raise ValueError('Must have non-zero noise.')
        toiter = zip(spectra, rms)
        nyspec = [s + (np.random.randn(s.size) * dT) for s, dT in toiter]
        return np.squeeze(nyspec)

    def gaussian(self, x, x0, dx, A):
        """Gaussian function."""
        return A * np.exp(-0.5 * np.power((x-x0)/dx, 2))

    def samples2percentiles(self, samples):
        """Returns the [16th, 50th, 84th] percentiles of the samples."""
        return np.array([np.percentile(s, [16, 50, 84]) for s in samples.T])

    def samples2uncertainties(self, samples):
        """Returns the percentiles in a [<y>, -dy, +dy] format."""
        pcnts = self.samples2percentiles(samples)
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts])
