"""
Functions to help plotting the MCMC fitting from fitter.py.
"""

import corner
import numpy as np
import matplotlib.pyplot as plt


def plotsampling(sampler, params, color='dodgerblue', title=None):
    """Plot the MCMC sampling. Include percentiles."""

    # Loop through each of the parameters and plot their sampling.
    for param, samples in zip(params, sampler.chain.T):
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


def plotcorner(sampler, params):
    """Plot the corner plot."""
    corner.corner(sampler.flatchain, labels=params,
                  quantiles=[0.16, 0.5, 0.84], show_titles=True)
    return


def plotobservations(trans, velaxs, spectra, rms, ax=None):
    """Plot the emission lines which are to be fit."""
    if ax is None:
        fig, ax = plt.subplots()
    for i, t in enumerate(trans):
        l = ax.plot(velaxs[i], spectra[i], lw=1.25,
                    label=r'J = %d - %d' % (t+1, t))
        ax.fill_between(velaxs[i],
                        spectra[i] - 3.0 * rms[i],
                        spectra[i] + 3.0 * rms[i],
                        lw=0.0, alpha=0.2, color=l[0].get_color(),
                        zorder=-3)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xlabel(r'${\rm Velocity \quad (km s^{-1})}$')
    ax.set_ylabel(r'${\rm Brightness \quad (K)}$')
    ax.legend(frameon=False, markerfirst=False)
    return ax


def plotbestfit(trans, velaxs, models, ax=None):
    """Plot the best-fit spectra."""
    if ax is None:
        fig, ax = plt.subplots()
    for x, y, J in zip(velaxs, models, trans):
        ax.scatter(x, y, color='r', edgecolor='k')
    ax.legend(fontsize=6)
    return ax
