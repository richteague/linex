# linex

Functions and classes to run line excitation analysis.

## Slab models

Basic slab model is `fitter_slabmodels.py`. This was expanded to consider two
temperatures, a low and high value in `fitter_slabmodels_gradient.py`, however
this did not seem to yield a good approach. Gaussian Processes were implemented
in `fitter_slabmodels_george` where both the variance and length scale of the
noise were allowed to vary. Finally `fitter_slabmodels_fluxcal` included the
flux calibration uncertainty.
