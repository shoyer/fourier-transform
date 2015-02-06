"""
fourier-transform
=================

Fourier Transform for Python with bookkeeping for labels, bounds, conventions
and units. A convenience wrapper for SciPy's FFT.

Usage
-----

    from fourier_transform import fourier_transform
    f, X = fourier_transform(t, x)
"""
from scipy.fftpack import fft, fftshift, ifftshift, fftfreq
import numpy as np


def _slice_along_axis(start=None, stop=None, step=None, axis=0, ndim=1):
    """
    Returns an N-dimensional slice along only the specified axis
    """
    return tuple(slice(start, stop, step)
                 if (n == axis) or (n == ndim + axis)
                 else slice(None)
                 for n in xrange(ndim))


def _is_constant(x, atol=1e-7, positive=None):
    """
    True if x is a constant array, within atol
    """
    x = np.asarray(x)
    return (np.max(np.abs(x - x[0])) < atol and
            (np.all((x > 0) == positive) if positive is not None else True))


def _symmetrize(t, x, axis=-1):
    """
    Zero-pad the provided signal to ensure the labels are symmetric.
    """
    if not _is_constant(np.diff(t), positive=True):
        raise ValueError('sample times must differ by a positive constant')

    t = np.asarray(t)
    x = np.asarray(x)

    T = max(t[-1], -t[0])
    dt = t[1] - t[0]

    N_plus = int((T - t[-1]) / dt) + 1
    N_minus = int((T + t[0]) / dt) + 1

    t_sym = np.concatenate([t[0] - dt * np.arange(1, N_minus)[::-1], t,
                            t[-1] + dt * np.arange(1, N_plus)])

    new_shape = tuple(n if i != axis and i != x.ndim + axis
                      else t_sym.size
                      for i, n in enumerate(x.shape))
    x_sym = np.zeros(new_shape, dtype=x.dtype)
    start, end = (np.searchsorted(t_sym, ti) for ti in (t[0], t[-1]))
    x_sym[_slice_along_axis(start, end + 1, axis=axis, ndim=x.ndim)] = x

    return t_sym, x_sym


def fourier_transform(t, x, axis=-1, sign=1, convention='angular',
                      unit_convert=1, rw_freq=0):
    r"""
    Fourier transform a signal at the labeled times using FFT

    By default, approximates the integral:

    .. math::
        X(\omega) = \int e^{s i (\omega - \omega_0) t} x(t) dt

    where :math:`s` is sign argument and :math:`\omega_0` is the rotating wave
    frequency.

    The signal is assumed to be zero at any times at which it is not provided.

    Parameters
    ----------
    t : np.ndarray
        1D array giving the times at which the signal is defined.
    x : np.ndarray
        Signal to Fourier transform.
    axis : int, optional
        Axis along which to apply the Fourier transform to ``x``.
    sign : {1, -1}, optional
        Sign in the exponent.
    convention : {'angular', 'linear'}, optional
        Return angular or linear frequencies.
    unit_convert : number, optional
        Unit conversion from frequency to time units.
    rw_freq : number, optional
        Frequency of the rotating frame in which the signal is sampled.

    Returns
    -------
    f : np.ndarray
        Frequencies at which the Fourier transformed signal is defined. These
        are equally spaced and ascending.
    X : np.ndarray
        The Fourier transformed signal.
    """
    if t.ndim != 1:
        raise ValueError('t must be one dimensional')
    if t.size != x.shape[axis]:
        raise ValueError('t must have the same length as the shape of x along '
                         'the given axis')
    if sign not in [-1, +1]:
        raise ValueError('invalid sign: %r' % sign)

    if convention == 'angular':
        unit_convert /= 2 * np.pi
    elif convention != 'linear':
        raise ValueError("convention must be 'angular' or 'linear'")

    t, x = _symmetrize(t, x, axis)

    N = x.shape[axis]
    dt = t[1] - t[0]

    f = fftshift(fftfreq(N, dt * unit_convert))
    X = fftshift(fft(ifftshift(x * dt, axes=axis), axis=axis), axes=axis)

    if sign == 1:
        f = -f[::-1]
        X = X[_slice_along_axis(step=-1, axis=axis, ndim=X.ndim)]
    f += rw_freq
    return f, X
