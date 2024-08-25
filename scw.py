"""Scalar cylindrical wave module.

.. autosummary::
   :toctree:

   periodic_to_spw
   rotate
   to_ssw
   translate
   translate_periodic

"""

import acoutreams._wavesacoustics as wv
import treams.special as sc
import numpy as np
from treams import lattice
import scipy.special as ss


def _translate_s(kz, mu, qz, m, krr, phi, z, *args, **kwargs):
    if abs(krr) < 1e-16 and abs(z) < 1e-16:
        return 0.0j
    return wv.tl_scw(kz, mu, qz, m, krr, phi, z, *args, **kwargs)

def _translate_r(kz, mu, qz, m, krr, phi, z, *args, **kwargs):
    return wv.tl_scw_r(kz, mu, qz, m, krr, phi, z, *args, **kwargs)


_translate_s = np.vectorize(_translate_s)
_translate_r = np.vectorize(_translate_r)


def translate(kz, mu, qz, m, krr, phi, z, singular=True, *args, **kwargs):
    """Translation coefficient for spherical modes.

    Returns the correct translation coefficient from :func:wv.tl_ssw_a,
    and :func:wv.tl_vsw_rB or a combination thereof for the specified mode and
    basis.

    Args:
        lambda_ (int, array_like): Degree of the destination mode
        mu (int, array_like): Order of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        kr (float or complex, array_like): Translation distance in units of the wave number
        theta (float, array_like): Polar angle
        phi (float, array_like): Azimuthal angle
        singular (bool, optional): If true, singular translation coefficients are used,
            else regular coefficients. Defaults to ``True``.

    Returns:
        complex
    """
    if singular:
        return _translate_s(kz, mu, qz, m, krr, phi, z, *args, **kwargs)
    return _translate_r(kz, mu, qz, m, krr, phi, z, *args, **kwargs)

def _rotate(kz, mu, qz, m, phi, *args, **kwargs):
    if (kz == qz) and (m == mu):
        return np.exp(1j * m * phi, *args, **kwargs)
    else:
        return 0.+0.j
    
_rotate = np.vectorize(_rotate)

def rotate(kz, mu, qz, m, phi, *args, **kwargs):
    return _rotate(kz, mu, qz, m, phi, *args, **kwargs)

   
def _periodic_to_spw(kx, ky, kzpw, kzcw, m, a, *args, **kwargs):
   krho = np.sqrt(kx * kx + ky * ky)
   ky_s = ky
   if np.abs(kzcw - kzpw) < 1e-14:
      if ky_s == 0:
            ky_s = 1e-20 + 1e-20j
      elif np.imag(ky_s) < 0 or (np.imag(ky_s) == 0 and np.real(ky_s) < 0):
            ky_s = -ky_s
      if krho == 0:
         return 2 * np.power(-1j, m) / (abs(a) * ky_s)
      return 2 * np.power((-1j * kx + ky) / krho, m, *args, **kwargs) / (abs(a) * ky_s)
   else:
      return 0.0j + sc.lpmv(0, 1, 0, *args, **kwargs)

_periodic_to_spw = np.vectorize(_periodic_to_spw)

def periodic_to_spw(kx, ky, kzpw, kzcw, m, a, *args, **kwargs):
    return _periodic_to_spw(kx, ky, kzpw, kzcw, m, a, *args, **kwargs)

def _to_sw(l, m, kz, mu, k, *args, **kwargs):
    if m == mu:
        return (
            np.power(1j, l - m, *args, **kwargs)
            * np.sqrt(4 * np.pi * (2 * l + 1), *args, **kwargs)
            * np.sqrt(ss.gamma(l - m + 1) / ss.gamma(l + m + 1), *args, **kwargs)
            * sc.lpmv(m, l, kz/k+0j, *args, **kwargs)
            )
    return 0.0j + sc.lpmv(0, 1, 0, *args, **kwargs)

_to_sw = np.vectorize(_to_sw)

def to_sw(l, m, kz, mu, k, *args, **kwargs):
    return _to_sw(l, m, kz, mu, k, *args, **kwargs)


def translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, eta=0):
    """Translation coefficients in a lattice.

    Returns the translation coefficents for the given modes in a lattice. The
    calculation uses the fast converging sums of :mod:`treams.lattice`.

    Args:
        ks (float or complex): Wave number of longitudinal waves in the medium
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension
            with `1 <= D <= 2`
        a (float, (D,D)-array): Lattice vectors in each row of the array
        rs (float, (M, 3)-array): Shift vectors with respect to one lattice point
        out (3- or 4-tuple of integer arrays): Output modes
        in_ (3- or 4-tuple of integer arrays): Input modes, if none are given equal to
            the output modes
        rsin (float): Shift vectors to use with the input modes, if non are given equal
            to `rs`
        eta (float or complex, optional): Cut between real and reciprocal space
            summation, if equal to zero, an estimation for the optimal value is done.

    Returns:
        complex array
    """
    if in_ is None:
        in_ = out
    out = (*(np.array(o) for o in out),)
    in_ = (*(np.array(i) for i in in_),)
    if len(out) < 2 or len(out) > 3:
        raise ValueError(f"invalid length of output modes {len(out)}, must be 2 or 3")
    elif len(out) == 2:
        out = (np.zeros_like(out[0]),) + out
    if len(in_) < 2 or len(in_) > 3:
        raise ValueError(f"invalid length of input modes {len(in_)}, must be 2 or 3")
    elif len(in_) == 2:
        in_ = (np.zeros_like(in_[0]),) + in_
    if rsin is None:
        rsin = rs
    modes = -out[2][:, None] + in_[2]
    ks = np.array(ks)
    ks = ks.reshape((-1,))
    krhos = np.sqrt((ks[0] * ks[0] - out[1][:, None] * in_[1]).astype(complex))  # todo: out not necessary this only simplifies krhos[compute] below
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    compute = np.equal(out[1][:, None], in_[1])
    kpar = np.array(kpar)
    rs = np.array(rs)
    if rs.ndim == 1:
        rs = rs.reshape((1, -1))
    rsin = np.array(rsin)
    if rsin.ndim == 1:
        rsin = rsin.reshape((1, -1))
    rsdiff = -rs[out[0], None, :] + rsin[in_[0], :]
    if kpar.ndim == 0 or kpar.shape[-1] == 1:
        dlms = lattice.lsumcw1d_shift(modes[compute], krhos[compute], kpar, a, rsdiff[compute, :2], eta)
    else:
        dlms = lattice.lsumcw2d(modes[compute], krhos[compute], kpar, a, rsdiff[compute, :2], eta)
    res = np.zeros((out[0].shape[0], in_[0].shape[0]), complex)
    res[compute] = dlms
    res = res * np.exp(-1j * in_[1] * rsdiff[:, :, 2])
    return res
