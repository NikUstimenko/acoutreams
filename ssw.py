"""Scalar spherical wave module.

.. autosummary::
   :toctree:

   periodic_to_scw
   periodic_to_spw
   rotate
   translate
   translate_periodic

"""

import acoutreams._wavesacoustics as wv
import treams.special as sc
import numpy as np
from treams import lattice
import scipy.special as ss

def _translate_s(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs):
    if abs(kr) < 1e-16:
        return 0.0j
    return wv.tl_ssw(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs)

def _translate_r(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs):
    return wv.tl_ssw_r(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs)


_translate_s = np.vectorize(_translate_s)
_translate_r = np.vectorize(_translate_r)


def translate(lambda_, mu, l, m, kr, theta, phi, singular=True, *args, **kwargs):
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
        return _translate_s(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs)
    return _translate_r(lambda_, mu, l, m, kr, theta, phi, *args, **kwargs)


def _rotate(lambda_, mu, l, m, phi, theta, psi, *args, **kwargs):
    """
    Rotation coefficient for the rotation around the Euler angles phi, theta and psi.
    It is intended to be used to construct the rotation matrix manually.
    """
    if lambda_ == l:
        return sc.wignerd(l, mu, m, phi, theta, psi, *args, **kwargs)
    else:
        return 0.0j

_rotate = np.vectorize(_rotate)


def rotate(lambda_, mu, l, m, phi, theta=0, psi=0, *args, **kwargs):
    """Rotation coefficient for scalar spherical modes.

    Returns the correct rotation coefficient from :func:`treams.special.wignerd`. The
    angles are given as Euler angles in `z-y-z`-convention. In the intrinsic (object
    fixed coordinate system) convention the rotations are applied in the order phi
    first, theta second, psi third. In the extrinsic (global or reference frame fixed
    coordinate system) the rotations are applied psi first, theta second, phi third.

    Args:
        lambda_ (int, array_like): Degree of the destination mode
        mu (int, array_like): Order of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        phi (float or complex, array_like): First Euler angle
        theta (float, array_like): Second Euler angle
        psi (float, array_like): Third Euler angle

    Returns:
        complex
    """
    return _rotate(lambda_, mu, l, m, phi, theta, psi, *args, **kwargs)


def _transl_a_lattice(lambda_, mu, l, m, dlms):
    pref = np.power(-1, np.abs(m)) * np.sqrt(4 * np.pi * (2 * l + 1) * (2 * lambda_ + 1)) * np.power(1.0j, lambda_ - l)
    dlm = 0
    res = 0
    for p in range(l + lambda_, max(abs(lambda_ - l), abs(mu - m)) - 1, -2):
        dlm = dlms[p * (p + 1) + m - mu]
        res += (dlm * np.power(1.0j, p) * np.sqrt(2 * p + 1) 
                * sc.wigner3j(l, lambda_, p, m, -mu, -m + mu) * sc.wigner3j(l, lambda_, p, 0, 0, 0))
    return res * pref

def _transl_a_lattice_r(lambda_, mu, l, m, dlms):
    pref = np.power(-1, np.abs(m)) * np.sqrt(4 * np.pi * (2 * l + 1) * (2 * lambda_ + 1)) * np.power(1.0j, lambda_ - l)
    dlm = 0
    res = 0
    for p in range(l + lambda_, max(abs(lambda_ - l), abs(mu - m)) - 1, -2):
        dlm = dlms[p * (p + 1) + m - mu]
        if p == 0:
            dlm += 1/np.sqrt(4*np.pi)
        res += dlm * np.power(1.0j, p) * np.sqrt(2 * p + 1) * sc.wigner3j(l, lambda_, p, m, -mu, -m + mu) * sc.wigner3j(l, lambda_, p, 0, 0, 0)
    return res * pref

def translate_periodic_r(ks, kpar, a, rs, out, in_=None, rsin=None, eta=0, func=lattice.lsumsw):
    """Regular translation coefficients in a lattice.

    Returns the translation coefficents for the given modes in a lattice. The calculation
    uses the fast converging sums of :mod:`treams.lattice`.


    Args:
        ks (float or complex): Wave number of longitudinal waves in the medium
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension with `1 <= D <= 3`
        a (float, (D,D)-array): Lattice vectors in each row of the array
        rs (float, (M, 3)-array): Shift vectors with respect to one lattice point
        out (2- or 3-tuple of integer arrays): Output modes
        in_ (2- or 3-tuple of integer arrays): Input modes, if none are given equal to
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
    modes = np.array([
        [l, m]
        for l in range(np.max(out[1]) + np.max(in_[1]) + 1)
        for m in range(-l, l + 1)
    ])
    kpar = np.array(kpar)
    rs = np.array(rs)
    if rs.ndim == 1:
        rs = rs.reshape((1, -1))
    rsin = np.array(rsin)
    if rsin.ndim == 1:
        rsin = rsin.reshape((1, -1))
    rsdiff = -rs[:, None, None, None, :] + rsin[:, None, None, :]

    dim = 1 if kpar.ndim == 0 else kpar.shape[-1]
    # The result has the shape (n_rs, n_rs, n_ks, n_modes)
    dlms = func(dim, modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)

    res = np.zeros((out[1:][0].shape[-1], in_[1:][0].shape[-1]), np.complex128)
     
    for i in range(out[1:][0].shape[-1]):
        for j in range(in_[1:][0].shape[-1]):
                res[i][j] = _transl_a_lattice_r(in_[1:][0][j], in_[1:][1][j], out[1:][0][i], out[1:][1][i], dlms[0][0][0])
    
    return res

def translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, eta=0, func=lattice.lsumsw):
    """Translation coefficients in a lattice.

    Returns the translation coefficents for the given modes in a lattice. The calculation
    uses the fast converging sums of :mod:`treams.lattice`.


    Args:
        ks (float or complex): Wave number of longitudinal waves in the medium
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension with `1 <= D <= 3`
        a (float, (D,D)-array): Lattice vectors in each row of the array
        rs (float, (M, 3)-array): Shift vectors with respect to one lattice point
        out (2- or 3-tuple of integer arrays): Output modes
        in_ (2- or 3-tuple of integer arrays): Input modes, if none are given equal to
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
    modes = np.array([
        [l, m]
        for l in range(np.max(out[1]) + np.max(in_[1]) + 1)
        for m in range(-l, l + 1)
    ])
    kpar = np.array(kpar)
    rs = np.array(rs)
    if rs.ndim == 1:
        rs = rs.reshape((1, -1))
    rsin = np.array(rsin)
    if rsin.ndim == 1:
        rsin = rsin.reshape((1, -1))
    rsdiff = -rs[:, None, None, None, :] + rsin[:, None, None, :]

    dim = 1 if kpar.ndim == 0 else kpar.shape[-1]
    # The result has the shape (n_rs, n_rs, n_ks, n_modes)
    dlms = func(dim, modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)
    res = np.zeros((out[1:][0].shape[-1], in_[1:][0].shape[-1]), np.complex128)
    for i in range(out[1:][0].shape[-1]):
        for j in range(in_[1:][0].shape[-1]):
                res[i][j] = _transl_a_lattice(in_[1:][0][j], in_[1:][1][j], out[1:][0][i], out[1:][1][i], dlms[0][0][0])  
    return res
         

def _periodic_to_spw(kx, ky, kz, l, m, area, *args, **kwargs):
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    phi = np.arctan2(ky, kx)
    kz_s = kz
    if kz == 0:
        kz_s = 1e-20 + 1e-20j
    elif (np.imag(kz_s) < 0) or (np.imag(kz_s) == 0 and np.real(kz_s) < 0):
        kz_s = -kz_s
    prefactor = (
        np.sqrt(np.pi * (2 * l + 1))
        * np.sqrt(ss.gamma(l - m + 1) / ss.gamma(l + m + 1))
        * np.exp(1j * m * phi)
        * np.power(-1j, l)
        / (area * k * kz_s)
    )
    return prefactor * sc.lpmv(m, l, kz/k+0j, *args, **kwargs)

_periodic_to_spw = np.vectorize(_periodic_to_spw)


def periodic_to_spw(kx, ky, kz, l, m, area, *args, **kwargs):
    """Convert periodic spherical wave to plane wave.

    Returns the coefficient for the basis change in a periodic arrangement of spherical
    modes to plane waves. For multiple positions only diagonal values (with respect to
    the position) are returned. A correct phase factor is still necessary for the full
    result.

    Args:
        kx (float, array_like): X component of destination mode wave vector
        ky (float, array_like): Y component of destination mode wave vector
        kz (float or complex, array_like): Z component of destination mode wave vector
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        area (float, array_like): Unit cell area

    Returns:
        complex
    """
    return _periodic_to_spw(kx, ky, kz, l, m, area, *args, **kwargs)

def _periodic_to_scw(kz, mu, l, m, k, a, *args, **kwargs):
    if mu != -m:
        return 0.0j + sc.lpmv(0, 1, 0, *args, **kwargs)
    prefactor = (
        np.sqrt(np.pi * (2 * l + 1), *args, **kwargs) *
        np.sqrt(ss.gamma(l - m + 1) / ss.gamma(l + m + 1), *args, **kwargs) * 
        0.5 * 
        np.power(-1j, l - m, *args, **kwargs)
        / (a * k)
    ) 
    return prefactor * sc.lpmv(m, l, kz/k+0j, *args, **kwargs)

 
_periodic_to_scw = np.vectorize(_periodic_to_scw)


def periodic_to_scw(kz, m, l, mu, k, area, *args, **kwargs):
    """Convert periodic spherical wave to plane wave.

    Returns the coefficient for the basis change in a periodic arrangement of spherical
    modes to plane waves. For multiple positions only diagonal values (with respect to
    the position) are returned. A correct phase factor is still necessary for the full
    result.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        kz (float, array_like): Z component of destination mode wave vector
        m (int, array_like): Order or the destination mode
        l (int, array_like): Degree of the source mode
        mu (int, array_like): Order of the source mode
        k (float or complex, array_like): Wave number
        area (float, array_like): Unit cell area

    Returns:
        complex
    """
    return _periodic_to_scw(kz, m, l, mu, k, area, *args, **kwargs)