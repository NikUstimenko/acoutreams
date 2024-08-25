"""Scalar plane wave module.

.. autosummary::
   :toctree:

   to_scw
   to_ssw
   translate
   permute_xyz

"""

import numpy as np
import treams.special as sc
import scipy.special as ss


def translate(kx, ky, kz, x, y, z, *args, **kwargs):
    r"""Translation coefficient for scalar plane wave modes

    The translation coefficient is the phase factor
    :math:`\mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}`.

    Args:
        kx, ky, kz (float or complex, array_like): Wave vector components
        x, y, z (float, array_like): Translation vector components

    Returns:
        complex
    """
    return np.exp(1j * (kx * x + ky * y + kz * z), *args, **kwargs)


def _to_ssw(l, m, kx, ky, kz, *args, **kwargs):
    phi = np.arctan2(ky, kx)
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    pref = (
        2 * np.sqrt(np.pi * (2 * l + 1))
        * np.sqrt(ss.gamma(l - m + 1) / ss.gamma(l + m + 1))
        * np.power(1j, l)
        * np.exp(-1j * m * phi)
    )
    return pref * sc.lpmv(m, l, kz / k, *args, **kwargs)

_to_ssw = np.vectorize(_to_ssw)

def to_ssw(l, m, kx, ky, kz, *args, **kwargs):
    """Coefficient for the expansion of a scalar plane wave in scalar spherical waves

    Returns the coefficient for the basis change from a scalar plane wave to a scalar spherical wave.
    For multiple positions only diagonal values (with respect to the position) are
    returned.

    Args:
        l (int, array_like): Degree of the scalar spherical wave
        m (int, array_like): Order of the scalar spherical wave
        kx (float, array_like): X component of scalar plane wave's wave vector
        ky (float, array_like): Y component of scalar plane wave's wave vector
        kz (float, array_like): Z component of scalar plane wave's wave vector

    Returns:
        complex
    """
    return _to_ssw(l, m, kx, ky, kz, *args, **kwargs)

    
def _to_scw(kzcw, m, kx, ky, kzpw, *args, **kwargs):  
    krho = np.sqrt(kx * kx + ky * ky)
    if np.abs(kzcw - kzpw) <= 1e-14:
        if m == 0:
            return np.power(1, 0, *args, **kwargs)
        if krho == 0:
            return np.power(1j, m, *args, **kwargs)
        return np.power((1j * kx + ky) / krho, m, *args, **kwargs)
    elif np.abs(kzcw - kzpw) > 1e-14:
        return 0.0j + sc.lpmv(0, 1, 0, *args, **kwargs)
    

_to_scw = np.vectorize(_to_scw)    

def to_scw(kzcw, m, kx, ky, kzpw, *args, **kwargs):
    """Coefficient for the expansion of a scalar plane wave in scalar cylindricrical waves

    Returns the coefficient for the basis change from a scalar plane wave to a scalar cylindrical wave.
    For multiple positions only diagonal values (with respect to the position) are returned.

    Args:
        kzcw (float, array_like): Z component of the scalar cylindrical wave
        m (int, array_like): Order of the scalar cylindrical wave
        kx (float, array_like): X component of scalar plane wave's wave vector
        ky (float, array_like): Y component of scalar plane wave's wave vector
        kzpw (float, array_like): Z component of scalar plane wave's wave vector

    Returns:
        complex
    """
    return _to_scw(kzcw, m, kx, ky, kzpw, *args, **kwargs)


def _xyz_to_zxy(kx, ky, kz, *args, **kwargs):
    return np.power(1, 0, *args, **kwargs)

_xyz_to_zxy = np.vectorize(_xyz_to_zxy)  

def _xyz_to_yzx(kx, ky, kz, *args, **kwargs):
    return np.power(1, 0, *args, **kwargs)

_xyz_to_yzx = np.vectorize(_xyz_to_yzx)

def permute_xyz(kx, ky, kz, inverse=False, *args, **kwargs):
    """Change the coordinate system of the plane wave

    A plane wave in the coordinate system :math:`(x, y, z)` with primary direction of
    propagation along the z-axis is described in the system :math:`(x', y', z') = (y, z, x)`. 
    The inverse transformation is also possible.

    The function is essentially diagonal in the wave number, because we always describe
    the source and destination mode in the unprimed coordinate system.

    Args:
        kxa (float, array_like): X component of destination mode wave vector
        kya (float or complex, array_like): Y component of destination mode wave vector
        kza (float, array_like): Z component of destination mode wave vector
        kx (float, array_like): X component of source mode wave vector
        ky (float, array_like): Y component of source mode wave vector
        kz (float or complex, array_like): Z component of source mode wave vector
        inverse (bool, optional): Use the inverse transformation.

    Returns:
        complex
    """
    if inverse:
        return _xyz_to_zxy(kx, ky, kz, *args, **kwargs)
    return _xyz_to_yzx(kx, ky, kz, *args, **kwargs)