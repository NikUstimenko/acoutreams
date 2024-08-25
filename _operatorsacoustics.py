"""Operators for common transformations including different types of waves."""

import numpy as np

import treams.special as sc
from treams import util
from treams._lattice import Lattice, WaveVector
from acoutreams._materialacoustics import AcousticMaterial
import acoutreams._coreacoustics as core
from treams._operators import Operator,FieldOperator
import acoutreams._wavesacoustics as wv
from acoutreams import ssw, spw, scw


def _ssw_rotate(phi, theta, psi, basis, to_basis, where):
    """Rotate scalar spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = ssw.rotate(
        *(m[:, None] for m in to_basis.lm), *basis.lm, phi, theta, psi, where=where
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(res, basis=(to_basis, basis))

def _scw_rotate(phi, basis, to_basis, where):
    """Rotate scalar cylindrical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = scw.rotate(*(m[:, None] for m in to_basis.zm), *basis.zm, phi, where=where)
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(res, basis=(to_basis, basis))


def _spwa_rotate(phi, basis, where):
    """Rotate scalar plane waves (actually rotates the basis)."""
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    kvecs = r @ np.array([basis.qx, basis.qy, basis.qz])
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    newbasis = core.ScalarPlaneWaveBasisByUnitVector(zip(*kvecs))
    if basis.lattice is not None:
        newbasis.lattice = basis.lattice.rotate(phi)
    if basis.kpar is not None:
        newbasis.kpar = basis.kpar.rotate(phi)
    return core.AcousticPhysicsArray(res, basis=(newbasis, basis))


def _spwp_rotate(phi, basis, where):
    """Rotate partial scalar plane waves (actually rotates the basis)."""
    if basis.alignment != "xy":
        raise ValueError(f"rotation on alignment: '{basis.alignment}'")
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1], [s1, c1]])
    kx, ky = basis[()]
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    newbasis = core.ScalarPlaneWaveBasisByComp(zip(*(r @ np.array([kx, ky]))))
    if basis.lattice is not None:
        newbasis.lattice = basis.lattice.rotate(phi)
    if basis.kpar is not None:
        newbasis.kpar = basis.kpar.rotate(phi)
    return core.AcousticPhysicsArray(res, basis=(newbasis, basis))


def rotate(phi, theta=0, psi=0, *, basis, where=True):
    """Rotation matrix.

    For the given Euler angles apply a rotation for the given basis. In some basis sets
    only rotations around the z-axis are permitted.

    Args:
        phi (float): First Euler angle (rotation about z)
        theta (float, optional): Second Euler angle (rotation about y)
        psi (float, optional): Third Euler angle (rotation about z)
        basis (:class:`AcousticBasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        where (array-like, bool, optional): Only evaluate parts of the rotation matrix,
            the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis

    if isinstance(basis, core.ScalarSphericalWaveBasis):
        return _ssw_rotate(phi, theta, psi, basis, to_basis, where)
    if theta != 0:
        raise ValueError("non-zero theta for rotation")
    phi = phi + psi
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        return _scw_rotate(phi, basis, to_basis, where)
    if to_basis != basis:
        raise ValueError("invalid basis")
    if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
        return _spwp_rotate(phi, basis, where)
    if isinstance(basis, core.ScalarPlaneWaveBasisByUnitVector):
        return _spwa_rotate(phi, basis, where)
    raise TypeError("invalid basis")


class AcousticRotate(Operator):
    _FUNC = staticmethod(rotate)

    def __init__(self, phi, theta=0, psi=0, *, isinv=False):
        super().__init__(phi, theta, psi, isinv=isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs and isinstance(kwargs["basis"], tuple):
            kwargs["basis"] = kwargs["basis"][::-1]
        return self.FUNC(*(-a for a in self._args[::-1]), **kwargs)
    

def _ssw_translate(r, basis, to_basis, k0, material, where):
    """Translate scalar spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = k0 * material.nmp
    r = sc.car2sph(r)
    res = ssw.translate(
        *(m[:, None] for m in to_basis.lm),
        *basis.lm,
        ks * r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        singular=False,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=(k0, k0),
        basis=(to_basis, basis),
        material=(material, material),
    )

def _scw_translate(r, basis, k0, to_basis, material, where):
    """Translate scalar cylindrical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = material.ks(k0)
    krhos = np.sqrt(ks * ks - basis.kz * basis.kz + 0j)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    r = sc.car2cyl(r)
    res = scw.translate(
        *(m[:, None] for m in to_basis.zm),
        *basis.zm,
        krhos * r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        singular=False,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res, k0=(k0, k0), basis=(to_basis, basis), material=(material, material)
    )

def _spw_translate(r, basis, k0, to_basis, material, modetype, where):
    """Translate scalar plane waves."""
    kvecs = basis.kvecs(k0, material, modetype)
    kx, ky, kz = to_basis.kvecs(k0, material, modetype)
    where = (
        where
        & (np.abs(kx[:, None] - kvecs[0]) < 1e-14)
        & (np.abs(ky[:, None] - kvecs[1]) < 1e-14)
        & (np.abs(kz[:, None] - kvecs[2]) < 1e-14)
    )
    res = spw.translate(
        *kvecs,
        r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=(k0,) * 2,
        basis=(to_basis, basis),
        material=(material,) * 2,
        modetype=(modetype,) * 2,
    )

def translate(
    r, *, basis, k0=None, material=AcousticMaterial(), modetype=None, where=True
):
    """Translation matrix.

    Translate the given basis modes along the translation vector.

    Args:
        r (array-like): Translation vector
        basis (:class:`AcousticBasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        material (:class:`AcousticMaterial` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode, only used for
            :class:`ScalarPlaneWaveBasisByComp`.
        where (array-like, bool, optional): Only evaluate parts of the translation
            matrix, the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    material = AcousticMaterial(material)

    r = np.asanyarray(r)
    if r.shape[-1] != 3:
        raise ValueError("invalid 'r'")
    if isinstance(basis, core.ScalarPlaneWaveBasis):
        if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _spw_translate(r, basis, k0, to_basis, material, modetype, where)
    if isinstance(basis, core.ScalarSphericalWaveBasis):
        return _ssw_translate(r, basis, to_basis, k0, material, where)
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        return _scw_translate(r, basis, k0, to_basis, material, where)
    raise TypeError("invalid basis")


class AcousticTranslate(Operator):
    """Translation matrix.

    When called as attribute of an object it returns a suitable translation matrix to
    transform it. See also :func:`translate`.
    """

    _FUNC = staticmethod(translate)

    def __init__(self, r, *, isinv=False):
        super().__init__(r, isinv=isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs and isinstance(kwargs["basis"], tuple):
            kwargs["basis"] = kwargs["basis"][::-1]
        return self.FUNC(np.negative(self._args[0]), **kwargs)


def _ssw_ssw_expand(basis, to_basis, to_modetype, k0, material, modetype, where):
    """Expand scalar spherical waves into scalar spherical waves."""
    if not (
        modetype == "regular" == to_modetype
        or modetype == "singular" == to_modetype
        or (modetype == "singular" and to_modetype == "regular")
    ):
        raise ValueError(f"invalid expansion from {modetype} to {to_modetype}")
    rs = sc.car2sph(to_basis.positions[:, None, :] - basis.positions)
    ks = k0 * AcousticMaterial().c / material.c
    res = ssw.translate(
        *(m[:, None] for m in to_basis.lm),
        *basis.lm,
        ks * rs[to_basis.pidx[:, None], basis.pidx, 0],
        rs[to_basis.pidx[:, None], basis.pidx, 1],
        rs[to_basis.pidx[:, None], basis.pidx, 2],
        singular=modetype != to_modetype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    res = core.AcousticPhysicsArray(
        res, k0=k0, basis=(to_basis, basis), material=material
    )
    if modetype == "singular" and to_modetype == "regular":
        res.modetype = (to_modetype, modetype)
    return res

def _ssw_scw_expand(basis, to_basis, k0, material, where):
    """Expand scalar cylindrical waves into scalar spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = material.ks(k0)
    res = scw.to_sw(
        *(m[:, None] for m in to_basis.lm),
        *basis.zm,
        ks,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        modetype=("regular", "regular"),
    )

def _ssw_spw_expand(basis, to_basis, k0, material, modetype, where):
    """Expand scalar plane waves into scalar spherical waves."""
    if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    res = spw.to_ssw(
        *(m[:, None] for m in to_basis.lm),
        *kvecs,
        where=where,
    ) * spw.translate(
        *kvecs,
        to_basis.positions[to_basis.pidx, None, 0],
        to_basis.positions[to_basis.pidx, None, 1],
        to_basis.positions[to_basis.pidx, None, 2],
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        material=material,
        modetype=("regular", modetype),
    )

def _scw_scw_expand(basis, to_basis, to_modetype, k0, material, modetype, where):
    """Expand scalar cylindrical waves into scalar cylindrical waves."""
    if modetype == "regular" == to_modetype or modetype == "singular" == to_modetype:
        modetype = to_modetype = None
    elif modetype != "singular" or to_modetype != "regular":
        raise ValueError(f"invalid expansion from {modetype} to {to_modetype}")
    rs = sc.car2cyl(to_basis.positions[:, None, :] - basis.positions)
    krhos = material.krhos(k0, basis.kz)
    res = scw.translate(
        *(m[:, None] for m in to_basis.zm),
        *basis.zm,
        krhos * rs[to_basis.pidx[:, None], basis.pidx, 0],
        rs[to_basis.pidx[:, None], basis.pidx, 1],
        rs[to_basis.pidx[:, None], basis.pidx, 2],
        singular=modetype != to_modetype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    res = core.AcousticPhysicsArray(res, k0=k0, basis=(to_basis, basis), material=material)
    if modetype == "singular" and to_modetype == "regular":
        res.modetype = (to_modetype, modetype)
    return res

def _scw_spw_expand(basis, to_basis, k0, material, modetype, where):
    """Expand scalar plane waves into scalar cylindrical waves."""
    if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    res = spw.to_scw(
        *(m[:, None] for m in to_basis.zm), *kvecs, where=where
    ) * spw.translate(
        *kvecs,
        to_basis.positions[to_basis.pidx, None, 0],
        to_basis.positions[to_basis.pidx, None, 1],
        to_basis.positions[to_basis.pidx, None, 2],
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        material=material,
        modetype=("regular", modetype),
    )


def _spw_spw_expand(basis, to_basis, k0, material, modetype, where):
    """Expand scalar plane waves into scalar plane waves."""
    if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    if isinstance(to_basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kx, ky, kz = to_basis.kvecs(k0, material, modetype)
    res = np.array(
        where
        & (kx[:, None] == kvecs[0])
        & (ky[:, None] == kvecs[1])
        & (kz[:, None] == kvecs[2]),
        int,
    )
    return core.AcousticPhysicsArray(
        res, basis=(to_basis, basis), k0=k0, material=material, modetype=modetype
    )


def expand(
    basis, modetype=None, *, k0=None, material=AcousticMaterial(), where=True
):
    """Expansion matrix.

    Expand the modes from one basis set to another basis set. If applicable the modetype
    can also be changed, like for spherical and cylindrical waves from `singular` to
    `regular`. Not all expansions are available, only those that result in a discrete
    set of modes. For example, plane waves can be expanded in spherical waves, but the
    opposite transformation generally requires a continuous spectrum (an integral) over
    plane waves.

    Args:
        basis (:class:`AcousticBasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        modetype (str, optional): Wave mode, used for
            :class:`ScalarSphericalWaveBasis` (and :class:`ScalarCylindricalWaveBasis`).
        k0 (float, optional): Wave number.
        material (:class:`AcousticMaterial` or tuple, optional): Material parameters.
        where (array-like, bool, optional): Only evaluate parts of the expansion matrix,
            the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    if isinstance(modetype, (tuple, list)):
        to_modetype, modetype = modetype
    else:
        to_modetype = None
    material = AcousticMaterial(material)
    if isinstance(basis, core.ScalarSphericalWaveBasis) and isinstance(
        to_basis, core.ScalarSphericalWaveBasis
    ):
        modetype = "regular" if modetype is None else modetype
        to_modetype = modetype if to_modetype is None else to_modetype
        return _ssw_ssw_expand(
            basis, to_basis, to_modetype, k0, material, modetype, where
        )
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        if isinstance(to_basis, core.ScalarCylindricalWaveBasis):
            modetype = "regular" if modetype is None else modetype
            to_modetype = modetype if to_modetype is None else to_modetype
            return _scw_scw_expand(
                basis, to_basis, to_modetype, k0, material, modetype, where
            )
        if isinstance(to_basis, core.ScalarSphericalWaveBasis):
            if modetype != "regular" and to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _ssw_scw_expand(basis, to_basis, k0, material, where)
    if isinstance(basis, core.ScalarPlaneWaveBasis):
        if isinstance(to_basis, core.ScalarPlaneWaveBasis):
            to_modetype = modetype if to_modetype is None else to_modetype
            return _spw_spw_expand(basis, to_basis, k0, material, modetype, where)
        if isinstance(to_basis, core.ScalarCylindricalWaveBasis):
            if to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _scw_spw_expand(basis, to_basis, k0, material, modetype, where)
        if isinstance(to_basis, core.ScalarSphericalWaveBasis):
            if to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _ssw_spw_expand(
                basis, to_basis, k0, material, modetype, where
            )
    raise TypeError("invalid basis")


class AcousticExpand(Operator):
    """Expansion matrix.

    When called as attribute of an object it returns a suitable transformation matrix to
    expand one set of modes into another basis set (and mode type, if applicable).
    See also :func:`expand`.
    """

    _FUNC = staticmethod(expand)

    def __init__(self, basis, modetype=None, *, isinv=False):
        args = (basis,) if modetype is None else (basis, modetype)
        super().__init__(*args, isinv=isinv)

    def __call__(self, **kwargs):
        if self.isinv:
            return self._call_inv(**kwargs)
        args = list(self._args)
        if "basis" in kwargs and not isinstance(args[0], tuple):
            args[0] = (args[0], kwargs.pop("basis"))
        if "modetype" in kwargs and len(args) > 1 and not isinstance(args[1], tuple):
            args[1] = (args[1], kwargs.pop("modetype"))
        return self.FUNC(*args, **kwargs)

    def _call_inv(self, **kwargs):
        args = list(self._args)
        if "basis" in kwargs and not isinstance(args[0], tuple):
            args[0] = (kwargs.pop("basis"), args[0])
        if "modetype" in kwargs and len(args) > 1 and not isinstance(args[1], tuple):
            args[1] = (kwargs.pop("modetype"), args[1])
        return self.FUNC(*args, **kwargs)

    def get_kwargs(self, obj, dim=-1):
        kwargs = super().get_kwargs(obj, dim)
        for name in ("basis", "modetype"):
            val = getattr(obj, name, None)
            if isinstance(val, tuple):
                val = val[dim]
            if val is not None:
                kwargs[name] = val
        return kwargs

    @property
    def inv(self):
        """Inverse expansion.

        The inverse transformation is not available for all transformations.
        """
        if len(self._args) == 1:
            basis, modetype = self._args[0], None
        else:
            basis, modetype = self._args
        if isinstance(basis, tuple):
            basis = tuple(basis[::-1])
        if isinstance(modetype, tuple):
            modetype = tuple(modetype[::-1])
        return type(self)(basis, modetype, isinv=not self.isinv)
    

def _sswl_expand(basis, to_basis, eta, k0, kpar, lattice, material, where):
    """Expand scalar spherical waves in a lattice."""
    ks = k0 * AcousticMaterial().c / material.c
    if lattice.dim == 3:
        try:
            length = len(kpar)
        except TypeError:
            length = 1
        if length == 1:
            lattice = Lattice(lattice, "z")
        elif length == 2:
            lattice = Lattice(lattice, "xy")
        elif length == 3:
            # Last attempt to determine the dimension of the sum
            if np.isnan(kpar[2]):
                lattice = Lattice(lattice, "xy")
            elif np.isnan(kpar[0]) and np.isnan(kpar[1]):
                lattice = Lattice(lattice, "z")
    kpar = WaveVector(kpar)
    if lattice.dim == 1:
        x = kpar[2]
    elif lattice.dim == 2:
        x = kpar[:2]
    else:
        x = kpar[:]
    res = ssw.translate_periodic(
        ks,
        x,
        lattice[...],
        to_basis.positions,
        to_basis[()],
        basis[()],
        basis.positions,
        eta=eta,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        lattice=lattice,
        modetype=("regular", "singular"),
        kpar=kpar,
    )


def _scw_ssw_expand(basis, to_basis, k0, kpar, lattice, material, where):
    """Expand scalar spherical waves into scalar cylindrical waves in a lattice."""
    ks = k0 * AcousticMaterial().c / material.c
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    kpar = WaveVector(kpar)
    res = ssw.periodic_to_scw(
        *(m[:, None] for m in to_basis.zm),
        *basis.lm,
        ks,
        Lattice(lattice, "z").volume,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        modetype="singular",
        lattice=lattice,
        kpar=kpar,
    )


def _spw_ssw_expand(
    basis, to_basis, k0, kpar, lattice, material, modetype, where
):
    """Expand scalar spherical waves into scalar plane waves in a lattice."""
    if modetype is None and isinstance(to_basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up"
    kpar = WaveVector(kpar)
    kvecs = to_basis.kvecs(k0, material, modetype)
    transl = spw.translate(
        *(b[:, None] for b in kvecs),
        -basis.positions[:, 0],
        -basis.positions[:, 1],
        -basis.positions[:, 2],
    )
    res = transl[:, basis.pidx] * ssw.periodic_to_spw(
        *(b[:, None] for b in kvecs),
        *basis.lm,
        Lattice(lattice, "xy").volume,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        kpar=kpar,
        lattice=lattice,
        material=material,
        modetype=(modetype, "singular"),
    )


def _scwl_expand(basis, to_basis, eta, k0, kpar, lattice, material, where):
    """Expand scalar cylindrical waves in a lattice."""
    ks = material.ks(k0)
    alignment = (
        "x" if not isinstance(lattice, Lattice) and np.size(lattice) == 1 else None
    )
    lattice = Lattice(lattice, alignment)
    if lattice.dim == 3:
        try:
            length = len(kpar)
        except TypeError:
            length = 1
        if length == 1:
            lattice = Lattice(lattice, "x")
        elif length == 2:
            lattice = Lattice(lattice, "xy")
        elif length == 3:
            # Last attempt to determine the dimension of the sum
            if np.isnan(kpar[1]):
                lattice = Lattice(lattice, "x")
                kpar = WaveVector(kpar, "x")
            else:
                lattice = Lattice(lattice, "xy")
    if lattice.dim == 1:
        kpar = WaveVector(kpar, "x")
        x = kpar[0]
    elif lattice.dim == 2:
        kpar = WaveVector(kpar)
        x = kpar[:2]
    res = scw.translate_periodic(
        ks,
        x,
        lattice[...],
        to_basis.positions,
        to_basis[()],
        basis[()],
        basis.positions,
        eta=eta,
    )
    res[..., np.logical_not(where)] = 0
    for b in (to_basis, basis):
        if b.kpar is not None:
            kpar = kpar & b.kpar
    return core.AcousticPhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        lattice=lattice,
        modetype=("regular", "singular"),
        kpar=kpar,
    )



def _spw_scw_expand(basis, to_basis, k0, lattice, kpar, material, modetype, where):
    """Expand scalar cylindrical waves into scalar plane waves in a lattice."""
    if modetype is None and isinstance(to_basis, core.ScalarPlaneWaveBasisByComp):
        modetype = "up"
    if len(kpar) == 1:
        kpar = WaveVector(kpar, alignment="x")
    kvecs = to_basis.kvecs(k0, material, modetype)
    transl = spw.translate(
        *(b[:, None] for b in kvecs),
        -basis.positions[:, 0],
        -basis.positions[:, 1],
        -basis.positions[:, 2],
    )
    res = transl[:, basis.pidx] * scw.periodic_to_spw(
        *(b[:, None] for b in kvecs),
        *basis.zm,
        lattice.volume,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        modetype=(modetype, "singular"),
        lattice=lattice,
        kpar=kpar,
    )

def expandlattice(
    lattice=None,
    kpar=None,
    basis=None,
    modetype=None,
    *,
    eta=0,
    k0=None,
    material=AcousticMaterial(),
    where=True,
):
    """Expansion matrix in lattices.

    Expand the modes from one basis set which are assumed to be periodically repeated on
    a lattice into another basis set.

    Args:
        lattice (:class:`~treams.Lattice` or array-like, optional): Lattice definition.
            In some cases this argument can be omitted, when the lattice can be inferred
            from the basis.
        kpar (sequence, optional): The components of the wave vector tangential to the
            lattice. In some cases this argument can be omitted, when the lattice can be
            inferred from the basis.
        basis (:class:`~AcousticBasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        eta (float or complex, optional): Split parameter used when the Ewald summation
            is applied for the lattice sums. By setting it to 0 the split is set
            automatically.
        material (:class:`~AcousticMaterial` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode, used for
            :class:`ScalarPlaneWaveBasisByComp`.
        where (array-like, bool, optional): Only evaluate parts of the expansion matrix,
            the give array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    if lattice is None and to_basis.lattice is not None:
        lattice = to_basis.lattice
    if not isinstance(lattice, Lattice) and np.size(lattice) == 1:
        alignment = "x" if isinstance(basis, core.ScalarCylindricalWaveBasis) else "z"
    elif isinstance(to_basis, core.ScalarPlaneWaveBasis) and isinstance(
        basis, core.ScalarCylindricalWaveBasis
    ):
        alignment = "x"
    else:
        alignment = None
    lattice = Lattice(lattice, alignment)
    if kpar is None:
        if basis.kpar is None:
            kpar = to_basis.kpar
        else:
            kpar = basis.kpar
    try:
        kpar = list(kpar)
    except TypeError:
        if kpar is None:
            kpar = [np.nan] * 3
        else:
            kpar = [kpar]
    material = AcousticMaterial(material)
    if isinstance(basis, core.ScalarSphericalWaveBasis):
        if isinstance(to_basis, core.ScalarSphericalWaveBasis):
            return _sswl_expand(
                basis, to_basis, eta, k0, kpar, lattice, material, where
            )
        if isinstance(to_basis, core.ScalarCylindricalWaveBasis):
            return _scw_ssw_expand(
                basis, to_basis, k0, kpar, lattice, material, where
            )
        if isinstance(to_basis, core.ScalarPlaneWaveBasis):
            if isinstance(modetype, tuple):
                modetype = modetype[0]
            return _spw_ssw_expand(
                basis, to_basis, k0, kpar, lattice, material, modetype, where
            )
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        if isinstance(to_basis, core.ScalarCylindricalWaveBasis):
            return _scwl_expand(basis, to_basis, eta, k0, kpar, lattice, material, where)
        if isinstance(to_basis, core.ScalarPlaneWaveBasis):
            if isinstance(modetype, tuple):
                modetype = modetype[0]
            return _spw_scw_expand(
                basis, to_basis, k0, lattice, kpar, material, modetype, where
            )
    raise TypeError("invalid basis")


class AcousticExpandLattice(Operator):
    """Expansion matrix in a lattice.

    When called as attribute of an object it returns a suitable transformation matrix to
    expand one set of modes that is periodically repeated into another basis set.
    See also :func:`expandlattice`.
    """

    _FUNC = staticmethod(expandlattice)

    def __init__(self, lattice=None, kpar=None, basis=None, modetype=None, eta=0):
        super().__init__(lattice, kpar, basis, modetype, eta)

    def __call__(self, **kwargs):
        if self.isinv:
            return self._call_inv(**kwargs)
        args = list(self._args)
        if "basis" in kwargs:
            if args[2] is None:
                args[2] = kwargs.pop("basis")
            else:
                args[2] = (args[2], kwargs.pop("basis"))
        if "modetype" in kwargs and args[3] is None:
            args[3] = kwargs.pop("modetype")
        for i, name in enumerate(("lattice", "kpar")):
            if name in kwargs and args[i] is None:
                args[i] = kwargs.pop(name)
        if args[4] != 0:
            kwargs["eta"] = args[4]
        args = args[:4]
        return self.FUNC(*args, **kwargs)

    def get_kwargs(self, obj, dim=-1):
        kwargs = super().get_kwargs(obj, dim)
        kwargs.pop("modetype", None)
        for name in ("basis", "lattice", "kpar"):
            val = getattr(obj, name, None)
            if isinstance(val, tuple):
                val = val[dim]
            if val is not None:
                kwargs[name] = val
        return kwargs

    @property
    def inv(self):
        """Inverse expansion for periodic arrangements.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _spwp_permute(basis, n, k0, material, modetype):
    """Permute axes in a scalar partial plane wave basis."""
    if material is None:
        raise TypeError("missing definition of 'material'")
    modetype = "up" if modetype is None else modetype
    kvecs = np.array(basis.kvecs(k0, material, modetype))
    where = (kvecs[..., None] == kvecs[:, None, :]).all(0)
    if n == 1:
        res = spw.permute_xyz(
            kvecs[0],
            kvecs[1],
            kvecs[2],
            where=where,
        )
    elif n == 2:
        res = spw.permute_xyz(
            kvecs[0],
            kvecs[1],
            kvecs[2],
            inverse=True,
            where=where,
        )
    else:
        res = np.eye(len(basis))
    res[np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(res, basis=(basis.permute(n), basis))    


def _spwa_permute(basis, n):
    """Permute axes in a scalar plane wave basis."""
    qx, qy, qz = basis.qx, basis.qy, basis.qz
    where = (qx[:, None] == qx) & (qy[:, None] == qy) & (qz[:, None] == qz)
    if n == 1:
        res = spw.permute_xyz(
            qx, qy, qz, where=where
        )
    elif n == 2:
        res = spw.permute_xyz(
            qx,
            qy,
            qz,
            inverse=True,
            where=where,
        )
    elif n == 0:
        res = np.eye(len(basis))
    res[np.logical_not(where)] = 0
    return core.AcousticPhysicsArray(res, basis=(basis.permute(n), basis))


def permute(n=1, *, basis, k0=None, material=None, modetype=None):
    """Permutation matrix.

    Permute the axes of a plane wave basis expansion.

    Args:
        n (int, optional): Number of permutations, defaults to 1.
        basis (:class:`~AcousticBasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        material (:class:`~AcousticMaterial` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
    """
    if n != int(n):
        raise ValueError("'n' must be integer")
    n = n % 3
    if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
        return _spwp_permute(basis, n, k0, material, modetype)
    if isinstance(basis, core.ScalarPlaneWaveBasisByUnitVector):
        return _spwa_permute(basis, n)
    raise TypeError("invalid basis")


class AcousticPermute(Operator):
    """Axes permutation matrix.

    When called as attribute of an object it returns a suitable transformation matrix to
    permute the axis definitions of plane waves. See also :func:`permute`.
    """

    _FUNC = staticmethod(permute)

    def __init__(self, n=1, *, isinv=False):
        super().__init__(n, isinv=isinv)

    @property
    def inv(self):
        return type(self)(*self._args, isinv=not self.isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs:
            kwargs["basis"] = kwargs["basis"].permute(self._args[0])
        return self.FUNC(-self._args[0], **kwargs)
    


def _ssw_vfield(r, basis, k0, material, modetype):
    """Velocity field of scalar spherical waves."""
    ks = k0 * AcousticMaterial().c / material.c
    rsph = sc.car2sph(r - basis.positions)
    res = None
    if modetype == "regular":
        res = wv.vsw_rL(
            basis.l,
            basis.m,
            ks * rsph[..., basis.pidx, 0],
            rsph[..., basis.pidx, 1],
            rsph[..., basis.pidx, 2],
        )
    elif modetype == "singular":
        res = wv.vsw_L(
            basis.l,
            basis.m,
            ks * rsph[..., basis.pidx, 0],
            rsph[..., basis.pidx, 1],
            rsph[..., basis.pidx, 2],
        )
    res *= -1j / (material.rho * material.c)
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vsph2car(res, rsph[..., basis.pidx, :]))    
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def _scw_vfield(r, basis, k0, material, modetype):
    """Velocity field of scalar cylindrical waves."""
    material = AcousticMaterial(material)
    ks = material.ks(k0)
    krhos = material.krhos(k0, basis.kz)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    rcyl = sc.car2cyl(r - basis.positions)
    if modetype == "regular":
        res = wv.vcw_rL(
            basis.kz,
            basis.m,
            krhos * rcyl[..., basis.pidx, 0],
            rcyl[..., basis.pidx, 1],
            rcyl[..., basis.pidx, 2],
            ks,
            krhos,
        )
    elif modetype == "singular":
        res = wv.vcw_L(
            basis.kz,
            basis.m,
            krhos * rcyl[..., basis.pidx, 0],
            rcyl[..., basis.pidx, 1],
            rcyl[..., basis.pidx, 2],
            ks,
            krhos,
        )   
    res *= -1j / (material.rho * material.c)
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vcyl2car(res, rcyl[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def _spw_vfield(r, basis, k0, material, modetype):
    """Velocity field of scalar plane waves."""
    res = None
    kvecs = basis.kvecs(k0, material, modetype)
    res = wv.vpw_L(*kvecs, r[..., 0], r[..., 1], r[..., 2])
    res *= -1j / (material.rho * material.c)
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["material"] = material
    return res


def vfield(r, *, basis, k0, material=AcousticMaterial(), modetype=None):
    r"""Velocity field.

    The resulting matrix maps the pressure field coefficients of the given basis to the
    velocity field in Cartesian coordinates.

    The velocity field is given in units of :math:`\frac{1}{rho c} p`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~AcousticBasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~AcousticMaterial` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
    """
    material = AcousticMaterial(material)
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.ScalarSphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _ssw_vfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _scw_vfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    if isinstance(basis, core.ScalarPlaneWaveBasis):
        if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _spw_vfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class VField(FieldOperator):
    """Velocity field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`vfield`.
    """

    _FUNC = staticmethod(vfield)


def _ssw_pfield(r, basis, k0, material, modetype):
    """Pressure field of scalar spherical waves."""
    ks = k0 * AcousticMaterial().c / material.c
    rsph = sc.car2sph(r - basis.positions)
    res = None
    if modetype == "regular":
        res = wv.ssw_rPsi(
            basis.l,
            basis.m,
            ks * rsph[..., basis.pidx, 0],
            rsph[..., basis.pidx, 1],
            rsph[..., basis.pidx, 2],
            )
    elif modetype == "singular":
            res = wv.ssw_Psi(
            basis.l,
            basis.m,
            ks * rsph[..., basis.pidx, 0],
            rsph[..., basis.pidx, 1],
            rsph[..., basis.pidx, 2],
        )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(np.array([res]).T)  
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res

def _scw_pfield(r, basis, k0, material, modetype):
    """Pressure field of scalar cylindrical waves."""
    material = AcousticMaterial(material)
    ks = material.ks(k0)
    krhos = material.krhos(k0, basis.kz)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    rcyl = sc.car2cyl(r - basis.positions)
    if modetype == "regular":
        res = wv.scw_rPsi(
            basis.kz,
            basis.m,
            krhos * rcyl[..., basis.pidx, 0],
            rcyl[..., basis.pidx, 1],
            rcyl[..., basis.pidx, 2],
        )
    elif modetype == "singular":
        res = wv.scw_Psi(
            basis.kz,
            basis.m,
            krhos * rcyl[..., basis.pidx, 0],
            rcyl[..., basis.pidx, 1],
            rcyl[..., basis.pidx, 2]
        )   
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(np.array([res]).T)  
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res

def _spw_pfield(r, basis, k0, material, modetype):
    """Pressure field of scalar plane waves."""
    res = None
    kvecs = basis.kvecs(k0, material, modetype)
    res = wv.spw_Psi(*kvecs, r[..., 0], r[..., 1], r[..., 2])
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(np.array([res]).T)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    return res


def pfield(r, *, basis, k0, material=AcousticMaterial(), modetype=None):
    """Pressure field.

    The resulting matrix maps the pressure field coefficients of the given basis to the
    pressure field in Cartesian coordinates.

    Args:
        r (array-like): Evaluation points
        basis (:class:`AcousticBasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`AcousticMaterial` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
    """
    material = AcousticMaterial(material)
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.ScalarSphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _ssw_pfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    if isinstance(basis, core.ScalarCylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _scw_pfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    if isinstance(basis, core.ScalarPlaneWaveBasis):
        if isinstance(basis, core.ScalarPlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _spw_pfield(r, basis, k0, material, modetype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class PField(FieldOperator):
    """Pressure field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`pfield`.
    """

    _FUNC = staticmethod(pfield)


