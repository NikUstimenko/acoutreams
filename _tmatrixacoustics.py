import warnings
import numpy as np
from treams.util import AnnotationError
from acoutreams._coreacoustics import ScalarSphericalWaveBasis as SSWB
from acoutreams._coreacoustics import ScalarPlaneWaveBasisByComp as SPWBC
from acoutreams._coreacoustics import ScalarCylindricalWaveBasis as SCWB
from acoutreams._coreacoustics import ScalarPlaneWaveBasisByUnitVector as SPWBUV
from acoutreams._materialacoustics import AcousticMaterial
from acoutreams.coeffsacoustics import mie_acoustics, mie_acoustics_cyl
from acoutreams. _coreacoustics import AcousticPhysicsArray
import acoutreams._operatorsacoustics as opa
import acoutreams._wavesacoustics as wv


class _AcousticInteraction:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype
        return self

    def __call__(self):
        basis = self._obj.basis
        return (
             np.eye(self._obj.shape[-1]) - self._obj @ opa.AcousticExpand(basis, "singular").inv
        )

    def solve(self):
        return np.linalg.solve(self(), self._obj)


class _AcousticInteractionApprox:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype
        return self

    def __call__(self):
        basis = self._obj.basis
        return self._obj @ opa.AcousticExpand(basis, "singular").inv
    
    def solve(self, order, error=np.inf):
        order = int(order)
        error = np.float64(error)
        if order < 0:
            raise TypeError("order must be non-negative")
        if error <= 0.:
            raise TypeError("error tolerance must be positive")
        coupling = self()
        if np.abs(coupling).max() >= 1:
            warnings.warn("the series diverges")
        dim = self._obj.shape[-1]
        identity = np.eye(dim)
        res = np.zeros((dim, dim), np.complex128) + identity
        res_prev = res
        for n in range(1, order + 1):
            res_prev = res
            res = res @ coupling + identity
        res = res @ self._obj
        res_prev = res_prev @ self._obj
        error_curr = np.linalg.norm(np.array(res - res_prev), np.inf)
        if error < np.inf and error_curr > error and order >= 1:
            warnings.warn("given order is not enough to solve with given error tolerance")
        return [res, error_curr]
   
    def solve2(self, error, order_max=np.inf):
        if order_max < np.inf:
            order_max = int(order_max)
        error = np.float64(error)
        if error <= 0.:
            raise TypeError("error tolerance must be positive")
        if order_max < 1:
            raise TypeError("maximal order must positive")
        dim = self._obj.shape[-1]
        res = np.zeros((dim, dim), np.complex128) + np.eye(dim)
        coupling = self()
        if np.abs(coupling).max() >= 1:
            warnings.warn("the series diverges")
        error_curr = np.inf
        order_curr = 0
        while error_curr > error and order_curr < order_max:
            res_prev = res
            res = res @ coupling + np.eye(dim)
            error_curr = np.max(np.abs((res - res_prev) @ self._obj))
            order_curr += 1
        res = res @ self._obj
        if error_curr > error and order_curr >= order_max:
            raise TypeError("given maximal order is not enough to solve with given error tolerance")
        return [res, order_curr]
    

class _AcousticLatticeInteraction:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype
        return self

    def __call__(self, lattice, kpar, *, eta=0):
        return np.eye(self._obj.shape[-1]) - self._obj @ opa.AcousticExpandLattice(
            lattice=lattice, kpar=kpar, eta=eta
        )

    def solve(self, lattice, kpar, *, eta=0):
        return np.linalg.solve(self(lattice, kpar, eta=eta), self._obj)
    

class _AcousticLatticeInteractionApprox:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype
        return self

    def __call__(self, lattice, kpar, *, eta=0):
        basis = self._obj.basis
        return self._obj @ opa.AcousticExpandLattice(
            lattice=lattice, kpar=kpar, eta=eta
        )
    
    def solve(self, lattice, kpar, *, eta=0, order, error=np.inf):
        order = int(order)
        error = np.float64(error)
        if order < 0:
            raise TypeError("order must be non-negative")
        if error <= 0:
            raise TypeError("error tolerance must be positive")
        coupling = self(lattice, kpar, eta=eta)
        if np.abs(coupling).max() >= 1:
            warnings.warn("the series diverges")
        dim = self._obj.shape[-1]
        identity = np.eye(dim)
        res = np.zeros((dim, dim), np.complex128) + identity
        res_prev = res
        for n in range(1, order + 1):
            res_prev = res
            res = res @ coupling + identity
        res = res @ self._obj
        res_prev = res_prev @ self._obj
        error_curr = np.linalg.norm(np.array(res - res_prev), np.inf)
        if error < np.inf and error_curr > error and order >= 1:
            warnings.warn("given order is not enough to solve with given error tolerance")
        return [res, error_curr]
    
    def solve2(self, error, lattice, kpar, *, eta=0, order_max=np.inf):
        if order_max < np.inf:
            order_max = int(order_max)
        error = np.float64(error)
        if error <= 0:
            raise TypeError("error tolerance must be positive")
        if order_max < 1:
            raise TypeError("maximal order must positive")
        dim = self._obj.shape[-1]
        res = np.zeros((dim, dim), np.complex128) + np.eye(dim)
        coupling = self(lattice, kpar, eta=eta)
        if np.abs(coupling).max() >= 1:
            warnings.warn("the series diverges")
        error_curr = np.inf
        order_curr = 0
        while error_curr > error and order_curr <= order_max:
            res_prev = res
            res = res @ coupling + np.eye(dim)
            error_curr = np.max(np.abs((res - res_prev) @ self._obj))
            order_curr += 1
        res = res @ self._obj
        if error_curr > error and order_curr > order_max:
            raise TypeError("given maximal order is not enough to solve with given error tolerance")
        return [res, order_curr]


class AcousticTMatrix(AcousticPhysicsArray):
    """Acoustic T-matrix with a spherical basis.

    The acoustic T-matrix is a square relating incident (regular) fields
    func:`ssw_rPsi` to the corresponding scattered fields :func:`ssw_Psi`. 
    The modes themselves are defined in :attr:`basis`. 
    Moreover, the wave number :attr:`k0` and, if not air, the material :attr:`material` are
    specified.

    Args:
        arr (float or complex, array-like): T-matrix itself.
        k0 (float): Wave number in air.
        basis (ScalarSphericalWaveBasis, optional): Basis definition.
        material (AcousticMaterial, optional): Embedding material, defaults to air.
        lattice (Lattice, optional): Lattice definition. If specified the T-Matrix is
            assumed to be periodically repeated in the defined lattice.
        kpar (list, optional): Phase factor for the periodic T-Matrix.
    """

    interaction = _AcousticInteraction()
    interactionapprox = _AcousticInteractionApprox()
    latticeinteraction = _AcousticLatticeInteraction()
    latticeinteractionapprox = _AcousticLatticeInteractionApprox()
    
    def _check(self):
        """Fill in default values or raise errors for missing attributes."""
        super()._check()
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise AnnotationError("invalid k0")
        modetype = self.modetype
        if modetype is None or (
            modetype[0] in (None, "singular") and modetype[1] in (None, "regular")
        ):
            self.modetype = ("singular", "regular")
        else:
            raise AnnotationError("invalid modetype")
        if self.basis is None:
            self.basis = SSWB.default(SSWB.defaultlmax(shape[0]))
        if self.material is None:
            self.material = AcousticMaterial()

    @property
    def ks(self):
        """Wave numbers (in medium)."""
        return self.material.ks(self.k0)

    @classmethod
    def sphere(cls, lmax, k0, radii, materials):
        """T-Matrix of a (multi-layered) sphere.

        Construct the T-matrix of the given order and material for a sphere. (The object
        can also consist of multiple concentric spherical shells with an arbitrary
        number of layers.)            

        Args:
            lmax (int): Non-negative integer for the maximum degree of the T-matrix.
            k0 (float): Wave number in air.
            radii (float or array): Radii from inside to outside of the sphere. For a
                simple sphere the radius can be given as a single number, for a multi-
                layered sphere it is a list of increasing radii for all shells.
            material (list[Material]): The material parameters from the inside to the
                outside. The last material in the list specifies the embedding medium.

        Returns:
            AcousticTMatrix
        """
        materials = [AcousticMaterial(m) for m in materials]
        if materials[-1].isshear:
            raise NotImplementedError
        radii = np.atleast_1d(radii)
        if radii.size != len(materials) - 1:
            raise ValueError("incompatible lengths of radii and materials")
        dim = SSWB.defaultdim(lmax)
        tmat = np.zeros((dim, dim), np.complex128)
        for l in range(lmax + 1):  # noqa: E741
            miecoeffs = mie_acoustics(l, k0 * radii, *zip(*materials))
            pos = SSWB.defaultdim(l - 1)                                                          #!!!!!!!!!!
            for i in range(2 * l + 1):
                tmat[pos + i, pos + i] = miecoeffs[i]
        return cls(tmat, k0=k0, basis=SSWB.default(lmax), material=materials[-1])
    
    @classmethod
    def cluster(cls, tmats, positions):
        r"""Block-diagonal T-matrix of multiple objects.

        Construct the initial block-diagonal T-matrix for a cluster of objects. The
        T-matrices in the list are placed together into a block-diagonal matrix and the
        complete (local) basis is defined based on the individual T-matrices and their
        bases together with the defined positions. In mathematical terms the matrix

        .. math::

            \begin{pmatrix}
                T_0 & 0 & \dots & 0 \\
                0 & T_1 & \ddots & \vdots \\
                \vdots & \ddots & \ddots & 0 \\
                0 & \dots & 0 & T_{N-1} \\
            \end{pmatrix}

        is created from the list of T-matrices :math:`(T_0, \dots, T_{N-1})`. Only
        T-matrices of the same wave number, and embedding material.

        Args:
            tmats (Sequence): List of T-matrices.
            positions (array): The positions of all individual objects in the cluster.

        Returns:
            AcousticTMatrix
        """
        for tm in tmats:
            if not tm.basis.isglobal:
                raise ValueError("global basis required")
        positions = np.array(positions)
        if len(tmats) < positions.shape[0]:
            warnings.warn("specified more positions than T-matrices")
        elif len(tmats) > positions.shape[0]:
            raise ValueError(
                f"'{len(tmats)}' T-matrices "
                f"but only '{positions.shape[0]}' positions given"
            )
        mat = tmats[0].material
        k0 = tmats[0].k0
        modes = [], []
        pidx = []
        dim = sum(tmat.shape[0] for tmat in tmats)
        tres = np.zeros((dim, dim), complex)
        i = 0
        for j, tm in enumerate(tmats):
            if tm.material != mat:
                raise ValueError(f"incompatible materials: '{mat}' and '{tm.material}'")
            if tm.k0 != k0:
                raise ValueError(f"incompatible k0: '{k0}' and '{tm.k0}'")
            dim = tm.shape[0]
            for m, n in zip(modes, tm.basis.lm):
                m.extend(list(n))
            pidx += [j] * dim
            tres[i : i + dim, i : i + dim] = tm
            i += dim
        basis = SSWB(zip(pidx, *modes), positions)
        return cls(tres, k0=k0, material=mat, basis=basis)
    
    @property
    def isglobal(self):
        """Test if a T-matrix is global.

        A T-matrix is considered global, when its basis refers to only a single point
        and it is not placed periodically in a lattice.
        """
        return self.basis.isglobal and self.lattice is None and self.kpar is None

    @property
    def xs_ext_avg(self):
        r"""Rotation and polarization averaged extinction cross section.

        The average is calculated as

        .. math::

            \langle \sigma_\mathrm{ext} \rangle
            = 2 \pi \sum_{lm} \frac{\Re(T_{lm,lm})}{k^2},

        where :math:`k` is the wave number in the embedding medium.    

        It is only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        k = self.ks
        res = -4 * np.pi * self.trace().real / (k * k)
        if res.imag == 0:
            return res.real
        return res

    @property
    def xs_sca_avg(self):
        r"""Rotation and polarization averaged scattering cross section.

        The average is calculated as

        .. math::

            \langle \sigma_\mathrm{sca} \rangle
            = 2 \pi \sum_{lm} \sum_{l'm'}
            \frac{|T_{lm,l'm'}|^2}{k^2}

        where :math:`k` is the wave number in the embedding medium. 
        It is only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        k = self.ks
        re, im = self.real, self.imag
        res = 4 * np.pi * np.sum((re * re + im * im) / (k * k))
        return res.real
    

    def xs(self, illu, flux=0.5):
        r"""Scattering and extinction cross section.

        Possible for all T-matrices (global and local) in non-absorbing embedding. The
        values are calculated by

        .. math::

            \sigma_\mathrm{sca}
            = \frac{1}{2 I}
            a_{lm}^\ast T_{l'm',lm}^\ast k^{-2} C_{l'm',l''m''}^{(1)}
            T_{l''m'',l'''m'''} a_{l'''m'''} \\
            \sigma_\mathrm{ext}
            = \frac{1}{2 I}
            a_{lm}^\ast k^{-2} T_{lm,l'm'} a_{l'm'}

        where :math:`a_{lm}` are the expansion coefficients of the illumination,
        :math:`T` is the T-matrix, :math:`C^{(1)}` is the (regular) translation
        matrix and :math:`k` is the wave number in the medium. All repeated indices
        are summed over. The incoming flux is :math:`I`.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux corresponding to the illumination. Used for
                the result's normalization. The flux is given in units of
                :math:`\frac{\text{V}^2}{{l^2}} \frac{1}{Z_0 Z}` where :math:`l` is the
                unit of length used in the wave number (and positions). A plane wave
                has the flux `0.5` in this normalization, which is used as default.

        Returns:
            tuple[float]
        """
        if not self.material.isreal:
            raise NotImplementedError
        illu = AcousticPhysicsArray(illu)
        illu_basis = illu.basis
        illu_basis = illu_basis[-2] if isinstance(illu_basis, tuple) else illu_basis
        if not isinstance(illu_basis, SSWB):
            illu = illu.expand(self.basis)
        p = self @ illu
        p_invksq = p * np.power(self.ks, -2)
        del illu.modetype
        return (
            0.5 * np.real(p.conjugate().T @ p_invksq.expand(p.basis)) / flux,
            -0.5 * np.real(illu.conjugate().T @ p_invksq) / flux,
        )
    
    def valid_points(self, grid, radii):
        """Points on the grid where the expansion is valid.

        The expansion of the acoustic wave fields is valid outside of the
        circumscribing spheres of each object. From a given set of coordinates mark
        those that are outside of the given radii.

        Args:
            grid (array-like): Points to assess. The last dimension needs length three
                and corresponds to the Cartesian coordinates.
            radii (Sequence[float]): Radii of the circumscribing spheres. Each radius
                corresponds to a position of the basis.

        Returns:
            array
        """
        grid = np.asarray(grid)
        if grid.shape[-1] != 3:
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones(grid.shape[:-1], bool)
        for r, p in zip(radii, self.basis.positions):
            res &= np.sum(np.power(grid - p, 2), axis=-1) > r * r
        return res
     
    def __getitem__(self, key):
        if isinstance(key, SSWB):
            key = np.array([self.basis.index(i) for i in key])
            key = (key[:, None], key)
        return super().__getitem__(key)
    

class AcousticTMatrixC(AcousticPhysicsArray):
    """T-matrix with a scalar cylindrical basis.

    The acoustic T-matrix is square relating incident (regular) fields
    :func:`scw_rPsi` to the scattered fields :func:`scw_Psi`. The modes themselves
    are defined in :attr:`basis`. Also, the wave number :attr:`k0` and, if not air, 
    the material :attr:`material` are specified.

    Args:
        arr (float or complex, array-like): T-matrix itself.
        k0 (float): Wave number in vacuum.
        basis (ScalarCylindricalWaveBasis, optional): Basis definition.
        material (AcousticMaterial, optional): Embedding material, defaults to air.
        lattice (Lattice, optional): Lattice definition. If specified the T-Matrix is
            assumed to be periodically repeated in the defined lattice.
        kpar (list, optional): Phase factor for the periodic T-Matrix.
    """

    interaction = _AcousticInteraction()
    latticeinteraction = _AcousticLatticeInteraction()

    def _check(self):
        """Fill in default values or raise errors for missing attributes."""
        super()._check()
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise AnnotationError("invalid k0")
        modetype = self.modetype
        if modetype is None or (
            modetype[0] in (None, "singular") and modetype[1] in (None, "regular")
        ):
            self.modetype = ("singular", "regular")
        else:
            raise AnnotationError("invalid modetype")
        if self.basis is None:
            self.basis = SCWB.default([0], SCWB.defaultmmax(shape[0]))
        if self.material is None:
            self.material = AcousticMaterial()

    @property
    def ks(self):
        """Wave numbers (in medium).

        """
        return self.material.ks(self.k0)

    @property
    def krhos(self):
        r"""Radial part of the wave.

        Calculate :math:`\sqrt{k^2 - k_z^2}`, where :math:`k` is the wave number in the
        medium for each illumination

        Returns:
            Sequence[complex]
        """
        return self.material.krhos(self.k0, self.basis.kz)

    @classmethod
    def cylinder(cls, kzs, mmax, k0, radii, materials):
        """T-Matrix of a (multi-layered) cylinder.

        Construct the T-matrix of the given order and material for an infinitely
        extended cylinder. (The object can also consist of multiple concentric
        cylindrical shells with an arbitrary number of layers.)

        Args:
            kzs (float, array_like): Z component of the cylindrical wave.
            mmax (int): Positive integer for the maximum order of the T-matrix.
            k0 (float): Wave number in vacuum.
            radii (float or array): Radii from inside to outside of the cylinder. For a
                simple cylinder the radius can be given as a single number, for a multi-
                layered cylinder it is a list of increasing radii for all shells.
            material (list[AcousticMaterial]): The material parameters from the inside to the
                outside. The last material in the list specifies the embedding medium.

        Returns:
            AcousticT-Matrix
        """
        materials = [AcousticMaterial(m) for m in materials]
        if materials[-1].isshear:
            raise NotImplementedError
        kzs = np.atleast_1d(kzs)
        radii = np.atleast_1d(radii)
        if radii.size != len(materials) - 1:
            raise ValueError("incompatible lengths of radii and materials")
        dim = SCWB.defaultdim(len(kzs), mmax)
        tmat = np.zeros((dim, dim), np.complex128)
        idx = 0
        for kz in kzs:
            for m in range(-mmax, mmax + 1):
                miecoeffs = mie_acoustics_cyl(kz, m, k0, radii, *zip(*materials))
                tmat[idx, idx] = miecoeffs
                idx += 1
        return cls(tmat, k0=k0, basis=SCWB.default(kzs, mmax), material=materials[-1])

    @classmethod
    def cluster(cls, tmats, positions):
        r"""Block-diagonal T-matrix of multiple objects.

        Construct the initial block-diagonal T-matrix for a cluster of objects. The
        T-matrices in the list are placed together into a block-diagonal matrix and the
        complete (local) basis is defined based on the individual T-matrices and their
        bases together with the defined positions. In mathematical terms the matrix

        .. math::

            \begin{pmatrix}
                T_0 & 0 & \dots & 0 \\
                0 & T_1 & \ddots & \vdots \\
                \vdots & \ddots & \ddots & 0 \\
                0 & \dots & 0 & T_{N-1} \\
            \end{pmatrix}

        is created from the list of T-matrices :math:`(T_0, \dots, T_{N-1})`. Only
        T-matrices of the same wave number, and embedding material
        can be combined.

        Args:
            tmats (Sequence): List of T-matrices.
            positions (array): The positions of all individual objects in the cluster.

        Returns:
            AcousticTMatrix
        """
        for tm in tmats:
            if not tm.basis.isglobal:
                raise ValueError("global basis required")
        positions = np.array(positions)
        if len(tmats) < positions.shape[0]:
            warnings.warn("specified more positions than T-matrices")
        elif len(tmats) > positions.shape[0]:
            raise ValueError(
                f"'{len(tmats)}' T-matrices "
                f"but only '{positions.shape[0]}' positions given"
            )
        mat = tmats[0].material
        k0 = tmats[0].k0
        modes = [], []
        pidx = []
        dim = sum(tmat.shape[0] for tmat in tmats)
        tres = np.zeros((dim, dim), complex)
        i = 0
        for j, tm in enumerate(tmats):
            if tm.material != mat:
                raise ValueError(f"incompatible materials: '{mat}' and '{tm.material}'")
            if tm.k0 != k0:
                raise ValueError(f"incompatible k0: '{k0}' and '{tm.k0}'")
            dim = tm.shape[0]
            for m, n in zip(modes, tm.basis.zm):
                m.extend(list(n))
            pidx += [j] * dim
            tres[i : i + dim, i : i + dim] = tm
            i += dim
        basis = SCWB(zip(pidx, *modes), positions)
        return cls(tres, k0=k0, material=mat, basis=basis)

    @classmethod
    def from_array(cls, tm, basis, *, eta=0):
        """1d array of spherical T-matrices."""
        return cls(
            (tm @ opa.AcousticExpand(basis).inv).expandlattice(basis=basis, eta=eta),
            lattice=tm.lattice,
            kpar=tm.kpar,
        )

    @property
    def xw_ext_avg(self):
        r"""Rotation and polarization averaged extinction cross width.

        The average is calculated as

        .. math::

            \langle \lambda_\mathrm{ext} \rangle
            = -\frac{2 \pi}{n_{k_z}} \sum_{sk_zm} \frac{\Re(T_{sk_zm,sk_zm})}{k_s}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s` and :math:`n_{k_z}` is the number of wave components
        :math:`k_z` included in the T-matrix. The average is taken over all given
        z-components of the wave vector and rotations around the z-axis. It is only
        implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        nk = np.unique(self.basis.kz).size
        res = -4 * np.real(np.trace(self)) / (self.ks * nk)                   #!!!!!!!!!!!!!!!!!!!
        if res.imag == 0:
            return res.real
        return res

    @property
    def xw_sca_avg(self):
        r"""Rotation and polarization averaged scattering cross width.

        The average is calculated as

        .. math::

            \langle \lambda_\mathrm{sca} \rangle
            = \frac{2 \pi}{n_{k_z}} \sum_{sk_zm} \sum_{s'{k_z}'m'}
            \frac{|T_{sk_zm,s'{k_z}'m'}|^2}{k_s}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s`. and :math:`n_{k_z}` is the number of wave components
        :math:`k_z` included in the T-matrix. The average is taken over all given
        z-components of the wave vector and rotations around the z-axis. It is only
        implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        ks = self.ks
        re, im = self.real, self.imag
        nk = np.unique(self.basis.kz).size
        res = 4 * np.sum((re * re + im * im) / (ks * nk))
        return res.real

    @property
    def isglobal(self):
        """Test if a T-matrix is global.

        A T-matrix is considered global, when its basis refers to only a single point
        and it is not placed periodically in a lattice.
        """
        return self.basis.isglobal and self.lattice is None and self.kpar is None

    def xw(self, illu, flux=0.5):
        r"""Scattering and extinction cross width.

        Possible for all T-matrices (global and local) in non-absorbing embedding. The
        values are calculated by

        .. math::

            \lambda_\mathrm{sca}
            = \frac{1}{2 I}
            a_{sk_zm}^\ast T_{s'{k_z}'m',sk_zm}^\ast k_{s'}^{-2}
            C_{s'l'm',s''l''m''}^{(1)}
            T_{s''l''m'',s'''l'''m'''} a_{s'''l'''m'''} \\
            \sigma_\mathrm{ext}
            = \frac{1}{2 I}
            a_{slm}^\ast k_s^{-2} T_{slm,s'l'm'} a_{s'l'm'}

        where :math:`a_{slm}` are the expansion coefficients of the illumination,
        :math:`T` is the T-matrix, :math:`C^{(1)}` is the (regular) translation
        matrix and :math:`k_s` are the wave numbers in the medium. All repeated indices
        are summed over. The incoming flux is :math:`I`.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux corresponding to the illumination. Used for
                the result's normalization. The flux is given in units of
                :math:`\frac{\text{V}^2}{{l^2}} \frac{1}{Z_0 Z}` where :math:`l` is the
                unit of length used in the wave number (and positions). A plane wave
                has the flux `0.5` in this normalization, which is used as default.

        Returns:
            tuple[float]
        """
        if not self.material.isreal:
            raise NotImplementedError
        illu = AcousticPhysicsArray(illu)
        illu_basis = illu.basis
        illu_basis = illu_basis[-2] if isinstance(illu_basis, tuple) else illu_basis
        if not isinstance(illu_basis, SCWB):
            illu = illu.expand(self.basis)
        p = self @ illu
        p_invksq = p * np.power(self.ks, -1)
        del illu.modetype
        return (
            2.0 * np.real(p.conjugate().T @ p_invksq.expand(p.basis)) / flux,
            -2.0 * np.real(illu.conjugate().T @ p_invksq) / flux,
        )

    def valid_points(self, grid, radii):
        grid = np.asarray(grid)
        if grid.shape[-1] not in (2, 3):
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones(grid.shape[:-1], bool)
        for r, p in zip(radii, self.basis.positions):
            res &= np.sum(np.power(grid[..., :2] - p[:2], 2), axis=-1) > r * r
        return res

    def __getitem__(self, key):
        if isinstance(key, SCWB):
            key = np.array([self.basis.index(i) for i in key])
            key = (key[:, None], key)
        return super().__getitem__(key)


def _plane_wave_partial_scalar(
    kpar, *, k0=None, basis=None, material=None, modetype=None
):
    if basis is None:
        basis = SPWBC.default([kpar])
    modetype = "up" if modetype is None else modetype
    if modetype not in ("up", "down"):
        raise ValueError(f"invalid 'modetype': {modetype}")
    kzs = AcousticMaterial(material).kzs(k0, *kpar)
    pol = wv.spw_Psi(*kpar, kzs, 0, 0, 0)
    res = [pol * (np.abs(np.array(kpar) - x[:2]) < 1e-14).all() for x in basis]
    return AcousticPhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype
    )


def _plane_wave_scalar(
    kvec, *, k0=None, basis=None, material=None, modetype=None
):
    if basis is None:
        basis = SPWBUV.default([kvec])
    norm = np.sqrt(np.sum(np.power(kvec, 2)))
    qvec = kvec / norm
    if None not in (k0, material):
        kvec = AcousticMaterial(material).ks(k0) * qvec
    else:
        kvec = qvec
    pol = wv.spw_Psi(*kvec, 0, 0, 0)
    res = [pol * (np.abs(qvec - x[:3]) < 1e-14).all() for x in basis]
    return AcousticPhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype
    )


def plane_wave_scalar(
    kvec, *, k0=None, basis=None, material=None, modetype=None
):
    """Array describing a scalar plane wave.

    Args:
        kvec (Sequence): Wave vector.
        basis (ScalarPlaneWaveBasis, optional): Basis definition.
        k0 (float, optional): Wave number in air.
        material (AcousticMaterial, optional): Material definition.
        modetype (str, optional): Mode type (see :ref:`params:Mode types`).
    """
    if len(kvec) == 2:
        return _plane_wave_partial_scalar(
            kvec,
            k0=k0,
            basis=basis,
            material=material,
            modetype=modetype,
        )
    if len(kvec) == 3:
        return _plane_wave_scalar(
            kvec,
            k0=k0,
            basis=basis,
            material=material,
            modetype=modetype,
        )
    raise ValueError(f"invalid length of 'kvec': {len(kvec)}")


def plane_wave_angle_scalar(theta, phi, **kwargs):
    qvec = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    return plane_wave_scalar(qvec, **kwargs)


def spherical_wave_scalar(
    l,  # noqa: E741
    m,
    *,
    k0=None,
    basis=None,
    material=None,
    modetype=None
):
    if basis is None:
        basis = SSWB.default(l)
    if not basis.isglobal:
        raise ValueError("basis must be global")
    res = [0] * len(basis)
    res[basis.index((0, l, m))] = 1
    return AcousticPhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype
    )


def cylindrical_wave_scalar(
    kz, m, *, k0=None, basis=None, material=None, modetype=None
):
    if basis is None:
        basis = SCWB.default([kz], abs(m))
    if not basis.isglobal:
        raise ValueError("basis must be global")
    res = [0] * len(basis)
    res[basis.index((0, kz, m))] = 1
    return AcousticPhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype
    )