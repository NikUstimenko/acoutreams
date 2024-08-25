import numpy as np
from treams import misc

class AcousticMaterial:
    r"""Material definition.

    The material properties are defined in the frequency domain through scalar values
    for density :math:`\rho`, the speed of longitudinal elastic waves :math:`c`, and
    the speed of transverse elastic waves :math:`c`. Materials are, thus, assumed to be linear, 
    time-invariant, homogeneous, isotropic, and local. Also, it is assumed that they have no gain.

    Args:
        rho (optional, complex): Mass density. Defaults to 1.3 [kg/m^3].
        c (optional, complex): Longitudinal speed of sound. Defaults to 343 [m/s].
        ct (optional, complex): Transverse speed of sound. Defaults to 0 [m/s].
    """

    def __init__(self, rho=1.3, c=343., ct=0.):
        """Initialization."""
        if isinstance(rho, AcousticMaterial):
            rho, c, ct = rho()
        elif isinstance(rho, (tuple, list, np.ndarray)):
            if len(rho) == 0:
               rho = 1.3
            elif len(rho) == 1:
                rho = rho[0]
            elif len(rho) == 2:
                rho, c = rho
            elif len(rho) == 3:
                rho, c, ct = rho
            else:
                raise ValueError("invalid material definition")
        self._rho = rho
        self._c = c
        self._ct = ct

    @property
    def rho(self):
        """Mass density.

        Returns:
            float or complex
        """
        return self._rho

    @property
    def c(self):
        """Speed of longitudinal elastic waves.

        Returns:
            float or complex
        """
        return self._c
    
    @property
    def ct(self):
        """Speed of transverse elastic waves.

        Returns:
            float or complex
        """
        return self._ct
    
    def __iter__(self):
        """Iterator for a tuple containing the material parameters.

        Useful for unpacking the material parameters into a function that takes these
        parameters separately, e.g. ``foo(*material)``.
        """
        return iter((self.rho, self.c, self.ct))
    
    @property
    def impedance(self):
        r"""Relative impedance for longitudinal waves.

        The relative impedance is defined by :math:`Z = \rho c`.

        Returns:
            complex
        """
        return self.rho * self.c
    
    @property
    def impedancet(self):
        r"""Relative impedance for transverse waves..

        The relative impedance is defined by :math:`Z = \rho ct`.

        Returns:
            complex
        """
        return self.rho * self.ct
    
    def __call__(self):
        """Return a tuple containing all material parameters.

        Returns:
            tuple
        """
        return self.rho, self.c, self.ct

    def __eq__(self, other):
        """Compare material parameters.

        Materials are considered equal, when all material parameters are equal. Also,
        compares with objects that contain at most three values.

        Returns:
            bool
        """
        if other is None:
            return False
        if not isinstance(other, AcousticMaterial):
            other = AcousticMaterial(*other)
        return (
            self.rho == other.rho
            and self.c == other.c
            and self.ct == other.ct
        )
    
    @property
    def isreal(self):
        """Test if the material has purely real parameters.

        Returns:
            bool
        """
        return all(i.imag == 0 for i in self)
    
    @property
    def isshear(self):
        """Test if the material supports transverse elastic waves.

        Returns:
            bool
        """
        return self.ct != 0

    def __str__(self):
        """All three material parameters.

        Returns:
            str
        """
        return "(" + ", ".join([str(i) for i in self()]) + ")"

    def __repr__(self):
        """Representation that allows recreating the object.

        Returns:
            str
        """
        return self.__class__.__name__ + str(self)

    def ks(self, k0):
        """Return the wave number of longitudinal waves in the medium.

        Args:
            k0 (float): Wave number in air.

        Returns:
            tuple
        """
        if isinstance(self.c, (tuple, list, np.ndarray)):
            return tuple(k0 * AcousticMaterial().c / c for c in self.c)
        else:
            return k0 * AcousticMaterial().c / self.c

        
    def kst(self, k0):
        """Return the wave number of transverse waves in the medium.

        Args:
            k0 (float): Wave number in air.

        Returns:
            tuple
        """
        if isinstance(self.ct, (tuple, list, np.ndarray)):
            return tuple(k0 * AcousticMaterial().c / ct for ct in self.ct)
        else:
            return k0 * AcousticMaterial().c / self.ct
    
    def krhos(self, k0, kz):
        r"""The (cylindrically) radial part of the wave vector for longitudinal waves .

        The cylindrically radial part is defined by :math:`k_\rho = \sqrt(k^2 - k_z^2)`.
        The returned values have non-negative imaginary parts.

        Args:
            k0 (float): Wave number in air.
            kz (float, array-like): z-component of the wave vector

        Returns:
            complex, array-like
        """
        ks = self.ks(k0)
        return misc.wave_vec_z(kz, 0, ks)

    def kzs(self, k0, kx, ky):
        r"""The z-component of the wave vector for longitudinal waves.

        The z-component of the wave vector is defined by
        :math:`k_z = \sqrt(k^2 - k_x^2 - k_y^2)`. The returned values have
        non-negative imaginary parts.

        Args:
            k0 (float): Wave number in air.
            kx (float, array-like): x-component of the wave vector
            ky (float, array-like): y-component of the wave vector

        Returns:
            complex, array-like
        """
        ks = self.ks(k0)
        return misc.wave_vec_z(kx, ky, ks)   
    
    def krhost(self, k0, kz):
        r"""The (cylindrically) radial part of the wave vector for transverse waves.

        The cylindrically radial part is defined by :math:`k_\rho = \sqrt(k^2 - k_z^2)`.
        The returned values have non-negative imaginary parts.

        Args:
            k0 (float): Wave number in air.
            kz (float, array-like): z-component of the wave vector

        Returns:
            complex, array-like
        """
        ks = self.kst(k0)
        return misc.wave_vec_z(kz, 0, ks)

    def kzst(self, k0, kx, ky):
        r"""The z-component of the wave vector for transverse waves.

        The z-component of the wave vector is defined by
        :math:`k_z = \sqrt(k^2 - k_x^2 - k_y^2)`. The returned values have
        non-negative imaginary parts.

        Args:
            k0 (float): Wave number in air.
            kx (float, array-like): x-component of the wave vector
            ky (float, array-like): y-component of the wave vector

        Returns:
            complex, array-like
        """
        ks = self.kst(k0)
        return misc.wave_vec_z(kx, ky, ks) 