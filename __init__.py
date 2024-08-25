"""AcouTREAMS: T-Matrix scattering code for acoustic computations.

.. currentmodule:: acoutreams

Classes
=======

The top-level classes and functions allow a high-level access to the functionality.

Basis sets
----------

.. autosummary::
   :toctree: generated/

   ScalarCylindricalWaveBasis
   ScalarPlaneWaveBasisByUnitVector
   ScalarPlaneWaveBasisByComp
   ScalarSphericalWaveBasis

Matrices and Arrays
-------------------

.. autosummary::
   :toctree: generated/

   AcousticPhysicsArray
   AcousticSMatrix
   AcousticSMatrices
   AcousticTMatrix
   AcousticTMatrixC

Functions
=========

.. autosummary::
   :toctree: generated/

   pfield
   vfield
   expand
   expandlattice
   permute
   plane_wave
   rotate
   translate

"""

from acoutreams._coreacoustics import (  # noqa: F401
   ScalarCylindricalWaveBasis,
   ScalarPlaneWaveBasisByUnitVector,
   ScalarPlaneWaveBasisByComp,
   ScalarSphericalWaveBasis,
)
from acoutreams._materialacoustics import AcousticMaterial  # noqa: F401
from acoutreams._operatorsacoustics import (  # noqa: F401
    PField,
    VField,
    AcousticExpand,
    AcousticExpandLattice,
    AcousticPermute,
    AcousticRotate,
    AcousticTranslate,
    pfield,
    vfield,
    expand,
    expandlattice,
    permute,
    rotate,
    translate,
)
from acoutreams._smatrixacoustics import (  # noqa: F401
    AcousticSMatrices,
    AcousticSMatrix,
    poynting_avg_z,
)
from acoutreams._tmatrixacoustics import (  # noqa: F401
    AcousticTMatrix,
    AcousticTMatrixC,
    cylindrical_wave_scalar,
    plane_wave_scalar,
    plane_wave_angle_scalar,
    spherical_wave_scalar,
)
