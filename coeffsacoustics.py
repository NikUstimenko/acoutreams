import treams.special
import numpy as np
from acoutreams._materialacoustics import AcousticMaterial

def mie_acoustics(l, x, *materials):
    
    # Returns acoustical Mie coefficients for a sphere

    # l - order
    # x = k0 * radius - size parameter (k0 is in air)
    # *materials - list of tuples of material parameters

    mat_sphere, mat_env = zip(*materials) 
    x_sphere = x * AcousticMaterial().c / mat_sphere[1]
    x_env = x * AcousticMaterial().c / mat_env[1]
    j = treams.special.spherical_jn(l, x_env, derivative=False)
    j_d = treams.special.spherical_jn(l, x_env, derivative=True)
    h = treams.special.spherical_hankel1(l, x_env)
    h_d = treams.special.spherical_hankel1_d(l, x_env)
    j1 = treams.special.spherical_jn(l, x_sphere, derivative=False)
    j1_d = treams.special.spherical_jn(l, x_sphere, derivative=True)
    if mat_sphere[2] == 0:
        delta = x_sphere * mat_env[0] / (x_env * mat_sphere[0]) 
        res = (delta * j1_d * j - j1 * j_d) / (j1 * h_d - delta * j1_d * h) 
    else:
        x_sphere_t = x * AcousticMaterial().c / mat_sphere[2]
        j1t = treams.special.spherical_jn(l, x_sphere_t, derivative=False)
        j1t_d = treams.special.spherical_jn(l, x_sphere_t, derivative=True)
        d44 = (l * (l + 1) - 0.5 * x_sphere_t**2) * j1 - 2 * x_sphere * j1_d
        d33 = (l * (l + 1) - 0.5 * x_sphere_t**2 - 1) * j1t - x_sphere_t * j1t_d
        d34 = x_sphere * j1_d - j1
        d43 = l * (l + 1) * (x_sphere_t * j1t_d - j1t)
        d24 = x_sphere * j1_d
        d23 = l * (l + 1) * j1t
        zeta = (d44*d33 - d34*d43) / (d24*d33 - d34*d23)
        res = (-(mat_env[0]/mat_sphere[0] * 0.5 * x_sphere_t**2/x_env * j + zeta * j_d) / 
               (mat_env[0]/mat_sphere[0] * 0.5 * x_sphere_t**2/x_env * h + zeta * h_d))

    return res * np.ones(2*l+1)

def mie_acoustics_cyl(kz, m, k0, radii, *materials):
    
    # Returns acoustical Mie coefficients for an infinite cylinder (along z-axis)

    # kz - z-component of the wave vector
    # m - angular momentum
    # k0 - wave number in air
    # radii - radius of a cylinder
    # *materials - list of tuples of material parameters

    mat_cyl, mat_env = zip(*materials) 
    k_cyl = k0 * AcousticMaterial().c / mat_cyl[1]
    k_env = k0 * AcousticMaterial().c / mat_env[1]
    krho_cyl = np.sqrt(k_cyl * k_cyl - kz * kz)
    krho_env = np.sqrt(k_env * k_env - kz * kz)
    x_cyl = krho_cyl * radii
    x_env = krho_env * radii
    j = treams.special.jv(m, x_env)
    j_d = treams.special.jv_d(m, x_env)
    h = treams.special.hankel1(m, x_env)
    h_d = treams.special.hankel1_d(m, x_env)
    j1 = treams.special.jv(m, x_cyl)
    j1_d = treams.special.jv_d(m, x_cyl)
    if mat_cyl[2] == 0:
        delta = x_cyl * mat_env[0] / (x_env * mat_cyl[0]) 
        res = (delta * j1_d * j - j1 * j_d) / (j1 * h_d - delta * j1_d * h)
    else:
        k_cyl_t = k0 * AcousticMaterial().c / mat_cyl[2]
        krho_cyl_t = np.sqrt(k_cyl_t * k_cyl_t - kz * kz)
        x_cyl_t = krho_cyl_t * radii
        j1t = treams.special.jv(m, x_cyl_t)
        j1t_d = treams.special.jv_d(m, x_cyl_t)
        j1t_dd = (-j1t_d + (x_cyl_t**2 + m**2) * j1t) / x_cyl_t
        j1_dd = (-j1_d + (x_cyl**2 + m**2) * j1) / x_cyl
        matrix = np.zeros((3, 3), complex)
        rhs = np.zeros(3, complex)
        lam_0 = mat_env[0] * mat_env[1]**2
        mu_1 = mat_cyl[0] * mat_cyl[2]**2
        lam_1 = mat_cyl[0] * mat_cyl[1]**2 - 2 * mu_1
        matrix[0][1] = 2j * kz * krho_cyl/k_cyl * j1_d
        matrix[0][2] = (krho_cyl_t**2 - kz**2)/k_cyl_t * j1t_d
        matrix[1][0] = k_env * lam_0 * h
        matrix[1][1] = 2 * mu_1 * krho_cyl**2/k_cyl * j1_dd - k_cyl * lam_1 * j1
        matrix[1][2] = 2j * mu_1 * kz * krho_cyl_t/k_cyl_t * j1t_dd
        rhs[1] = -np.power(-1j, m) * k_env * lam_0 * j
        matrix[2][0] = -krho_env/k_env * h_d
        matrix[2][1] = krho_cyl/k_cyl * j1_d
        matrix[2][2] = 1j * kz/k_cyl_t * j1t_d
        rhs[2] = np.power(-1j, m) * krho_env/k_env * j_d
        #print(rhs.shape)
        res = np.linalg.solve(matrix, rhs)
        res = res[0] * np.power(1j, m)

    return res

def fresnel_acoustics(kzs, rhos):
    res = np.zeros((2, 2, 1, 1), complex)
    res[0][0][0][0] = 2 * rhos[0] * kzs[1] / (rhos[0] * kzs[1] + rhos[1] * kzs[0])
    res[1][1][0][0] = 2 * rhos[1] * kzs[0] / (rhos[0] * kzs[1] + rhos[1] * kzs[0])
    res[0][1][0][0] = (-rhos[0] * kzs[1] + rhos[1] * kzs[0]) / (rhos[0] * kzs[1] + rhos[1] * kzs[0])
    res[1][0][0][0] = -res[0][1]
    return res