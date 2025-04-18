"""
Mie Theory and Optical Properties Module
----------------------------------------

Developed by: Julien El Hajj, Gilles Ledoux and Samy Merabia
contact: julienhajj99@gmail.com

This module computes the absorption efficiency (Q_abs) of spherical nanoparticles,
including bare gold nanoparticles and gold core–shell structures, with full 
temperature-dependent dielectric properties for the gold core.

This code supports various shell configurations around the gold core:

- Dense silica (SiO₂) shell
- Porous silica shell filled with:
    - Water
    - Vacuum

The effective optical properties of porous shells are computed using the
Bruggeman effective medium approximation.

Additionally, the code is flexible and can handle reversed geometries as well,
such as a silica core with a gold shell, by simply swapping the core and shell
parameters in the `Q_abs` function.
"""

import numpy as np
from scipy.constants import pi, c
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

"""
This section defines functions to compute the absorption efficiency (Q_abs)
of spherical core–shell or homogeneous nanoparticles using Mie theory.
Functions:
----------

- psi_n, psi_n_prime:
    Riccati–Bessel function and its derivative (ψₙ and ψₙ′).

- chi_n, chi_n_prime:
    Riccati–Neumann function and its derivative (χₙ and χₙ′).

- xi_n, xi_n_prime:
    Riccati–Hankel function of the first kind and its derivative (ξₙ and ξₙ′).

- mie_coefficients:
    Computes Mie coefficients a_n and b_n for layered spheres.

- Q_abs:
    Calculates the absorption efficiency Q_abs using the Mie coefficients.

- refractive_index_water:
    Returns the refractive index of water using a Sellmeier-type formula.

- refractive_index_sio2:
    Returns the refractive index of amorphous silica (SiO₂).

- bruggeman_ema, bruggeman_ema_vacuum:
    Calculate the effective refractive index of a porous shell using the
    Bruggeman effective medium approximation, with either water or vacuum
    filling the pores.
"""

def psi_n(n, x):
    """Riccati–Bessel: ψₙ(x) = x * jₙ(x)."""
    return x * spherical_jn(n, x)

def psi_n_prime(n, x):
    """Derivative: ψₙ'(x) = jₙ(x) + x * jₙ'(x)."""
    return spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)

def chi_n(n, x):
    """Riccati–Neumann: χₙ(x) = - x * yₙ(x)."""
    return -x * spherical_yn(n, x)

def chi_n_prime(n, x):
    """Derivative: χₙ'(x) = - [ yₙ(x) + x * yₙ'(x) ]."""
    return - (spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True))

def xi_n(n, x):
    """Riccati–Hankel (first kind): ξₙ(x) = ψₙ(x) - i * χₙ(x)."""
    return psi_n(n, x) - 1j * chi_n(n, x)

def xi_n_prime(n, x):
    """Derivative: ξₙ'(x) = ψₙ'(x) - i * χₙ'(x)."""
    return psi_n_prime(n, x) - 1j * chi_n_prime(n, x)

def mie_coefficients(n, m1, m2, x1, x2):
    # Compute intermediate An and Bn using inner parameters (x1)
    An_num = (m2 * psi_n(n, m2 * x1) * psi_n_prime(n, m1 * x1) - 
              m1 * psi_n_prime(n, m2 * x1) * psi_n(n, m1 * x1))
    An_den = (m2 * chi_n(n, m2 * x1) * psi_n_prime(n, m1 * x1) - 
              m1 * chi_n_prime(n, m2 * x1) * psi_n(n, m1 * x1))
    An = An_num / An_den

    Bn_num = (m2 * psi_n(n, m1 * x1) * psi_n_prime(n, m2 * x1) - 
              m1 * psi_n_prime(n, m1 * x1) * psi_n(n, m2 * x1))
    Bn_den = (m2 * chi_n_prime(n, m2 * x1) * psi_n(n, m1 * x1) - 
              m1 * chi_n(n, m2 * x1) * psi_n_prime(n, m1 * x1))
    Bn = Bn_num / Bn_den

    # Compute a_n and b_n using outer parameters (x2)
    an_num = psi_n(n, x2) * (psi_n_prime(n, m2 * x2) - An * chi_n_prime(n, m2 * x2)) \
             - m2 * psi_n_prime(n, x2) * (psi_n(n, m2 * x2) - An * chi_n(n, m2 * x2))
    an_den = xi_n(n, x2) * (psi_n_prime(n, m2 * x2) - An * chi_n_prime(n, m2 * x2)) \
             - m2 * xi_n_prime(n, x2) * (psi_n(n, m2 * x2) - An * chi_n(n, m2 * x2))
    an = an_num / an_den
         
    bn_num = m2 * psi_n(n, x2) * (psi_n_prime(n, m2 * x2) - Bn * chi_n_prime(n, m2 * x2)) \
             - psi_n_prime(n, x2) * (psi_n(n, m2 * x2) - Bn * chi_n(n, m2 * x2))
    bn_den = m2 * xi_n(n, x2) * (psi_n_prime(n, m2 * x2) - Bn * chi_n_prime(n, m2 * x2)) \
             - xi_n_prime(n, x2) * (psi_n(n, m2 * x2) - Bn * chi_n(n, m2 * x2))
    bn = bn_num / bn_den

    return an, bn

def Q_abs(wavelength_nm, R1_nm, R2_nm, n1, n2, nm_medium):
    """
    Absorption efficiency Q_abs computed via:
      Q_abs = (2/x2^2) sum_{n=1}^{N} (2n+1)[Re(a_n+b_n) - |a_n|^2 - |b_n|^2]
    where
      x1 = 2π * R1_nm * nm_medium / wavelength_nm
      x2 = 2π * R2_nm * nm_medium / wavelength_nm.
    For a bare sphere, R1_nm = R2_nm.
    """
    x1 = 2 * pi * R1_nm * nm_medium / wavelength_nm
    x2 = 2 * pi * R2_nm * nm_medium / wavelength_nm
    N = int(x2 + 4.3 * (x2 ** (1/3)) + 1)
    Qabs = 0.0
    for n in range(1, N):
        an, bn = mie_coefficients(n, n1/nm_medium, n2/nm_medium, x1, x2)
        Qabs += (2*n + 1) * (np.real(an + bn) - np.abs(an)**2 - np.abs(bn)**2)
    Qabs *= 2 / x2**2
    return Qabs

def refractive_index_water(lambda_um):
    return np.sqrt(1 + 0.5684027565 * lambda_um**2 / (lambda_um**2 - 0.005101829712) +
                      0.1726177391 * lambda_um**2 / (lambda_um**2 - 0.01821153936) +
                      0.02086189578 * lambda_um**2 / (lambda_um**2 - 0.02620722293) +
                      0.1130748688 * lambda_um**2 / (lambda_um**2 - 10.69792721))

# Refractive Index of SiO2 (Equation 11)
def refractive_index_sio2(lambda_um):
    return np.sqrt(1 + 0.6961663 * lambda_um**2 / (lambda_um**2 - 0.0684043**2) +
                      0.4079426 * lambda_um**2 / (lambda_um**2 - 0.1162414**2) +
                      0.8974794 * lambda_um**2 / (lambda_um**2 - 9.896161**2))

def bruggeman_ema(lambda_um, porosity):
    n_water = refractive_index_water(lambda_um)
    n_sio2 = refractive_index_sio2(lambda_um)

    def equation(n_eff):
        # Bruggeman equation: sum of weighted permittivities is zero
        return porosity * (n_water**2 - n_eff**2) / (n_water**2 + 2 * n_eff**2) + \
               (1 - porosity) * (n_sio2**2 - n_eff**2) / (n_sio2**2 + 2 * n_eff**2)

    # Solve for the effective refractive index using an optimization approach
    result = minimize(lambda n_eff: abs(equation(n_eff[0])), [1.5], bounds=[(1.0, 2.0)])
    return result.x[0]

def bruggeman_ema_vacuum(lambda_um, porosity):
    n_vacuum = 1.0  # Refractive index of vacuum
    n_sio2 = refractive_index_sio2(lambda_um)

    def equation(n_eff):
        return porosity * (n_vacuum**2 - n_eff**2) / (n_vacuum**2 + 2 * n_eff**2) + \
               (1 - porosity) * (n_sio2**2 - n_eff**2) / (n_sio2**2 + 2 * n_eff**2)

    result = minimize(lambda n_eff: abs(equation(n_eff[0])), [1.5], bounds=[(1.0, 2.0)])
    return result.x[0]

"""
Temperature-Dependent Dielectric Function for Gold
--------------------------------------------------

This section implements the intraband and interband contributions to the
dielectric function of gold (ε) as a function of electron temperature (Te),
based on a Drude-Lorentz model with Te-dependent parameters.

Constants:
----------
- k_B, h_eVs: Boltzmann and Planck constants (in eV units).
- e, epsilon0, m_e, hbar: physical constants in SI units.

Functions:
----------

1. Fermi–Dirac and Oscillator Parametrization
---------------------------------------------
- fermi_dirac(E, T, mu):
    Standard Fermi–Dirac distribution function.

- a_j_Te(...):
    Te-dependent oscillator amplitude a_j(Te), computed via FD ratios.

- chi(Te, mu0):
    Simple empirical model for the Te-dependence of chemical potential.

- omega_j_Te(...):
    Computes Te-dependent resonance energy ω_j(Te) from χ(Te).

2. Electron-Electron Damping (γ_ee)
-----------------------------------
- gamma_ee_in_eV(Te):
    Converts γ_ee(Te) data (given in 10¹⁵ s⁻¹) to eV using Planck's constant.
    The damping is interpolated from tabulated values.

3. Interband Contribution
-------------------------
- lorentz_term(E, j, Te):
    Returns the j-th Lorentz oscillator contribution to ε_inter(E, Te).

- epsilon_inter_Te(E_eV, Te):
    Sums all Lorentz terms to obtain the full Te-dependent interband ε.

4. Intraband (Drude) Contribution
---------------------------------
- ElectronicDensityInterpolator:
    Helper class to interpolate electron density n_sp as a function of Te.

- get_dampingtot(Te):
    Interpolates the total collision frequency ν_total from tabulated values.

- epsilon_intraband(omega_rad_s, Te):
    Computes the Te-dependent Drude response using ω_p²(Te) and ν(Te).
"""

k_B = 8.617333262e-5  # eV/K (Boltzmann's constant in eV/K)
h_eVs = 4.135667696e-15  # Planck's constant in eV·s

def fermi_dirac(E, T, mu):
    """
    Fermi–Dirac distribution at energy E [eV],
    temperature T [K], and chemical potential mu [eV].
    """
    return 1.0 / (1.0 + np.exp((E - mu) / (k_B * T)))

def a_j_Te(a_j0, E_j, Te, T_room, mu):
    """
    Temperature-dependent amplitude a_j(Te):
      a_j(Te) = a_j0 * [ f(E_j, Te) / f(E_j, T_room) ].

    E_j here is the 'transition energy' for the j-th oscillator,
    often taken as hbar*omega_j0 (room-temp resonance).
    """
    fd_ratio = fermi_dirac(E_j, Te, 0) / fermi_dirac(E_j, T_room, 0)
    return a_j0 * fd_ratio

def chi(Te, mu0, c=0.5):
    """
    Example chemical potential shift function:
      chi(Te) = mu0 - c*(k_B * Te).

    c is a dimensionless factor controlling how quickly
    the resonance shifts with temperature.
    """
    return mu0 + 5.5 - c * (k_B * Te)

def omega_j_Te(omega_j0, Te, chi_func, T_room=300.0):
    """
    Temperature-dependent resonance:
      (hbar*omega_j)(Te) = (hbar*omega_j0) + [chi(Te) - chi(T_room)].
    """
    return omega_j0 + (chi_func(Te,0) - chi_func(T_room,0))

Te_table = np.array([
    296.400563040419, 740.8807560828482, 1185.3609491252773, 1777.9207721697167,
    2222.320530866681, 2765.4936657952967, 3209.812990146793, 3900.9048863864864,
    4542.610094510359, 5332.394932636236, 6220.55097526644, 6911.482002815204,
    7947.4763724110235, 8933.923185200085, 10561.75346873115, 11992.036999798915,
    13274.401769555607, 14605.91192439172, 15986.647898652729, 17120.933038407406,
    18255.05730947115, 19241.262819223815, 19833.098733159066
])

# gamma_ee_table[i] = e.g. 0.0043 => means 0.0043 * 1e15 s^-1
gamma_ee_table = np.array([
    0.00430323748, 0.00575105570, 0.00719887392, 0.00587170722,
    0.00406193445, 0.00257389905, 0.00249346471, 0.01327166700,
    0.02421073799, 0.03792479389, 0.06760506736, 0.08489845164,
    0.12712648301, 0.17603056505, 0.24890408204, 0.32242107380,
    0.38664789865, 0.46048662779, 0.54067967022, 0.60213151015,
    0.67009853207, 0.72877538709, 0.75942087271
])

interp_gamma_ee_raw = interp1d(Te_table, gamma_ee_table,
                               kind='linear', fill_value='extrapolate')

def gamma_ee_in_eV(Te):
    """
    Convert from "gamma_ee in 10^15 s^-1" => eV, using:
      1) multiply by 1e15 => s^-1
      2) multiply by h [eV·s] => eV
    """
    gamma_sinv = interp_gamma_ee_raw(Te) * 1.e15
    return gamma_sinv * h_eVs

##############################################################################
# Room-temperature oscillator parameters
##############################################################################
a_vals_0    = np.array([4.577,   4.999,  12.281,  24.999,  34.999])
omega_vals_0= np.array([2.784,   3.183,   3.799,   4.695,   6.349])  # eV
Gamma_ei_0   = np.array([0.499,   0.699,   0.999,   1.651,   2.190])  # eV

def lorentz_term(E, j, Te):
    """
    Returns the j-th oscillator contribution to eps_inter(E, Te).
    E is photon energy in eV (scalar or array).
    """
    # (A) T-dependent amplitude
    #   E_j ~ the 'transition energy' relevant for FD ratio => use the
    #   room-temp resonance as that transition level.
    E_j = -h_eVs*omega_vals_0[j]   
    a_j = a_j_Te(a_vals_0[j], E_j, Te, T_room=300.0, mu=0.0)

    # (B) T-dependent resonance
    w0_Te = omega_j_Te(omega_vals_0[j], Te, chi_func=chi, T_room=300.0)
    w0_sq = w0_Te**2

    # (C) T-dependent damping = gamma_ei(300K) + gamma_ee(Te)
    gamma_total = Gamma_ei_0[j] + gamma_ee_in_eV(Te)

    # (D) Lorentz denominator: (w0^2 - E^2 - i E gamma)
    return (a_j) / ((w0_sq - E**2) - 1j*(E * gamma_total))

def epsilon_inter_Te(E_eV, Te):
    """
    Sum of 5 Lorentz oscillators with T-dependent parameters,
    giving the complex interband dielectric function.
    """
    E = np.atleast_1d(E_eV)
    total = np.zeros_like(E, dtype=np.complex128)
    for j in range(5):
        total += lorentz_term(E, j, Te)
    return total if E.shape[0] > 1 else total[0]

#------------------------------------
# PHYSICAL CONSTANTS (SI units)
#------------------------------------
e        = 1.60217662e-19        # Coulomb (elementary charge)
epsilon0 = 8.85418782e-12        # F/m (vacuum permittivity)
m_e      = 9.10938356e-31        # kg (electron mass)
hbar     = 1.054571817e-34       # J·s (reduced Planck constant)
eV_to_J  = 1.602176634e-19       # 1 eV = this many Joules

def eV_to_omega(eV):
    return (eV * eV_to_J) / hbar

epsilon_infinity = 3.104  # for gold
# We'll use the free-electron mass for m_sp^*
m_sp_star = m_e

class ElectronicDensityInterpolator:
    def __init__(self, Te_values, nsd_values):
        """
        Initialize the interpolator with electronic temperature (Te) and electronic density (nsd) data.
        
        Parameters:
        - Te_values: np.array, electronic temperature values (in y2)
        - nsd_values: np.array, corresponding electronic density values (in y)
        """
        self.Te_values = Te_values
        self.nsd_values = nsd_values
        self.interpolator = interp1d(Te_values, nsd_values, kind='linear', fill_value='extrapolate')

    def get_density(self, Te):
        """
        Get the electronic density corresponding to the given electronic temperature.
        
        Parameters:
        - Te: float or np.array, electronic temperature(s)
        
        Returns:
        - nsd: float or np.array, electronic density at the given temperature(s)
        """
        return self.interpolator(Te)

# Define the given data
Te_article = np.array([
    0.625076069758137, 1.2019891155043219, 1.8752282092744217, 2.6447064142020054,
    3.3181193817050065, 3.9917931598073473, 4.761706049067168, 5.531879748926329,
    6.3508250308625875, 7.169857249665295, 8.133552414236785, 8.808617182202287,
    9.62851876966947, 10.400518143724032, 10.979517674264953, 11.510267243927464,
    12.089788395667071, 12.910211604332938, 13.2483090779476, 13.634482638707775,
    13.828177977152997, 14.214525411646068, 14.552449011527834, 14.938796446020906,
    15.325317754246871, 15.712012936205735, 16.098708118164595, 16.727174725714193,
    17.162467616017256, 17.74285813642134, 18.177977152991506, 18.661520004172978,
    19.145062855354453, 19.58018187192462, 19.822648792446934, 20.305843896162617
]) * 1e3  # K

nsp_articles = np.array([
    5.950423303199887, 5.97006385421151, 5.999641268474676,
    6.019299756062566, 6.058706414119673, 6.0783110919787635, 6.10783469651313,
    6.147187544841442, 6.166810159276798, 6.2061271344525775, 6.245408236475821,
    6.284725211651601, 6.324006313674845, 6.3731525326445695, 6.42215525900416,
    6.481130721767828, 6.5400703113789636, 6.559621179509254, 6.598902281532499,
    6.638111637250681, 6.697087100014349, 6.7264313387860515, 6.775434065145643,
    6.824436791505235, 6.8832687616587735, 6.932235614865832, 6.991067585019369,
    7.03027694073755, 7.079136174487012, 7.128031281389006, 7.157411393313241,
    7.216279236619312, 7.245515855933416, 7.3043478260869525, 7.333656191706123,
    7.382587171760652
]) * 1e28  # m^3

electronic_density_interpolator = ElectronicDensityInterpolator(Te_article, nsp_articles)

Te_values_nu_total = np.array([
    292.6829268292695, 634.1463414634154, 1024.3902439024405, 1365.8536585365864,
    1756.0975609756106, 2073.170731707318, 2439.0243902439033, 2926.829268292684,
    3463.414634146342, 4243.902439024392, 5170.731707317076, 6048.780487804881,
    7170.731707317076, 7707.317073170736, 8292.682926829271, 9170.731707317076,
    10292.682926829273, 11609.756097560981, 12585.365853658539, 14048.780487804881,
    15219.512195121959, 16341.463414634152, 17463.41463414635, 18292.682926829275,
    19365.853658536595
])

nu_total_values = np.array([
    0.08993576017130689, 0.08993576017130689, 0.089935760171306,
    0.08565310492505418, 0.08993576017130778, 0.08993576017130778,
    0.0942184154175596, 0.0985010706638132, 0.10278372591006502,
    0.11563169164882314, 0.14132762312634028, 0.16702355460385654,
    0.20985010706638407, 0.235546038543899, 0.2612419700214157,
    0.3040685224839428, 0.3597430406852271, 0.4325481798715245,
    0.4882226980728084, 0.573875802997863, 0.6423982869379063,
    0.7109207708779501, 0.7837259100642462, 0.8351177730192787,
    0.8993576017130684
])

nu_total_interp = interp1d(Te_values_nu_total, nu_total_values, kind='linear', fill_value='extrapolate')

def get_dampingtot(Te):
    """
    Get the total collision frequency (nu_total) for a given electron temperature (Te).

    Parameters:
    - Te: float or np.array, electron temperature(s)

    Returns:
    - nu_total: float or np.array, total collision frequency at the given temperature(s)
    """
    return nu_total_interp(Te)

def epsilon_intraband(omega_rad_s, Te):
    """
    Compute the Drude (intraband) dielectric function at angular freq (rad/s),
    for electron temperature Te (K).
    
    Returns a complex number eps_intra.
    """
    # 1) conduction-electron density as function of Te
    n_sp = electronic_density_interpolator.get_density(Te)  # m^-3
    
    # 2) Plasma frequency, w_p(Te). Use SI units.
    #    w_p^2 = e^2 * n_sp / (epsilon0 * m_sp_star)
    w_p_sq = (e**2) * n_sp / (epsilon0 * m_sp_star)
    
    # 3) Total damping frequency, in s^-1
    #    user data are in "10^15 s^-1" => multiply by 1.e15
    nu_total = get_dampingtot(Te) * 1.e15
    
    # 4) Drude denominator
    denom = omega_rad_s**2 + 1j * omega_rad_s * nu_total
    
    # 5) Full intraband dielectric function
    eps_drude = epsilon_infinity - w_p_sq / denom
    
    return eps_drude