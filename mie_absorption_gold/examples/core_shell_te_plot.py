"""
Example 1: Absorption Efficiency of a Core–Shell Nanoparticle in Water
----------------------------------------------------------------------

This block computes and plots the absorption efficiency (Q_abs) of a 
spherical core–shell nanoparticle consisting of a 50 nm gold core and 
a 5 nm thick silica shell, embedded in water. The calculation is repeated 
for various electron temperatures (Te = 300, 1000, 5000, 10000 K), 
which affect the gold dielectric function.

- Uses temperature-dependent dielectric model for gold (core).
- Uses Sellmeier model for silica shell and water medium.
- Plots Q_abs as a function of wavelength for each Te.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from mie_absorption import (
    Q_abs,
    epsilon_inter_Te,
    epsilon_intraband,
    refractive_index_sio2,
    refractive_index_water
)

# Utility function: λ[nm] → E[eV]
def lambda_nm_to_energy_eV(lambda_nm):
    hc_eVnm = 1239.84193
    return hc_eVnm / lambda_nm

if __name__ == "__main__":
    # Wavelength axis
    wavelengths = np.linspace(400, 800, 200)
    λ_um = wavelengths * 1e-3

    # Geometry
    R_core_nm = 50.0
    R_shell_nm = R_core_nm + 5.0

    # Medium (water)
    n_medium = refractive_index_water(λ_um)

    # Electron temperatures
    Te_list = [300, 1000, 5000, 10000]

    plt.figure(figsize=(6, 4))

    for Te in Te_list:
        # Gold dielectric (Te-dependent)
        E_eV = lambda_nm_to_energy_eV(wavelengths)
        ω = 2 * pi * c / (wavelengths * 1e-9)
        eps_tot = epsilon_inter_Te(E_eV, Te) + epsilon_intraband(ω, Te)
        n_core = np.sqrt(eps_tot)

        # Silica shell
        n_shell = refractive_index_sio2(λ_um)

        # Absorption efficiency
        Qvals = [
            Q_abs(wl, R_core_nm, R_shell_nm, nc, ns, nm)
            for wl, nc, ns, nm in zip(wavelengths, n_core, n_shell, n_medium)
        ]

        plt.plot(wavelengths, Qvals, label=f"Te = {Te} K")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$Q_{\mathrm{abs}}$")
    plt.title("Absorption Efficiency for Gold@SiO₂ (50+5 nm)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
