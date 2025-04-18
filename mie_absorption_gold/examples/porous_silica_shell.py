"""
Example 3: Porous Silica Shell Filled with Water or Vacuum
----------------------------------------------------------

This example computes the absorption efficiency (Q_abs) of a core–shell 
nanoparticle composed of a 50 nm gold core and a 5 nm porous silica shell 
with 50% porosity, embedded in water. 

Two cases are compared:
- Pores filled with water
- Pores filled with vacuum (n = 1)

The shell refractive index is calculated using the Bruggeman EMA model.
Electron temperature is fixed at 300 K.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from mie_absorption import (
    Q_abs,
    epsilon_inter_Te,
    epsilon_intraband,
    refractive_index_water,
    bruggeman_ema,
    bruggeman_ema_vacuum
)

def lambda_nm_to_energy_eV(lambda_nm):
    hc_eVnm = 1239.84193
    return hc_eVnm / lambda_nm

if __name__ == "__main__":
    wavelengths = np.linspace(400, 800, 200)
    λ_um = wavelengths * 1e-3

    # Convert λ[nm] → E[eV]
    E_eV = lambda_nm_to_energy_eV(wavelengths)
    ω = 2 * pi * c / (wavelengths * 1e-9)

    # Geometry
    R_core_nm  = 50.0
    R_shell_nm = R_core_nm + 5.0

    # Electron temperature
    Te = 300

    # Gold core refractive index
    eps_gold = epsilon_inter_Te(E_eV, Te) + epsilon_intraband(ω, Te)
    n_gold = np.sqrt(eps_gold)

    # Porosity of silica shell
    porosity = 0.5

    # Shell index: water- and vacuum-filled pores
    n_shell_water  = np.array([bruggeman_ema(l, porosity) for l in λ_um])
    n_shell_vacuum = np.array([bruggeman_ema_vacuum(l, porosity) for l in λ_um])

    # Medium = water
    n_medium = refractive_index_water(λ_um)

    # Q_abs for both cases
    Qabs_water = [
        Q_abs(wl, R_core_nm, R_shell_nm, nc, ns, nm)
        for wl, nc, ns, nm in zip(wavelengths, n_gold, n_shell_water, n_medium)
    ]

    Qabs_vacuum = [
        Q_abs(wl, R_core_nm, R_shell_nm, nc, ns, nm)
        for wl, nc, ns, nm in zip(wavelengths, n_gold, n_shell_vacuum, n_medium)
    ]

    # Plot results
    plt.figure(figsize=(6, 4))
    plt.plot(wavelengths, Qabs_water, label="Porous SiO₂ (filled with water)")
    plt.plot(wavelengths, Qabs_vacuum, label="Porous SiO₂ (filled with vacuum)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"$Q_{\mathrm{abs}}$")
    plt.title("Q_abs at Te = 300 K with Porous Shell")
    plt.legend()
    plt.tight_layout()
    plt.show()
