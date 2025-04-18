"""
Example 2: Absorption of Bare Gold Nanoparticles (Size Sweep)
-------------------------------------------------------------

This script computes and plots the absorption efficiency (Q_abs) of bare
spherical gold nanoparticles of varying radii (20–100 nm), embedded in water. 

The temperature-dependent dielectric function of gold is used for each
electron temperature (Te = 300, 1000, 5000, 10000 K), and the results are
shown across 2×2 subplots.

- Demonstrates size-dependent resonance shifts and broadening.
- Medium is water for all cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
from mie_absorption import (
    Q_abs,
    epsilon_inter_Te,
    epsilon_intraband,
    refractive_index_water
)

def energy_eV_to_lambda_nm(E_eV):
    return 1239.84193 / E_eV

if __name__ == "__main__":
    # Wavelength axis (nm)
    wavelengths = np.linspace(400, 800, 200)
    λ_um = wavelengths * 1e-3

    # Radii in nm
    radii = [20, 30, 40, 50, 60, 80, 100]

    # Water as surrounding medium
    n_medium = refractive_index_water(λ_um)

    # Electron temperatures
    Te_list = [300, 1000, 5000, 10000]

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, Te in zip(axes, Te_list):
        E_eV = energy_eV_to_lambda_nm(wavelengths)
        ω = 2 * pi * c / (wavelengths * 1e-9)
        eps_core = epsilon_inter_Te(E_eV, Te) + epsilon_intraband(ω, Te)
        n_core = np.sqrt(eps_core)

        for R_nm in radii:
            Qvals = [
                Q_abs(wl, R_nm, R_nm, nc, nc, nm)
                for wl, nc, nm in zip(wavelengths, n_core, n_medium)
            ]
            ax.plot(wavelengths, Qvals, label=f"R = {R_nm} nm")

        ax.set_title(f"Te = {Te} K", fontsize=12)
        ax.set_xlabel("Wavelength (nm)", fontsize=10)
        ax.set_ylabel(r"$Q_{\mathrm{abs}}$", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
