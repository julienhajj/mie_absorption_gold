
# Mie Absorption Efficiency: Gold Nanoparticles with Te-Dependent Dielectric Function

This repository contains a modular and extensible Python implementation of **Mie theory** to compute the **absorption efficiency** (`Q_abs`) of spherical core-shell nanoparticles, with a focus on **gold cores** whose dielectric function evolves with **electronic temperature (Te)**. The implementation includes dense and porous silica shells and supports water or vacuum as embedding media.

> 📌 Developed by Julien El Hajj, Gilles Ledoux, and Samy Merabia. For inquiries: julienhajj99@gmail.com

---

## ✨ Features
- Full implementation of Mie theory for multilayered spheres
- Temperature-dependent dielectric model for gold (interband + intraband)
- Support for:
  - Bare gold nanoparticles
  - Core-shell (gold@SiO2)
  - Porous shells using Bruggeman Effective Medium Approximation (EMA)
  - Arbitrary surrounding medium (e.g., water)
- Easily extendable to SiO2@Au or other reverse configurations

---

## 🧪 Examples
The `examples/` folder includes ready-to-run scripts:

### 1. `core_shell_te_plot.py`
Absorption efficiency of a **50 nm gold core + 5 nm silica shell** at:
- Te = 300, 1000, 5000, 10000 K
- In water

### 2. `bare_gold_sizes.py`
Size-dependent absorption for **bare gold spheres** with:
- Radii = 20–100 nm
- Te = 300, 1000, 5000, 10000 K
- In water

### 3. `porous_shell.py`
Effect of porosity in the shell (50%):
- Compares water-filled vs vacuum-filled pores
- Te = 300 K only

📂 See `examples/example_outputs/` for resulting plots.

---

## 📦 Installation
No package installation required. Uses standard scientific Python libraries:
```bash
pip install numpy matplotlib scipy
```

---

## ▶️ How to Use
You can run any example directly:
```bash
python examples/core_shell_te_plot.py
```
Or import the functions in your own scripts:
```python
from mie_absorption import Q_abs, epsilon_inter_Te, epsilon_intraband
```

---

## 📄 License
This project is licensed under the MIT License (see `LICENSE`).

---

## 📚 Citation
This code is developed in support of an upcoming publication. Please check back for the DOI or citation once the article is accepted.

---

## 🔓 Release Info
This repository is currently **private** and will be made public upon acceptance of the related publication.

---
