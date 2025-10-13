# LiquidCrystals.jl

**LiquidCrystals.jl** is an open-source Julia package for simulating the GENERIC concentration-dependent nematohydrodynamic equations of liquid crystals, providing researchers with a reproducible, efficient, and flexible tool for studying binary mixtures of liquid crystals.

## Features

- Finite-difference upwind schemes for concentration and tensorial liquid-crystal ordering
- Galerkin Lattice Boltzmann methods for mass and momentum balance in combination with **Trixi.jl**
- Open-source, designed for reproducibility and community use.

## Installation

From a [Julia](https://julialang.org/downloads/) REPL session run

```
julia> ]

pkg> dev https://github.com/depablogroup/LiquidCrystals.jl.git
```


## Dependencies

The Finite Element aspect will be handled through `Gridap.jl`

## Usage

Examples and notebooks demonstrating the library can be found in the `examples/notebooks` folder.

## Examples

### Binary mixture with topological defects
The figure below shows a simulated binary mixture: two isotropic droplets immersed in a nematic environment.  
 

<p align="center">
  <img src="assets/Picture8b.png" width="500"/>
  <br><i>Figure: Two isotropic droplets immersed in a nematic environment. Director vectors (white lines) and scalar order parameter (color map 0–0.75).</i>
</p>

---

### Droplet under semi-Couette flow
The figures below show the evolution of an axial droplet immersed in an isotropic fluid under semi-Couette flow.  
- Left: director vectors (gray lines, scaled according to order parameter values) and scalar order parameter (color map 0–0.8).  
- Top right: concentration profile (color map 0–1).  
- Bottom right: velocity profile (x-component from -0.07 to 0.07).  

<p align="center">
  <img src="assets/Picture9.png" width="220"/>
  <img src="assets/Picture10.png" width="220"/>
  <img src="assets/Picture11.png" width="220"/>
  <img src="assets/Picture12.png" width="220"/>
  <br><i>Figures: Time evolution of an axial droplet under semi-Couette flow. Successive time steps from initial conditions to final state.</i>
</p>

---

### Droplet under parabolic flow
The figure below shows a snapshot of an axial droplet simulation immersed in an isotropic fluid under parabolic (Poiseuille-like) flow.  

<p align="center">
  <img src="assets/Picture16.png" width="500"/>
  <br><i>Figure: Axial droplet under parabolic flow. Director vectors (gray lines), scalar order parameter, concentration and velocity profile shown.</i>
</p>

---

### Active nematic
The figure below shows a snapshot of an active nematic system.  

<p align="center">
  <img src="assets/Picture22.png" width="500"/>
  <br><i>Figure: Active nematic snapshot. Director vectors, scalar order parameter and velocity field shown.</i>
</p>

The video below shows the complete dynamics:

<p align="center">
  <a href="assets/Video6_small.mp4">
    <img src="assets/Picture22.png" width="400" alt="Active nematic video thumbnail">
    <br>Click to watch the video
  </a>
</p>


---

## Citation

If you use this package in your research, please cite the repository and relevant publications associated with LiquidCrystals.jl, including Jonathan Salmerón's PhD thesis: On the Dynamics of Active and Multi-Component Liquid Crystals (The University of Chicago 2024, https://doi.org/10.6082/uchicago.13995)
