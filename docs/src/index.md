```@meta
CurrentModule = LiquidCrystals
DocTestSetup = quote
    using LiquidCrystals
end
```

# DOLCE.jl Documentation
*A versatile package for Continuum Simulations of Liquid Crystals in Julia*

## Introduction
*Dynamics and Order in Liquid Crystalline Environments* `DOLCE` is a simulation package for continuum simulations of Liquid Crystals. In particular, it provides implementations of methods developed by the de Pablo Lab at the University of Chicago. The main functionalities of this package include:

* Ginzburg-Landau relaxation of Q-tensor field
* Artificial annealing of Q-tensor field
* Finite element discretization of geometries
* Postprocessing programs to calculate different order parameters and polarized optical microscopy (POM) images

The best way to get started with `DOLCE` is by working through the documented examples.

!!! note
    `DOLCE` is still under development. If you find a bug, or have ideas for improvements, feel free to open an issue or make a pull request.

## Installation

You can install DOLCE from the Pkg REPL (press `]` in the Julia REPL to eneter `pkg>` mode):
```
pkg> add https://github.com/depablogroup/LiquidCrystals.jl.git
```

!!! note
    Alternative installation method:
    ```julia
    julia> import Pkg; Pkg.pkg"add https://github.com/depablogroup/LiquidCrystals.jl.git"
    ```
To load the package, use 
```julia
using DOLCE
```
## Debugging information

## Credits
The main developers of DOLCE.jl are:
* Viviana Palacio-Betancur
* Jonathan Salmerón-Hernández
* [Dr. Pablo Zubieta Rico](https://github.com/pabloferz)

The following people have contributed to the development of computational methods implemented within the [de Pablo Group](https://pme.uchicago.edu/group/de-pablo-group) at The University of Chicago:
* Prof. Juan P. Hernández-Ortiz
* Prof. Julio C. Armas-Pérez
* Dr. Alejandro Londoño-Hurtado
* Prof. José A. Martínez-González
* Dr. Ye Zhou
* Dr. Tyler Roberts
* Dr. Mohammad Rahimi

All current contributors are affiliated with the Pritzker School of Molecular Engineering at the University of Chicago and all the work here has been supervised by [Prof. Juan J. de Pablo](https://pme.uchicago.edu/faculty/juan-de-pablo).
