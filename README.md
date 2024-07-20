# LiquidsCrystals.jl

Library to solve the Landau-de Gennes model for nematic liquid crystals.

## Installation

From a [Julia](https://julialang.org/downloads/) REPL session run

```
julia> ]

pkg> dev https://github.com/depablogroup/LiquidCrystals.jl.git
```

## Dependencies

The Finite Element aspect will be handled through `FEniCS.jl`. This requires a prior installation of `FEniCS`, which in turn has been deprecated in favor of `FEniCSX`. Installation of legacy `FEniCS` is proven to be tricky. I've made attempts using:
    - Docker containers from the official source
    - Docker containers from archived repos in the community
        - Both of these result in a myriad of errors related to compatibility with old libraries, and deprecated formats of old Docker images.
    - Tried building directly from source, which requires compilation of each separate component of `FEniCS`-> FFC (compiler), Dolphin (solver), mshr (mesh generation). The installation of Dolphin depends on old packages of numpy and it's no longer supported by `FEniCS`. This is when I stopped. Community posts suggest usind an old Docker images of Dolphin but I'm not sure how to develop code that require Docker containers for one part of the package. 
    **Using Conda** This worked!
    #. Install a distribution of conda: anaconda, *miniconda*, miniforge
    #. Create a conda environment for fenics
    ```
        conda create --name fenicsconda
        conda fenicsconda activate
        conda install conda-forge::fenics
    ```
    Then, install `juliaup` by doing
    ```
        conda install -c conda-forge juliaup
    ````
    You can add the specific version of julia by typing `juliaup add <version>`. `julia-release` is the default.
    


## Usage

Look at the `examples/notbooks` folder to see the library in action.
