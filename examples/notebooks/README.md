# LiquidsCrystals.jl Example Notebooks

Collection of notebooks for learning how to use the library to build nematic liquid
crystals simulations.

## Getting started

Open a Julia REPL session from the `examples/notebooks` folder and run

```
julia> ]

(@v1.7) pkg> activate --temp

(jl_temp) pkg> dev https://github.com/depablogroup/LiquidCrystals.jl.git

(jl_temp) pkg> activate .

(notebooks) pkg> instantiate
```

This will take some time the first time you run it. After the process finishes, press
the Backspace key to go back to the `julia>` prompt. Then run

```
julia> using Pluto

julia> Pluto.run()
```

This will open a web browser window where you can open the notebook you wish to run.
