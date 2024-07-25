using Documenter, DocumenterTools

makedocs(
    sitename = "Continuum Simulations of Liquid Crystals",
    modules = [LiquidCrystals],
    doctest = true,
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = Any[
        "Home" => "index.md",
        "Installation" => "user/install.md",
        "The science behind DOLCE" => [
            "intro/lc_intro.md",
            "intro/fem_intro.md",
            "intro/relax_gl.md",
            "intro/relax_mc.md"
            ],
        "User Guide" => [
            "user/initial.md",
            "user/output.md"
        ],
        "Examples" => ["overview.md"],
        "API Reference" => ["lib/public.md"] 
    ],

    ) 