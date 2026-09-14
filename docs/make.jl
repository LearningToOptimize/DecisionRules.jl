using Documenter
using Literate
using DecisionRules

# Convert Literate.jl sources to markdown, in place, wherever they live: a case
# study that owns a runnable walkthrough keeps it beside its prose rather than in
# a separate examples pile.
for dir in (joinpath(@__DIR__, "src", "examples"),
            joinpath(@__DIR__, "src", "casestudies", "hydro"))
    isdir(dir) || continue
    for file in readdir(dir)
        endswith(file, ".jl") || continue
        Literate.markdown(joinpath(dir, file), dir; documenter=true, credit=false)
    end
end

makedocs(;
    modules=[DecisionRules],
    sitename="DecisionRules.jl",
    warnonly=true,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://LearningToOptimize.github.io/DecisionRules.jl",
        size_threshold=300 * 1024,
    ),
    pages=[
        "Home" => "index.md",
        "Part I — Theory" => [
            "Multistage stochastic optimization" => "theory/multistage.md",
            "The TS-DDR framework" => "algorithm.md",
            "Stochastic dual dynamic programming" => "theory/sddp.md",
            "Extensions: mixed gradients, critics, risk" => "theory/extensions.md",
        ],
        "Part II — Package Guide" => [
            "Getting started" => "guide/getting_started.md",
            "Uncertainty sampling" => "sampling.md",
            "Gradient fallback" => "gradient_fallback.md",
            "GPU acceleration" => "gpu_acceleration.md",
            "API reference" => "api.md",
        ],
        "Part III — Case Studies" => [
            "Battery-storage AC-OPF" => "casestudies/battery_storage_opf.md",
            "Long-term hydrothermal planning" => [
                "Overview" => "casestudies/hydro/index.md",
                "The problem" => "casestudies/hydro/problem.md",
                "Valuing water: two approaches" => "casestudies/hydro/method.md",
                "Results" => "casestudies/hydro/results.md",
                "Walkthrough" => "casestudies/hydro/walkthrough.md",
            ],
            "Rocket control" => "examples/rocket.md",
            "Stochastic lot-sizing (integer variables)" => "examples/inventory.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/LearningToOptimize/DecisionRules.jl",
    devbranch="main",
)
