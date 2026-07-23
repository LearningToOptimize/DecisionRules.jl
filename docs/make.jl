using Documenter
using Literate
using DecisionRules

# Convert Literate.jl sources to markdown
examples_src = joinpath(@__DIR__, "src", "examples")
examples_out = joinpath(@__DIR__, "src", "examples")
for file in readdir(examples_src)
    endswith(file, ".jl") || continue
    Literate.markdown(
        joinpath(examples_src, file), examples_out;
        documenter=true, credit=false,
    )
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
            "Rocket control" => "examples/rocket.md",
            "Stochastic lot-sizing (integer variables)" => "examples/inventory.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/LearningToOptimize/DecisionRules.jl",
    devbranch="main",
)
