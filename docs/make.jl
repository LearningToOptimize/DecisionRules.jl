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
    ),
    pages=[
        "Home" => "index.md",
        "Algorithm" => "algorithm.md",
        "Gradient Fallback" => "gradient_fallback.md",
        "Uncertainty Sampling" => "sampling.md",
        "Examples" => [
            "Hydropower Scheduling" => "examples/hydro.md",
            "Rocket Control" => "examples/rocket.md",
            "Stochastic Lot-Sizing (Integer Variables)" => "examples/inventory.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/LearningToOptimize/DecisionRules.jl",
    devbranch="main",
)
