
include("../../requirements.jl")

include("../../types/global.jl")
include("../../types/classification/nodes.jl")
include("../../types/classification/tree.jl")
include("../../types/classification/metrics.jl")

include("../../types/regression/nodes.jl")
include("../../types/regression/tree.jl")
include("../../types/regression/metrics.jl")

include("../../functions/genetics.jl")
include("../../functions/plot.jl")


include("../../mlj/mlj.jl")


struct RegressionForest <: AbstractDecisionForest{RegressionTree}
  trees::Vector{RegressionTree}
  features::AbstractDataFrame
  targets::AbstractVector{Float64}
  bag_colmask::Vector{BitVector}
  bag_rowmask::BitMatrix
  bag_samples::Matrix{Int}
  num_trees::Int
  num_sampled_features::Int
end


kwargs = Dict{Any,Any}(
  :generation_size => 100,
  :num_trees => 1,
  :max_depth => 7,
  :split_probability => 0.5,
  :leaf_prediction_type => MeanPrediction,
  :num_generations => 50,
  :max_generations_stagnant => 50,
  :fitness_function_type => AdjustedRSquaredFitness,
  :penalty_type => NodePenalty,
  :penalty_weight => 0.2,
  :elite_proportion => 0.2,
  :num_mutations => 1,
  :mutation_probability => 1.0
)
"""
    regression_forest(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)

# Keyword Arguments (For The Forest)
- `num_trees::Int=500`: how many decision trees to create.
- `num_sampled_features::Int=Int(floor(sqrt(ncol(X))))`: how many features to randomly provide to each individual decision tree.

# Keyword Arguments (For The Trees)
- `generation_size::Int=500`: how many trees are part of each generation.
- `max_depth::Int=5`: maximum depth of generated tree.
- `split_probability::Float64=0.5`: probability of splitting a node into another branch instead of a leaf during population initialisation.
- `leaf_prediction_function::Union{Nothing, Function}=nothing`: function that computes the prediction for leaf 
based on the subset of `y` and `X` that ends up in that leaf, if `nothing` uses `leaf_prediction_type` to determine the function.
- `leaf_prediction_type::Type{<:LeafPredictionType}=MeanPrediction`: the type of leaf prediction to use.

# Keyword Arguments (For Genetic Alogrithm)
- `num_generations::Int=200`: number of generations to train for.
- `max_generations_stagnant::Int=Int(floor(num_generations * 0.2))`: stopping criterion, stop training if best tree's fitness does not improve for this number of generations.
- `fitness_function::Any`: appropriate fitness function for the type of decision trees provided, if `nothing` used `fitness_function_type` to create one.
- `fitness_function_type::Union{ClassificationFitnessType, RegressionFitnessType}`: which fitness metric to evaluate the tree with. 
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `elite_proportion::Float64=0.3`: proportion of top-ranked trees to propogate unchanged into next generation.
- `num_mutations::Int=1`: number of nodes to mutate.
- `mutation_probability::Float64=0.5`: probability of randomly selected node being mutated.
- `max_depth::Int`: max depth of tree used for penalty calculation, defaults to the tree's `max_depth`.

"""
function regression_forest_v1(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)
  n = nrow(X)
  p = ncol(X)
  num_trees = get(kwargs, :num_trees, 500)
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  generation_size = get(kwargs, :generation_size, 500)

  trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)


  treechan = Channel{Any}(num_trees)

  @showprogress desc = "Growing forest..." barglyphs = BarGlyphs("[=> ]") @threads for i in 1:num_trees
    Xsub = view(X, bag_samples[:, i], bag_features[:, i])
    Ysub = view(y, bag_samples[:, i])

    initial_tree_population = random_regression_trees(generation_size, Xsub, Ysub; kwargs...)

    best_tree, _ = train(initial_tree_population; verbosity=0, kwargs...)

    put!(treechan, (i => best_tree))
  end

  for i in 1:num_trees
    chanout = take!(treechan)
    trees[chanout[1]] = chanout[2]
  end

  return trees
end

function regression_forest_v2(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)
  n = nrow(X)
  p = ncol(X)
  num_trees = get(kwargs, :num_trees, 500)
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  generation_size = get(kwargs, :generation_size, 500)

  trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)

  treechan = Channel{Any}(num_trees)
  tasks = Vector{Task}(undef, num_trees)
  prog = Progress(num_trees, desc="Growing Forest V2", barglyphs=BarGlyphs("[=> ]"))

  for i in 1:num_trees
    tasks[i] = @spawn begin
      Xsub = view(X, bag_samples[:, i], bag_features[:, i])
      Ysub = view(y, bag_samples[:, i])

      initial_tree_population = random_regression_trees(generation_size, Xsub, Ysub; kwargs...)

      best_tree, _ = train(initial_tree_population; verbosity=0, kwargs...)

      put!(treechan, (i => best_tree))
      next!(prog)
    end
  end

  wait.(tasks)
  finish!(prog)

  for i in 1:num_trees
    chanout = take!(treechan)
    trees[chanout[1]] = chanout[2]
  end

  return trees
end



X = DataFrame(rand(Uniform(-5, 5), 1000, 10), :auto)
y = map(row -> rand(Gamma(exp(1 / (2 + sum(abs.(Vector(row)) .* collect(1:10)))), 10)), eachrow(X))

function test_pm(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)

  n = nrow(X)
  p = ncol(X)

  num_trees = get(kwargs, :num_trees, 500)
  @show num_trees
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  @show num_sampled_features
  generation_size = get(kwargs, :generation_size, 500)
  @show generation_size

  @show kwargs

  forest_trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)

  treechan = Channel{Any}(num_trees)

  pg = Progress(num_trees, desc="Growing Forest V2", barglyphs=BarGlyphs("[=> ]"))
  tasks = Vector{Task}(undef, num_trees)
  for i in 1:num_trees
    tasks[i] = @spawn begin
      tree_population = random_regression_trees(generation_size, X, y; verbosity=0)
      best_tree, _ = train(tree_population; kwargs..., verbosity=0)
      put!(treechan, i => best_tree)
      @spawn :interactive next!(pg)
    end
  end
  wait.(tasks)
  finish!(pg)

  for i in 1:num_trees
    chanout = take!(treechan)
    forest_trees[chanout[1]] = chanout[2]
  end

  return forest_trees
end


function test_pm2(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)

  n = nrow(X)
  p = ncol(X)

  num_trees = get(kwargs, :num_trees, 500)
  @show num_trees
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  @show num_sampled_features
  generation_size = get(kwargs, :generation_size, 500)
  @show generation_size

  @show kwargs

  forest_trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)

  treechan = Channel{Any}(num_trees)

  # pg = Progress(num_trees, desc="Growing Forest V2", barglyphs=BarGlyphs("[=> ]"))
  # tasks = Vector{Task}(undef, num_trees)
  # for i in 1:num_trees
  #   tasks[i] = @spawn begin
  #     tree_population = random_regression_trees(generation_size, X, y; verbosity=0)
  #     best_tree, _ = train(tree_population; kwargs..., verbosity=0)
  #     put!(treechan, i => best_tree)
  #     @spawn :interactive next!(pg)
  #   end
  # end
  # wait.(tasks)
  # finish!(pg)

  @showprogress desc = "Growing Tree..." barglyphs = BarGlyphs("[=> ]") @threads for i in 1:num_trees
    tree_population = random_regression_trees(generation_size, X, y; verbosity=0)
    best_tree, _ = train(tree_population; kwargs..., verbosity=0)
    put!(treechan, i => best_tree)
  end

  for i in 1:num_trees
    chanout = take!(treechan)
    forest_trees[chanout[1]] = chanout[2]
  end

  return forest_trees
end


function test_pm3(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)

  n = nrow(X)
  p = ncol(X)

  num_trees = get(kwargs, :num_trees, 500)
  @show num_trees
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  @show num_sampled_features
  generation_size = get(kwargs, :generation_size, 500)
  @show generation_size

  @show kwargs

  forest_trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)

  treechan = Channel{Any}(num_trees)

  pg = Progress(num_trees, desc="Growing Forest V2", barglyphs=BarGlyphs("[=> ]"))
  tasks = Vector{Task}(undef, num_trees)
  @sync for i in 1:num_trees
    @spawn begin
      tree_population = random_regression_trees(generation_size, X, y; verbosity=0)
      best_tree, _ = train(tree_population; kwargs..., verbosity=0)
      put!(treechan, i => best_tree)
      @spawn :interactive next!(pg)
    end
  end
  finish!(pg)

  for i in 1:num_trees
    chanout = take!(treechan)
    forest_trees[chanout[1]] = chanout[2]
  end

  return forest_trees
end



function test_pm4(X::AbstractDataFrame, y::AbstractVector{Float64}; kwargs...)

  n = nrow(X)
  p = ncol(X)

  num_trees = get(kwargs, :num_trees, 500)
  @show num_trees
  num_sampled_features = get(kwargs, :num_sampled_features, Int(floor(sqrt(p))))
  @show num_sampled_features
  generation_size = get(kwargs, :generation_size, 500)
  @show generation_size

  @show kwargs

  forest_trees = Vector{RegressionTree}(undef, num_trees)
  row_indices = collect(1:n)
  col_indices = collect(1:p)

  bag_samples = rand(row_indices, n, num_trees)
  bag_features = reduce(hcat, map(x -> sample(col_indices, (num_sampled_features), replace=false, ordered=true), 1:num_trees))

  bag_rowmask = zeros(Int, n, num_trees)
  bag_colmask = zeros(Int, p, num_trees)

  for i in 1:num_trees
    bag_rowmask[bag_samples[:, i], i] .= 1
    bag_colmask[bag_features[:, i], i] .= 1
  end

  bag_colmask = map(col -> BitVector(reshape(col, p)), eachcol(bag_colmask))
  bag_rowmask = BitMatrix(bag_rowmask)

  ThreadsX.foreach(referenceable(forest_trees), referenceable(collect(1:num_trees))) do ft, i
    tree_population = random_regression_trees(generation_size, X, y; verbosity=0)
    best_tree, _ = train(tree_population; kwargs..., verbosity=0)
    ft[] = best_tree
  end

  return forest_trees
end
