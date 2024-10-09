function R²(ŷ, y, arg...)
  R_total = sum((y .- mean(y)) .^ 2)
  R_res = sum((y - ŷ) .^ 2)

  return 1 - (R_res / R_total)
end

function adjusted_R²(ŷ, y, args...)
  tree = args[1]
  n = nrow(tree.features)
  p = length(keys(tree.nodemap))

  df_total = n - 1
  df_tree = n - p - 1

  R_total = sum((y .- mean(y)) .^ 2)
  R_res = sum((y - ŷ) .^ 2)

  return 1 - ((R_res / df_tree) / R_total / (df_total))

end



abstract type RegressionFitnessType end

struct RSquaredFitness <: RegressionFitnessType end
struct AdjustedRSquaredFitness <: RegressionFitnessType end

function fitness_type_function(metric_type::Type{T}) where {T<:RegressionFitnessType}
  T <: RegressionFitnessType || error("`metric_type` = `$(metric_type)` is not a recognised RegressionFitnessMetric.")

  if T <: RSquaredFitness
    return R²
  elseif T <: AdjustedRSquaredFitness
    return adjusted_R²
  else
    error("`metric_type` = `$(metric_type)` not recognised")
  end
end

"""
    fitness(tree::ClassificationTree; kwargs...)

Compute raw fitness of `tree` without a penalty.

# Keyword Arguments
- `fitness_function::Any`: any function that can compute regression fitness from `ŷ` and `y`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::RegressionFitnessType=RSquaredFitness`: which fitness metric to evaluate the tree with. 

"""
function fitness(tree::RegressionTree; kwargs...)
  fitness_function = get(kwargs, :fitness_function, fitness_type_function(get(kwargs, :fitness_function_type, RSquaredFitness)))

  ŷ = outcome_predictions(tree)
  y = tree.targets

  return fitness_function(ŷ, y, tree)
end

"""
    fitness(tree::ClassificationTree; kwargs...)

Compute fitness of `tree` with a penalty.

# Keyword Arguments
- `fitness_function::Any`: any function that can compute regression fitness from `ŷ` and `y`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::RegressionFitnessType=RSquaredFitness`: which fitness metric to evaluate the tree with.  
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `max_depth::Int=tree.maxdepth`: max depth of tree used for penalty calculation.

"""
function penalised_fitness(tree::RegressionTree; kwargs...)
  penalty_type = get(kwargs, :penalty_type, DepthPenalty)
  penalty_type <: PenaltyType || error("`penalty` - $(penalty_type) not recognised")
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  max_depth = get(kwargs, :max_depth, tree.max_depth)

  raw_fitness = fitness(tree; kwargs...)
  penalty = penalty_type == DepthPenalty ?
            (AT.treeheight(tree.root) / max_depth) :
            (length(keys(tree.nodemap)) / 2^(max_depth))

  raw_penalised_fitness = raw_fitness - penalty_weight * penalty

  rescaled_penalised_fitness = (1 + exp(-1)) / (1 + exp(-1 * (raw_penalised_fitness / 1)))

  return (raw=raw_fitness, penalised=raw_penalised_fitness, rescaled=rescaled_penalised_fitness)
end


"""
    prune(tree::RegressionTree; kwargs...)

Prune a copy of `tree` and return pruned tree.

# Keyword Arguments
- `max_depth::Int=tree.maxdepth`: depth of pruned tree.

"""
function prune(tree::RegressionTree; kwargs...)
  tcopy = copy(tree)
  return prune!(tcopy, kwargs...)
end

"""
    prune!(tree::ClassificationTree; kwargs...)

Prune `tree` in place and return pruned tree.

# Keyword Arguments
- `max_depth::Int=tree.maxdepth`: depth of pruned tree.

"""
function prune!(tree::RegressionTree; kwargs...)
  max_depth = get(kwargs, :max_depth, tree.max_depth)
  leaf_prediction = create_prediction_selector(tree)

  function process_node(node::RegressionTreeNode)
    current_node_level = nodelevel(node)
    original_node_number = nodenumber(tree, node)
    original_node_mask = tree.nodemap[original_node_number].rowmask
    if current_node_level + 1 > max_depth && isbranch(node)
      new_outcome = leaf_prediction(original_node_mask)
      new_node = leaf(RegressionTreeNode, new_outcome; attribute_labels_dict=node.attribute_labels)

      parent_node = node.parent[]
      if isleftchild(node)
        parent_node.left[] = nothing
        leftchild!(parent_node, new_node)
      else
        parent_node.right[] = nothing
        rightchild!(parent_node, new_node)
      end
    elseif isbranch(node)
      process_node(node.left[])
      process_node(node.right[])
    end
  end

  process_node(tree.root)

  reset_nodemap!(tree)
  return tree
end

"""
    evaluate_fitness(trees::Vector{RegressionTree}; kwargs...)

Evaluate the fitness of each tree in `trees` and returns vector of the following fitness values: raw_fitness, raw_penalised_fitness, rescaled_penalised_fitness.

# Keyword Arguments
- `fitness_function::Any`: any function that can compute regression fitness from `ŷ` and `y`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::RegressionFitnessType=RSquaredFitness`: which fitness metric to evaluate the tree with.
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `max_depth::Int`: max depth of tree used for penalty calculation, defaults to the tree's `max_depth`.

"""
function evaluate_fitness(trees::Vector{RegressionTree}; kwargs...)
  n = length(trees)
  fitchannel = Channel(n)

  tree_fitnesses = Vector{FitnessValues}(undef, n)

  @threads for i in 1:n
    all_fitness_values = penalised_fitness(trees[i]; kwargs...)
    put!(fitchannel, i => all_fitness_values)
  end

  for i in 1:n
    out = take!(fitchannel)
    tree_fitnesses[out[1]] = out[2]
  end

  return tree_fitnesses
end
