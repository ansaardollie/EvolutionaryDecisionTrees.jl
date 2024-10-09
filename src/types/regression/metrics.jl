function R²(ŷ, y)
  R_total = sum((y .- mean(y)) .^ 2)
  R_res = sum((y - ŷ) .^ 2)

  return 1 - (R_res / R_total)
end

function create_adjusted_R²(tree::RegressionTree)
  n = nrow(tree.features)
  p = length(keys(tree.nodemap))

  df_total = n - 1
  df_tree = n - p - 1

  function adjusted_R²(ŷ, y)
    R_total = sum((y .- mean(y)) .^ 2)
    R_res = sum((y - ŷ) .^ 2)

    return 1 - ((R_res / df_tree) / R_total / (df_total))
  end

  return adjusted_R²
end

abstract type RegressionFitnessMetric end

struct RSquaredFitness <: RegressionFitnessMetric end
struct AdjustedRSquaredFitness <: RegressionFitnessMetric end

function fitness_function(tree::RegressionTree, metric_type::Type{T}) where {T<:RegressionFitnessMetric}
  T <: RegressionFitnessMetric || error("`metric_type` = `$(metric_type)` is not a recognised RegressionFitnessMetric.")

  if T <: RSquaredFitness
    return R²
  elseif T <: AdjustedRSquaredFitness
    return create_adjusted_R²(tree)
  else
    error("`metric_type` = `$(metric_type)` not recognised")
  end
end

function fitness(tree::RegressionTree; kwargs...)
  metric = get(kwargs, :metric, fitness_function(tree, get(kwargs, :metric_type, RSquaredFitness)))

  ŷ = outcome_predictions(tree)
  y = tree.targets

  return metric(ŷ, y)
end

function penalised_fitness(tree::RegressionTree; kwargs...)
  penalty = get(kwargs, :penalty, DepthPenalty)
  penalty <: PenaltyType || error("`penalty` - $(penalty) not recognised")
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  maxdepth = get(kwargs, :maxdepth, tree.maxdepth)
  raw_values = get(kwargs, :raw, false)
  fv = fitness(tree; kwargs...)
  raw_penalised_fitness = fv - penalty_weight * (penalty == DepthPenalty ? (AT.treeheight(tree.root) / maxdepth) : (length(keys(tree.nodemap)) / 2^(maxdepth)))

  return (raw_values ? raw_penalised_fitness : (1 + exp(-1)) / (1 + exp(-1 * (raw_penalised_fitness / 1))))
end


function prune(tree::RegressionTree; maxdepth::Int=tree.maxdepth)
  tcopy = copy(tree)
  return prune!(tcopy, maxdepth=maxdepth)
end

function prune!(tree::RegressionTree; maxdepth::Int=tree.maxdepth)

  leaf_prediction = create_prediction_selector(tree)

  function process_node(node::RegressionTreeNode)
    current_node_level = nodelevel(node)
    original_node_number = nodenumber(tree, node)
    original_node_mask = tree.nodemap[original_node_number].rowmask
    if current_node_level + 1 > maxdepth && isbranch(node)
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


function evaluate_fitness(trees::Vector{RegressionTree}; kwargs...)
  n = length(trees)
  fitchannel = Channel(n)

  all_fitnesses = Vector{Float64}(undef, n)

  @threads for i in 1:n
    fv = penalised_fitness(trees[i]; kwargs...)
    put!(fitchannel, i => fv)
  end

  for i in 1:n
    out = take!(fitchannel)
    all_fitnesses[out[1]] = out[2]
  end

  return all_fitnesses
end
