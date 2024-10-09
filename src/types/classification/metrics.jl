
function informedness(ŷ, y)
  cm = SM.confusion_matrix(ŷ, y)
  n = sum(cm.mat)
  prevalances = (sum(cm.mat, dims=1)./n)[1, :]
  tpr = SM.MulticlassTruePositiveRate(; average=SM.NoAvg(), return_type=Vector)
  tp_rates = tpr(cm)
  tnr = SM.MulticlassTrueNegativeRate(; average=SM.NoAvg(), return_type=Vector)
  tn_rates = tnr(cm)
  class_informedness = tp_rates + tn_rates .- 1
  return sum(class_informedness .* prevalances)
end


SMB.is_measure(m::typeof(informedness)) = true
SMB.kind_of_proxy(m::typeof(informedness)) = LearnAPI.LiteralTarget()


function markedness(ŷ, y)
  cm = SM.confusion_matrix(ŷ, y)
  n = sum(cm.mat)
  biases = (sum(cm.mat, dims=2)./n)[:, 1]
  ppr = SM.MulticlassPositivePredictiveValue(; average=SM.NoAvg(), return_type=Vector)
  pp_rates = ppr(cm)
  npr = SM.MulticlassNegativePredictiveValue(; average=SM.NoAvg(), return_type=Vector)
  np_rates = npr(cm)
  class_markedness = pp_rates + np_rates .- 1
  return sum(class_markedness .* biases)
end

SMB.is_measure(m::typeof(markedness)) = true
SMB.kind_of_proxy(m::typeof(markedness)) = LearnAPI.LiteralTarget()


abstract type ClassificationPredictionType end

struct LeafLabel <: ClassificationPredictionType end
struct LeafDistribution <: ClassificationPredictionType end

abstract type ClassificationFitnessType end

struct BinaryF1ScoreFitness <: ClassificationFitnessType end
struct MultiF1ScoreFitness <: ClassificationFitnessType end
struct AccuracyFitness <: ClassificationFitnessType end
struct BalancedAccuracyFitness <: ClassificationFitnessType end
struct MatthewsCorrelationFitness <: ClassificationFitnessType end
struct InformednessFitness <: ClassificationFitnessType end
struct MarkednessFitness <: ClassificationFitnessType end

function fitness_type_function(fitness_type::Type{T}) where {T<:ClassificationFitnessType}
  T <: ClassificationFitnessType || error("`fitness_type` = `$(fitness_type)` is not a recognised ClassificationFitnessMetric.")

  if T == BinaryF1ScoreFitness
    return SM.FScore()
  elseif T == MultiF1ScoreFitness
    return SM.MulticlassFScore()
  elseif T == AccuracyFitness
    return SM.Accuracy()
  elseif T == BalancedAccuracyFitness
    return SM.BalancedAccuracy()
  elseif T == MatthewsCorrelationFitness
    return SM.MatthewsCorrelation()
  elseif T == InformednessFitness
    return informedness
  elseif T == MarkednessFitness
    return markedness
  else
    error("`metric_type` = `$(fitness_type)` not recognised")
  end
end

abstract type PenaltyType end

struct DepthPenalty <: PenaltyType end
struct NodePenalty <: PenaltyType end


"""
    fitness(tree::ClassificationTree; kwargs...)

Compute raw fitness of `tree` without a penalty.

# Keyword Arguments
- `fitness_function::Any`: any function that implements `StatisticalMeasuresBase.is_measure()`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::ClassificationFitnessType=InformednessFitness`: which fitness metric to evaluate the tree with. 
- `label_type::ClassificationOutcomeType=LeafLabel`: how outcomes in leaf nodes should be predicted.
"""
function fitness(tree::ClassificationTree; kwargs...)
  fitness_function = get(kwargs, :fitness_function, fitness_type_function(get(kwargs, :fitness_function_type, InformednessFitness)))
  SMB.is_measure(fitness_function) || error("Given `fitness_function` is not a valid statistical measure.")
  label_type = get(kwargs, :label_type, LeafLabel)

  ŷ = label_type == LeafDistribution ? mode.(outcome_probability_predictions(tree)) : outcome_class_predictions(tree)
  y = tree.targets

  return fitness_function(ŷ, y)
end

"""
    penalised_fitness(tree::ClassificationTree; kwargs...)

Compute fitness of `tree` with a penalty.

# Keyword Arguments
- `fitness_function::Any`: any function that implements `StatisticalMeasuresBase.is_measure()`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::ClassificationFitnessType=InformednessFitness`: which fitness metric to evaluate the tree with. 
- `label_type::ClassificationOutcomeType=LeafLabel`: how outcomes in leaf nodes should be predicted.
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `max_depth::Int=tree.maxdepth`: max depth of tree used for penalty calculation.

"""
function penalised_fitness(tree::ClassificationTree; kwargs...)
  penalty_type = get(kwargs, :penalty_type, DepthPenalty)
  penalty_type <: PenaltyType || error("`penalty_type` - $(penalty_type) not recognised")
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  max_depth = get(kwargs, :max_depth, tree.max_depth)

  raw_fitness = fitness(tree; kwargs)
  penalty = penalty_type == DepthPenalty ?
            (AT.treeheight(tree) / max_depth) :
            (length(keys(tree.nodemap)) / 2^(max_depth))
  raw_penalised_fitness = raw_fitness - penalty_weight * penalty
  rescaled_penalised_fitness = (1 + exp(-1)) / (1 + exp(-1 * (raw_penalised_fitness / 1)))

  return (raw=raw_fitness, penalised=raw_penalised_fitness, rescaled=rescaled_penalised_fitness)
end


"""
    prune(tree::ClassificationTree; kwargs...)

Prune a copy of `tree` and return pruned tree.

# Keyword Arguments
- `max_depth::Int=tree.maxdepth`: depth of pruned tree.

"""
function prune(tree::ClassificationTree; kwargs...)
  tcopy = copy(tree)
  return prune!(tcopy, kwargs...)
end

"""
    prune!(tree::ClassificationTree; kwargs...)

Prune `tree` in place and return pruned tree.

# Keyword Arguments
- `max_depth::Int=tree.maxdepth`: depth of pruned tree.

"""
function prune!(tree::ClassificationTree; kwargs...)
  max_depth = get(kwargs, :max_depth, tree.max_depth)

  random_outcome = create_outcome_randomiser(tree)

  function process_node(node::ClassificationTreeNode)
    current_node_level = nodelevel(node)
    original_node_number = nodenumber(tree, node)
    original_node_mask = tree.nodemap[original_node_number].rowmask
    if current_node_level + 1 > max_depth && isbranch(node)
      new_outcome = random_outcome(original_node_mask)
      new_node = leaf(ClassificationTreeNode, new_outcome; attribute_labels_dict=node.attribute_labels)

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
    evaluate_fitness(trees::Vector{ClassificationTree}; kwargs...)

Evaluate the fitness of each tree in `trees` and returns vector of `FitnessValues`.

# Keyword Arguments
- `fitness_function::Any`: any function that implements `StatisticalMeasuresBase.is_measure()`, if `nothing` uses the `fitness_function_type` to create one.
- `fitness_function_type::ClassificationFitnessType=InformednessFitness`: which fitness metric to evaluate the tree with. 
- `label_type::ClassificationOutcomeType=LeafLabel`: how outcomes in leaf nodes should be predicted.
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `max_depth::Int`: max depth of tree used for penalty calculation, defaults to the tree's `max_depth`.

"""
function evaluate_fitness(trees::Vector{ClassificationTree}; kwargs...)
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
