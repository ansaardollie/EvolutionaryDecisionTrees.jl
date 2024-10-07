
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


SMB.is_measure(informedness) = true
SMB.kind_of_proxy(informedness) = LearnAPI.LiteralTarget()


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

SMB.is_measure(markedness) = true
SMB.kind_of_proxy(markedness) = LearnAPI.LiteralTarget()


abstract type TargetType end

struct LeafLabel <: TargetType end
struct LeafDistribution <: TargetType end

abstract type FitnessMetric end

struct F1ScoreFitness <: FitnessMetric end
struct AccuracyFitness <: FitnessMetric end
struct BalancedAccuracyFitness <: FitnessMetric end
struct MatthewsCorrelationFitness <: FitnessMetric end
struct InformednessFitness <: FitnessMetric end
struct MarkednessFitness <: FitnessMetric end

function metric_function(tree::ClassificationTree, metric_type::Type{A}) where {A<:FitnessMetric}
  if metric_type == F1ScoreFitness
    return ST.elscitype(tree.targets) <: Union{Missing,ST.OrderedFactor{2}} ? SM.FScore() : SM.MulticlassFScore()
  elseif metric_type == AccuracyFitness
    return SM.Accuracy()
  elseif metric_type == BalancedAccuracyFitness
    return SM.BalancedAccuracy()
  elseif metric_type == MatthewsCorrelationFitness
    return SM.MatthewsCorrelation()
  elseif metric_type == InformednessFitness
    return informedness
  elseif metric_type == MarkednessFitness
    return markedness
  else
    error("`metric_type` = `$(metric_type)` not recognised")
  end
end

abstract type PenaltyType end

struct DepthPenalty <: PenaltyType end
struct NodePenalty <: PenaltyType end

function fitness(
  tree::ClassificationTree;
  metric,
  target::Type{T}=LeafLabel
) where {T<:TargetType}
  SMB.is_measure(metric) || error("Given metric is not a valid statistical measure.")
  ŷ = target == LeafDistribution ? mode.(outcome_probability_predictions(tree)) : outcome_class_predictions(tree)
  y = tree.targets
  return metric(ŷ, y)
end

function fitness(
  tree::ClassificationTree,
  metric_type::Type{A};
  target::Type{B}=LeafLabel
) where {A<:FitnessMetric,B<:TargetType}
  ST.elscitype(tree.targets) <: ST.Finite || error("Tree's target is not finite")

  return fitness(tree; metric=metric_function(tree, metric_type), target=target)
end


function penalised_fitness(
  tree::ClassificationTree,
  fitness_value::Float64;
  penalty::Type{A}=DepthPenalty,
  penalty_weight::Float64=0.05,
  maxdepth::Int=tree.maxdepth
) where {A<:PenaltyType}
  if penalty == DepthPenalty
    return fitness_value - penalty_weight * (AT.treeheight(tree.root) / maxdepth)
  elseif penalty == NodePenalty
    return fitness_value - penalty_weight * (length(keys(tree.nodemap)) / (2^(maxdepth)))
  else
    error("`penalty` = `$(penalty)` not recognised")
  end
end

function penalised_fitness(
  tree::ClassificationTree,
  metric_type::Type{A};
  target::Type{B}=LeafLabel,
  penalty_weight::Float64=0.05,
  penalty::Type{C}=DepthPenalty,
  maxdepth::Int=tree.maxdepth
) where {A<:FitnessMetric,B<:TargetType,C<:PenaltyType}
  tf = fitness(tree, metric_type; target=target)
  return penalised_fitness(tree, tf; penalty=penalty, penalty_weight=penalty_weight, maxdepth=maxdepth)
end

function penalised_fitness(
  tree;
  metric,
  target::Type{A}=LeafLabel,
  penalty::Type{B}=DepthPenalty,
  penalty_weight::Float64=0.05,
  maxdepth::Int=tree.maxdepth
) where {A<:TargetType,B<:PenaltyType}
  tf = fitness(tree; metric=metric, target=target)
  return penalised_fitness(tree, tf; penalty=penalty, penalty_weight=penalty_weight, maxdepth=maxdepth)
end


function prune(tree::ClassificationTree; maxdepth::Int=tree.maxdepth)
  tcopy = copy(tree)
  return prune!(tcopy)
end

function prune!(tree::ClassificationTree; maxdepth::Int=tree.maxdepth)
  random_outcome = create_outcome_randomiser(tree)

  random_outcome = create_outcome_randomiser(tree)

  function process_node(node::ClassificationTreeNode)
    current_node_level = nodelevel(node)
    original_node_number = nodenumber(tree, node)
    original_node_mask = tree.nodemap[original_node_number].rowmask
    if current_node_level + 1 > maxdepth && isbranch(node)
      new_outcome = random_outcome(original_node_mask)
      node.outcome = new_outcome
      node.threshold = nothing
      node.attribute = nothing
      node.left[] = nothing
      node.right[] = nothing
    elseif isbranch(node)
      process_node(node.left[])
      process_node(node.right[])
    end
  end

  process_node(tree.root)

  reset_nodemap!(tree)
  return tree
end

function evaluate_fitness(trees::Vector{ClassificationTree}; kwargs...)
  target = get(kwargs, :target, LeafLabel)
  penalty = get(kwargs, :penalty, DepthPenalty)
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  maxdepth = get(kwargs, :maxdepth, trees[1].maxdepth)
  metricfunc = get(kwargs, :metric, metric_function(trees[1], get(kwargs, :metric_type, InformednessFitness)))
  verbose = get(kwargs, :verbose, false)

  n = length(trees)
  fitchannel = Channel(n)

  all_fitnesses = Vector{Float64}(undef, n)
  if verbose
    @showprogress desc = "Evaluating Fitness Of Trees..." barglyphs = BarGlyphs("[=> ]") @threads for i in 1:n
      fv = penalised_fitness(trees[i]; metric=metricfunc, target=target, penalty=penalty, penalty_weight=penalty_weight, maxdepth=maxdepth)
      put!(fitchannel, i => fv)
    end
  else
    @threads for i in 1:n
      fv = penalised_fitness(trees[i]; metric=metricfunc, target=target, penalty=penalty, penalty_weight=penalty_weight, maxdepth=maxdepth)
      put!(fitchannel, i => fv)
    end
  end


  for i in 1:n
    out = take!(fitchannel)
    all_fitnesses[out[1]] = out[2]
  end

  return all_fitnesses
end
