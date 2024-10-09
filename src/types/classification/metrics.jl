
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


abstract type TargetType end

struct LeafLabel <: TargetType end
struct LeafDistribution <: TargetType end

abstract type FitnessMetric end

struct BinaryF1ScoreFitness <: FitnessMetric end
struct MultiF1ScoreFitness <: FitnessMetric end
struct AccuracyFitness <: FitnessMetric end
struct BalancedAccuracyFitness <: FitnessMetric end
struct MatthewsCorrelationFitness <: FitnessMetric end
struct InformednessFitness <: FitnessMetric end
struct MarkednessFitness <: FitnessMetric end

function metric_function(metric_type::DataType)
  metric_type <: FitnessMetric || error("`metric_type` = `$(metric_type)` is not a recognised FitnessMetric.")

  if metric_type == BinaryF1ScoreFitness
    return SM.FScore()
  elseif metric_type == MultiF1ScoreFitness
    return SM.MulticlassFScore()
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


function fitness(tree::ClassificationTree; kwargs...)
  metric = get(kwargs, :metric, metric_function(get(kwargs, :metric_type, InformednessFitness)))
  SMB.is_measure(metric) || error("Given metric is not a valid statistical measure.")
  target = get(kwargs, :target, LeafLabel)
  ŷ = target == LeafDistribution ? mode.(outcome_probability_predictions(tree)) : outcome_class_predictions(tree)
  y = tree.targets
  return metric(ŷ, y)
end

"""
    penalised_fitness(tree::ClassificationTree; kwargs...)

Compute fitness of `tree` with a penalty.

# Keyword Arguments
- `metric_type::FitnessMetric=InformednessFitness`: which fitness metric to evaluate the tree with. 
- `metric::Any`: any function that implements `StatisticalMeasuresBase.is_measure()`, if not provided uses the `metric_type` to create one.
- `penalty::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `maxdepth::Int=tree.maxdepth`: max depth of tree used for penalty calculation.

"""
function penalised_fitness(tree::ClassificationTree; kwargs...)
  penalty = get(kwargs, :penalty, DepthPenalty)
  penalty <: PenaltyType || error("`penalty` - $(penalty) not recognised")
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  maxdepth = get(kwargs, :maxdepth, tree.maxdepth)
  fv = fitness(tree; kwargs)
  raw_penalty = fv - penalty_weight * (penalty == DepthPenalty ? (AT.treeheight(tree) / maxdepth) : (length(keys(tree.nodemap)) / 2^(maxdepth)))

  return 1 / (1 + exp(-1 * raw_penalty))
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

function evaluate_fitness(trees::Vector{ClassificationTree}; kwargs...)
  target = get(kwargs, :target, LeafLabel)
  penalty = get(kwargs, :penalty, DepthPenalty)
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  maxdepth = get(kwargs, :maxdepth, trees[1].maxdepth)
  metricfunc = get(kwargs, :metric, metric_function(get(kwargs, :metric_type, InformednessFitness)))


  n = length(trees)
  fitchannel = Channel(n)

  all_fitnesses = Vector{Float64}(undef, n)

  @threads for i in 1:n
    fv = penalised_fitness(trees[i]; metric=metricfunc, target=target, penalty=penalty, penalty_weight=penalty_weight, maxdepth=maxdepth)
    put!(fitchannel, i => fv)
  end

  for i in 1:n
    out = take!(fitchannel)
    all_fitnesses[out[1]] = out[2]
  end

  return all_fitnesses
end
