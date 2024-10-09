
mutable struct ClassificationTree <: AbstractDecisionTree{ClassificationTreeNode}
  root::ClassificationTreeNode
  features::AbstractDataFrame
  targets::CategoricalVector
  nodemap::NodeMap{ClassificationTreeNode}
  max_depth::Int
  leaf_predictor::Function
end

CategoricalArrays.levels(tree::ClassificationTree) = levels(tree.targets)

abstract type ClassificationPredictionType end

struct RandomClassPrediction <: ClassificationPredictionType end
struct ModePrediction <: ClassificationPredictionType end

function leaf_predictor(T::Type{<:ClassificationPredictionType})
  if T <: RandomClassPrediction
    return (target_subset, feature_subset) -> begin
      levels = target_subset.pool.levels
      return rand(levels)
    end
  elseif T <: ModePrediction
    return (target_subset, feature_subset) -> begin
      levels = target_subset.pool.levels
      counts = map(l -> count(==(l), target_subset), levels)
      max_count_idx = argmax(counts)
      return target_subset.pool[max_count_idx]
    end
  else
    error("ClassificationPredictionType $(T) not recognised")
  end
end


function create_prediction_selector(tree::ClassificationTree)
  y = tree.targets
  X = tree.features

  n_classes = length(y.pool.levels)
  function prediction(mask::Union{Nothing,BitVector})
    mask = isnothing(mask) ? BitVector(ones(length(y))) : mask
    y_subset = @view y[mask]
    X_subset = @view X[mask, :]
    return sum(mask) > 0 ? tree.leaf_predictor(y_subset, X_subset) : y.pool[rand(1:n_classes)]
  end

  return prediction
end

# function create_outcome_randomiser(tree::ClassificationTree)
#   targets = tree.targets
#   function random_outcome(mask::Union{Nothing,BitArray{1}})
#     indices = isnothing(mask) ? BitVector(repeat([1], length(targets))) : mask
#     available_values = @view targets[indices]
#     return length(available_values) > 0 ? rand(available_values) : CategoricalValue(rand(levels(tree)), tree.targets)
#   end
#   return random_outcome
# end




"""
    random_classification_tree(X::AbstractDataFrame, y::CategoricalVector; kwargs...)

# Keyword Arguments
- `max_depth::Int=5`: maximum depth of generated tree.
- `split_probability::Float64=0.5`: probability of splitting a node into another branch instead of a leaf.
- `leaf_prediction_function::Union{Nothing, Function}=nothing`: function that computes the prediction for leaf 
based on the subset of `y` and `X` that ends up in that leaf, if `nothing` uses `leaf_prediction_type` to determine the function.
- `leaf_prediction_type::Type{<:ClassificationPredictionType}=RandomPrediction`: the type of leaf prediction to use.
"""
function random_classification_tree(X::AbstractDataFrame, y::CategoricalVector; kwargs...)
  max_depth = get(kwargs, :max_depth, 5)
  split_probability = get(kwargs, :split_probability, 0.5)
  leaf_prediction_type = get(kwargs, :leaf_prediction_type, RandomValuePrediction)
  leaf_prediction_type <: RegressionPredictionType || error("`leaf_prediction_type` = `$(leaf_prediction_type)` is not a recognised LeafPredictionType")
  leaf_prediction_function = get(kwargs, :leaf_prediction_function, leaf_predictor(leaf_prediction_type))

  attribute_labels = Dict(pairs(names(X)))

  root_attr = rand(1:ncol(X))
  root_threshold = rand(@view X[!, root_attr])

  root = branch(ClassificationTreeNode, root_attr, root_threshold, attribute_labels_dict=attribute_labels)
  leftmask = (@view X[!, root_attr]) .< root_threshold
  rightmask = (@view X[!, root_attr]) .>= root_threshold

  children_to_randomise = Vector{@NamedTuple begin
    parent::ClassificationTreeNode
    mask::BitVector
    direction::DataType
    parent_index::Union{Int,Nothing}
  end}()

  push!(children_to_randomise, (parent=root, mask=rightmask, direction=RightChild, parent_index=1))
  push!(children_to_randomise, (parent=root, mask=leftmask, direction=LeftChild, parent_index=1))

  tree = ClassificationTree(
    root,
    X,
    y,
    NodeMap{ClassificationTreeNode}(1 => (node=root, rowmask=BitVector(ones(nrow(X))), parent=nothing)),
    max_depth,
    leaf_prediction_function
  )

  leaf_prediction = create_prediction_selector(tree)
  random_decision = create_decision_randomiser(tree)
  current_num_nodes = 1
  while length(children_to_randomise) > 0
    current_num_nodes += 1
    parent, mask, direction, parent_index = pop!(children_to_randomise)
    if (nodelevel(parent) == max_depth - 1) || rand() >= split_probability
      leaf_node = child!(direction, parent, leaf_prediction(mask))
      tree.nodemap[current_num_nodes] = (node=leaf_node, rowmask=mask, parent=parent_index)
    else
      decision = random_decision(mask, prevattr=parent.attribute, prevthreshold=parent.threshold)
      branch_node = child!(direction, parent, decision.attribute, decision.threshold)

      tree.nodemap[current_num_nodes] = (node=branch_node, rowmask=mask, parent=parent_index)
      push!(children_to_randomise, (parent=branch_node, mask=decision.rightmask, direction=RightChild, parent_index=current_num_nodes))
      push!(children_to_randomise, (parent=branch_node, mask=decision.leftmask, direction=LeftChild, parent_index=current_num_nodes))
    end
  end



  return tree
end


"""
    random_classification_tree(generation_size::Int, X::AbstractDataFrame, y::CategoricalVector; kwargs...)

# Keyword Arguments
- `max_depth::Int=5`: maximum depth of generated tree.
- `split_probability::Float64=0.5`: probability of splitting a node into another branch instead of a leaf.
- `leaf_prediction_function::Union{Nothing, Function}=nothing`: function that computes the prediction for leaf 
based on the subset of `y` and `X` that ends up in that leaf, if `nothing` uses `leaf_prediction_type` to determine the function.
- `leaf_prediction_type::Type{<:ClassificationPredictionType}=RandomPrediction`: the type of leaf prediction to use.
"""
function random_classification_trees(generation_size::Int, X::AbstractDataFrame, y::CategoricalVector; kwargs...)
  treechan = Channel{Any}(generation_size)

  for i in 1:generation_size
    t = random_classification_tree(X, y; kwargs...)

    put!(treechan, t)
  end

  all_trees = [take!(treechan) for i in 1:generation_size]

  return all_trees
end


function outcome_predictions(tree::ClassificationTree)
  ŷ = similar(tree.targets)
  ŷ.pool = tree.targets.pool

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ

end

function outcome_predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
  ŷ = similar(tree.targets, nrow(Xnew))
  ŷ.pool = tree.targets.pool

  new_nodemap = create_nodemap(tree, Xnew)

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.new_nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ

end

# function outcome_class_predictions(tree::ClassificationTree)
#   ŷ = similar(tree.targets)
#   ŷ.pool = tree.targets.pool

#   leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
#   for leaf = leaf_nodes
#     ŷ[leaf.rowmask] .= leaf.node.outcome
#   end

#   return ŷ
# end

# function outcome_class_predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
#   ŷ = similar(tree.targets, nrow(Xnew))
#   ŷ.pool = tree.targets.pool

#   new_nodemap = create_nodemap(tree, Xnew)

#   leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), new_nodemap)))
#   for leaf = leaf_nodes
#     ŷ[leaf.rowmask] .= leaf.node.outcome
#   end

#   return ŷ

# end

# function outcome_probability_predictions(tree::ClassificationTree)
#   numlevels = length(levels(tree))
#   leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
#   filter!(x -> sum(x.rowmask) > 0, leaf_nodes)
#   probs = Matrix{Float64}(undef, length(tree.targets), numlevels)

#   for i in eachindex(leaf_nodes)
#     leafinfo = leaf_nodes[i]
#     ysubset = @view tree.targets[leafinfo.rowmask]
#     prob_vector = map(level -> sum(ysubset .== level) / length(ysubset), levels(tree))
#     probs[leafinfo.rowmask, :] .= reshape(prob_vector, (1, numlevels))
#   end

#   return MMI.UnivariateFinite(levels(tree), probs, pool=tree.targets.pool)
# end


# function outcome_probability_predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
#   numlevels = length(levels(tree))
#   leaf_nodes_with_index = collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)) #map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
#   filter!(x -> sum(x[2].rowmask) > 0, leaf_nodes_with_index)
#   probs_in_original_leaf = Dict{Int,Vector{Float64}}()

#   for i in eachindex(leaf_nodes_with_index)
#     node_number = leaf_nodes_with_index[i][1]
#     leafinfo = leaf_nodes_with_index[i][2]
#     ysubset = @view tree.targets[leafinfo.rowmask]
#     prob_vector = map(level -> sum(ysubset .== level) / length(ysubset), levels(tree))
#     probs_in_original_leaf[node_number] = prob_vector
#   end


#   probs_in_new_data = Matrix(undef, nrow(Xnew), length(levels(tree)))

#   new_nodemap = create_nodemap(tree, Xnew)

#   for i in eachindex(leaf_nodes_with_index)
#     node_number = leaf_nodes_with_index[i][1]
#     new_row_mask = new_nodemap[node_number].rowmask
#     prob_vector = probs_in_original_leaf[node_number]
#     probs_in_new_data[new_row_mask, :] .= reshape(prob_vector, 1, size(probs_in_new_data)[2])
#   end

#   return MMI.UnivariateFinite(levels(tree), probs_in_new_data, pool=tree.targets.pool)
# end