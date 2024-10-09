
mutable struct ClassificationTree <: AbstractDecisionTree{ClassificationTreeNode}
  root::ClassificationTreeNode
  features::AbstractDataFrame
  targets::CategoricalVector
  nodemap::NodeMap{ClassificationTreeNode}
  maxdepth::Int
end



CategoricalArrays.levels(tree::ClassificationTree) = levels(tree.targets)




function create_outcome_randomiser(tree::ClassificationTree)
  targets = tree.targets
  function random_outcome(mask::Union{Nothing,BitArray{1}})
    indices = isnothing(mask) ? BitVector(repeat([1], length(targets))) : mask
    available_values = @view targets[indices]
    return length(available_values) > 0 ? rand(available_values) : CategoricalValue(rand(levels(tree)), tree.targets)
  end
  return random_outcome
end

function create_attribute_randomiser(tree::ClassificationTree)
  function random_attribute()
    attrindices = 1:ncol(tree.features)
    return rand(attrindices)
  end
  return random_attribute
end

function create_decision_randomiser(tree::ClassificationTree)
  random_attribute = create_attribute_randomiser(tree)
  X = tree.features
  function random_decision(mask::Union{Nothing,BitArray{1}}; prevattr::Union{Nothing,Int}=nothing, prevthreshold::Union{Nothing,Float64}=nothing)
    attribute = random_attribute()
    indices = isnothing(mask) ? BitVector(repeat([1], nrow(X))) : mask
    available_thresholds = @view X[indices, attribute]
    if length(available_thresholds) == 0
      attribute = prevattr
      threshold = prevthreshold
    else
      threshold = rand(available_thresholds)
    end
    left_split_indices = ((@view X[!, attribute]) .< threshold)
    right_split_indices = .!left_split_indices
    leftmask = indices .& left_split_indices
    rightmask = indices .& right_split_indices
    return (
      attribute=attribute,
      threshold=threshold,
      leftmask=leftmask,
      rightmask=rightmask
    )
  end
  return random_decision
end

function random_classification_tree(X, y; maxdepth=5, probsplit=0.5)
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

  tree = ClassificationTree(root, X, y, NodeMap{ClassificationTreeNode}(1 => (node=root, rowmask=BitVector(ones(nrow(X))), parent=nothing)), maxdepth)

  random_outcome = create_outcome_randomiser(tree)
  random_decision = create_decision_randomiser(tree)
  current_num_nodes = 1
  while length(children_to_randomise) > 0
    current_num_nodes += 1
    parent, mask, direction, parent_index = pop!(children_to_randomise)
    if (nodelevel(parent) == maxdepth - 1) || rand() >= probsplit
      leaf_node = child!(direction, parent, random_outcome(mask))
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


function random_classification_trees(numtrees, X, y; maxdepth=5, probsplit=0.5)
  treechan = Channel{Any}(numtrees)

  for i in 1:numtrees
    t = random_classification_tree(X, y; maxdepth=maxdepth, probsplit=probsplit)

    put!(treechan, t)
  end

  all_trees = [take!(treechan) for i in 1:numtrees]

  return all_trees
end




function outcome_class_predictions(tree::ClassificationTree)
  ŷ = similar(tree.targets)
  ŷ.pool = tree.targets.pool

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ
end

function outcome_class_predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
  ŷ = similar(tree.targets, nrow(Xnew))
  ŷ.pool = tree.targets.pool

  new_nodemap = create_nodemap(tree, Xnew)

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), new_nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ

end

function outcome_probability_predictions(tree::ClassificationTree)
  numlevels = length(levels(tree))
  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  filter!(x -> sum(x.rowmask) > 0, leaf_nodes)
  probs = Matrix{Float64}(undef, length(tree.targets), numlevels)

  for i in eachindex(leaf_nodes)
    leafinfo = leaf_nodes[i]
    ysubset = @view tree.targets[leafinfo.rowmask]
    prob_vector = map(level -> sum(ysubset .== level) / length(ysubset), levels(tree))
    probs[leafinfo.rowmask, :] .= reshape(prob_vector, (1, numlevels))
  end

  return MMI.UnivariateFinite(levels(tree), probs, pool=tree.targets.pool)
end


function outcome_probability_predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
  numlevels = length(levels(tree))
  leaf_nodes_with_index = collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)) #map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  filter!(x -> sum(x[2].rowmask) > 0, leaf_nodes_with_index)
  probs_in_original_leaf = Dict{Int,Vector{Float64}}()

  for i in eachindex(leaf_nodes_with_index)
    node_number = leaf_nodes_with_index[i][1]
    leafinfo = leaf_nodes_with_index[i][2]
    ysubset = @view tree.targets[leafinfo.rowmask]
    prob_vector = map(level -> sum(ysubset .== level) / length(ysubset), levels(tree))
    probs_in_original_leaf[node_number] = prob_vector
  end


  probs_in_new_data = Matrix(undef, nrow(Xnew), length(levels(tree)))

  new_nodemap = create_nodemap(tree, Xnew)

  for i in eachindex(leaf_nodes_with_index)
    node_number = leaf_nodes_with_index[i][1]
    new_row_mask = new_nodemap[node_number].rowmask
    prob_vector = probs_in_original_leaf[node_number]
    probs_in_new_data[new_row_mask, :] .= reshape(prob_vector, 1, size(probs_in_new_data)[2])
  end

  return MMI.UnivariateFinite(levels(tree), probs_in_new_data, pool=tree.targets.pool)
end