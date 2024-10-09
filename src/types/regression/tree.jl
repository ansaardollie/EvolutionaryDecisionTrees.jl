

mutable struct RegressionTree <: AbstractDecisionTree{RegressionTreeNode}
  root::RegressionTreeNode
  features::AbstractDataFrame
  targets::Vector{Float64}
  nodemap::NodeMap{RegressionTreeNode}
  maxdepth::Int
  leafpredictor::Function
end

abstract type LeafPredictionType end

struct MeanPrediction <: LeafPredictionType end
struct MedianPrediction <: LeafPredictionType end
struct MidpointPrediction <: LeafPredictionType end
struct RandomPrediction <: LeafPredictionType end

function leaf_predictor(T::Type{<:LeafPredictionType})
  if T <: MeanPrediction
    return (target_subset, feature_subset) -> mean(target_subset)
  elseif T <: MedianPrediction
    return (target_subset, feature_subset) -> median(target_subset)
  elseif T <: MidpointPrediction
    return (target_subset, feature_subset) -> 0.5 * (minimum(target_subset) + maximum(target_subset))
  elseif T <: RandomPrediction
    return (target_subset, feature_subset) -> rand(target_subset)
  else
    error("LeafPredictionType $(T) not recognised")
  end
end


function create_prediction_selector(tree::RegressionTree)
  y = tree.targets
  X = tree.features

  function prediction(mask::Union{Nothing,BitVector})
    mask = isnothing(mask) ? BitVector(ones(length(y))) : mask
    y_subset = @view y[mask]
    X_subset = @view X[mask, :]
    return sum(mask) > 0 ? tree.leafpredictor(y_subset, X_subset) : mean(y)
  end

  return prediction
end

function random_regression_tree(X, y; leafpredfunc::Union{Nothing,Function}=nothing, leafpredtype::Type{T}=MeanPrediction, maxdepth=5, probsplit=0.5) where {T<:LeafPredictionType}
  attribute_labels = Dict(pairs(names(X)))

  lpf = isnothing(leafpredfunc) ? leaf_predictor(leafpredtype) : leafpredfunc

  root_attr = rand(1:ncol(X))
  root_threshold = rand(@view X[!, root_attr])

  root = branch(RegressionTreeNode, root_attr, root_threshold, attribute_labels_dict=attribute_labels)
  leftmask = (@view X[!, root_attr]) .< root_threshold
  rightmask = (@view X[!, root_attr]) .>= root_threshold

  children_to_randomise = Vector{@NamedTuple begin
    parent::RegressionTreeNode
    mask::BitVector
    direction::DataType
    parent_index::Union{Int,Nothing}
  end}()

  push!(children_to_randomise, (parent=root, mask=rightmask, direction=RightChild, parent_index=1))
  push!(children_to_randomise, (parent=root, mask=leftmask, direction=LeftChild, parent_index=1))

  tree = RegressionTree(
    root,
    X,
    y,
    NodeMap{RegressionTreeNode}(1 => (node=root, rowmask=BitVector(ones(nrow(X))), parent=nothing)),
    maxdepth,
    lpf
  )

  leaf_prediction = create_prediction_selector(tree)
  random_decision = create_decision_randomiser(tree)
  current_num_nodes = 1
  while length(children_to_randomise) > 0
    current_num_nodes += 1
    parent, mask, direction, parent_index = pop!(children_to_randomise)
    if ((nodelevel(parent) == maxdepth - 1) || rand() >= probsplit)
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


function random_regression_trees(numtrees, X, y; leafpredfunc::Union{Nothing,Function}=nothing, leafpredtype::Type{T}=MeanPrediction, maxdepth=5, probsplit=0.5) where {T<:LeafPredictionType}
  treechan = Channel{Any}(numtrees)

  for i in 1:numtrees
    t = random_regression_tree(X, y; leafpredfunc=leafpredfunc, leafpredtype=leafpredtype, maxdepth=maxdepth, probsplit=probsplit)

    put!(treechan, t)
  end

  all_trees = [take!(treechan) for i in 1:numtrees]

  return all_trees
end



function outcome_predictions(tree::RegressionTree)
  ŷ = similar(tree.targets)

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ
end


function reset_nodemap!(tree::RegressionTree)

  new_nodemap = NodeMap{RegressionTreeNode}()

  prediction_selector = create_prediction_selector(tree)

  current_num_nodes_processed = 0

  function process_node(node::RegressionTreeNode, prev_mask::BitVector, parent_idx::Union{Nothing,Int}=nothing)
    current_num_nodes_processed += 1

    if isroot(node)
      newmask = prev_mask
    else
      newmask = prev_mask .& (
        isleftchild(node) ?
        ((@view tree.features[:, node.parent[].attribute]) .< node.parent[].threshold) :
        ((@view tree.features[:, node.parent[].attribute]) .>= node.parent[].threshold)
      )
    end

    if isleaf(node)
      new_outcome = prediction_selector(newmask)
      new_leaf = leaf(RegressionTreeNode, new_outcome, attribute_labels_dict=node.attribute_labels)
      if isleftchild(node)
        node.parent[].left[] = nothing
        leftchild!(node.parent[], new_leaf)
      else
        node.parent[].right[] = nothing
        rightchild!(node.parent[], new_leaf)
      end
      new_nodemap[current_num_nodes_processed] = (node=new_leaf, rowmask=newmask, parent=parent_idx)
    else
      new_nodemap[current_num_nodes_processed] = (node=node, rowmask=newmask, parent=parent_idx)
      process_node(node.left[], newmask, current_num_nodes_processed)
      process_node(node.right[], newmask, current_num_nodes_processed)
    end

  end

  process_node(tree.root, BitVector(ones(length(tree.targets))), nothing)

  empty!(tree.nodemap)

  for nmi in new_nodemap
    tree.nodemap[nmi[1]] = nmi[2]
  end

  return tree


end

#=
if !isroot(node)
      parent_attr = parent(node).attribute
      parent_threshold = parent(node).threshold

      additional_mask = isleftchild(node) ? ((@view tree.features[!, parent_attr]) .< parent_threshold) : ((@view tree.features[!, parent_attr]) .>= parent_threshold)
    else
      additional_mask = BitVector(ones(nrow(tree.features)))
    end

    current_mask = prev_mask .& additional_mask

    if isleaf(node)
      new_outcome = prediction_selector(current_mask)
      if new_outcome != node.outcome
        parent_node = node.parent[]

        new_node = leaf(RegressionTreeNode, new_outcome; attribute_labels_dict=node.attribute_labels)
        if isleftchild(node)
          parent_node.left[] = nothing
          leftchild!(parent_node, new_node)
        else
          parent_node.right[] = nothing
          rightchild!(parent_node, new_node)
        end
      end
    end

    new_nodemap[current_num_nodes_processed] = (node=node, rowmask=current_mask, parent=parent_idx)

    if isbranch(node)
      process_node(node.left[], current_mask, current_num_nodes_processed)
      process_node(node.right[], current_mask, current_num_nodes_processed)
    end
=#