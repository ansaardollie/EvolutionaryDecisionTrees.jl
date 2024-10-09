const RefValue = Base.RefValue
const LabelDict = Dict{Int,String}
const ConstraintDict = Dict{Int,@NamedTuple{lower_bound::Union{Nothing,Float64}, upper_bound::Union{Nothing,Float64}, closed_lower::Bool, closed_upper::Bool}}

abstract type ChildDirection end

struct LeftChild <: ChildDirection end
struct RightChild <: ChildDirection end

abstract type AbstractTreeNode end


const FitnessValues = @NamedTuple{raw::Float64, penalised::Float64, rescaled::Float64}

parent(node::AbstractTreeNode) = node.parent[]
left(node::AbstractTreeNode) = node.left[]
right(node::AbstractTreeNode) = node.right[]

isleaf(node::AbstractTreeNode) = isnothing(node.attribute)
isbranch(node::AbstractTreeNode) = !isnothing(node.attribute) && !isnothing(node.threshold)
isroot(node::AbstractTreeNode) = isnothing(parent(node))
isleftchild(node::AbstractTreeNode) = isroot(node) ? false : left(parent(node)) == node
isrightchild(node::AbstractTreeNode) = !isleftchild(node)
attrlabels(node::AbstractTreeNode) = node.attribute_labels
attrlabel(node::AbstractTreeNode) = isnothing(node.attribute_labels) ? "Feature $(attribute)" : node.attribute_labels[node.attribute]

childdir(node::AbstractTreeNode) = isleftchild(node) ? LeftChild : isrightchild(node) ? RightChild : nothing


AT.nodevalue(node::AbstractTreeNode) = node

function Base.show(io::IO, node::AbstractTreeNode)
  if isleaf(node)
    outcome = outcomelabel(node)
    print(io, outcome)
  elseif isbranch(node)
    attribute = isnothing(attrlabels(node)) ? "Feature $(node.attribute)" : attrlabels(node)[node.attribute]
    print(io, "$attribute < $(node.threshold)")
  end
end

function Base.display(node::AbstractTreeNode)
  AT.print_tree(node)
end

AT.parent(node::AbstractTreeNode) = parent(node)
Base.parent(node::AbstractTreeNode) = parent(node)

AT.childtype(::Type{AbstractTreeNode}) = AbstractTreeNode
AT.ParentLinks(::Type{AbstractTreeNode}) = AT.StoredParents()

AT.NodeType(::Type{AbstractTreeNode}) = AT.HasNodeType()
AT.nodetype(::Type{AbstractTreeNode}) = AbstractTreeNode

function AT.children(node::AbstractTreeNode)
  if isnothing(left(node)) && isnothing(right(node))
    return ()
  elseif isnothing(left(node)) && !isnothing(right(node))
    return (right(node),)
  elseif !isnothing(left(node)) && isnothing(right(node))
    return (left(node),)
  else
    return (left(node), right(node))
  end
end

function AT.nextsibling(node::AbstractTreeNode)
  isleftchild(node) && return parent(node).right
  return nothing
end

function AT.prevsibling(node::AbstractTreeNode)
  isrightchild(node) && return left(node)
  return nothing
end


function branch(
  ::Type{T},
  attribute::Int,
  threshold::Float64;
  parent::Union{Nothing,T}=nothing,
  left::Union{Nothing,T}=nothing,
  right::Union{Nothing,T}=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing
) where {T<:AbstractTreeNode}
  return T(
    attribute,
    threshold,
    nothing,
    RefValue{Union{Nothing,T}}(parent),
    RefValue{Union{Nothing,T}}(left),
    RefValue{Union{Nothing,T}}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict
  )
end



function leaf(
  ::Type{T},
  outcome::V;
  parent::Union{Nothing,T}=nothing,
  left::Union{Nothing,T}=nothing,
  right::Union{Nothing,T}=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing
) where {T<:AbstractTreeNode,V<:Union{CategoricalValue,Float64}}

  return T(
    nothing,
    nothing,
    outcome,
    RefValue{Union{Nothing,T}}(parent),
    RefValue{Union{Nothing,T}}(left),
    RefValue{Union{Nothing,T}}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict
  )
end

function leftchild!(parent::T, outcome::V) where {T<:AbstractTreeNode,V<:Union{CategoricalValue,Float64}}
  isnothing(left(parent)) || error("Left child already assigned.")
  child = leaf(T, outcome, parent=parent)
  parent.left[] = child
end

function leftchild!(parent::T, attribute::Int, threshold::Float64) where {T<:AbstractTreeNode}
  isnothing(left(parent)) || error("Left child already assigned.")
  child = branch(T, attribute, threshold, parent=parent)
  parent.left[] = child
end

function leftchild!(parent_node::T, child_node::T) where {T<:AbstractTreeNode}
  isnothing(left(parent_node)) || error("Left child already assigned.")
  isnothing(parent(child_node)) || error("Parent already assigned.")
  child_node.parent[] = parent_node
  parent_node.left[] = child_node
end

function rightchild!(parent::T, outcome::V) where {T<:AbstractTreeNode,V<:Union{CategoricalValue,Float64}}
  isnothing(right(parent)) || error("Right child already assigned.")
  child = leaf(T, outcome, parent=parent)
  parent.right[] = child
end

function rightchild!(parent::T, attribute::Int, threshold::Float64) where {T<:AbstractTreeNode}
  isnothing(right(parent)) || error("Right child already assigned.")
  child = branch(T, attribute, threshold, parent=parent)
  parent.right[] = child
end

function rightchild!(parent_node::T, child_node::T) where {T<:AbstractTreeNode}
  isnothing(right(parent_node)) || error("Right child already assigned.")
  isnothing(parent(child_node)) || error("Parent already assigned.")
  child_node.parent[] = parent_node
  parent_node.right[] = child_node
end

function child!(::Type{T}, parent, args...) where {T<:ChildDirection}
  if T <: LeftChild
    return leftchild!(parent, args...)
  elseif T <: RightChild
    return rightchild!(parent, args...)
  end
end


function Base.copy(node::T) where {T<:AbstractTreeNode}
  isnothing(node) && return nothing
  if isroot(node)
    root = branch(T, node.attribute, node.threshold, attribute_labels_dict=node.attribute_labels)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(root, lc)
    rightchild!(root, rc)
    return root
  elseif isleaf(node)
    leaf_node = leaf(T, node.outcome, attribute_labels_dict=node.attribute_labels)
    return leaf_node
  else
    isbranch(node)
    branch_node = branch(T, node.attribute, node.threshold, attribute_labels_dict=node.attribute_labels)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(branch_node, lc)
    rightchild!(branch_node, rc)
    return branch_node
  end
end

function subtree(node::T, depth::Int) where {T<:AbstractTreeNode}
  if depth == -1
    return nothing
  else
    node_copy = copy(node)
    node_copy.left[] = nothing
    node_copy.right[] = nothing

    left_subtree = subtree(left(node), depth - 1)
    right_subtree = subtree(right(node), depth - 1)

    node_copy.left[] = left_subtree
    node_copy.right[] = right_subtree

    return node_copy
  end

end

function nodelevel(node::T) where {T<:AbstractTreeNode}
  if isroot(node)
    return 0
  else
    return 1 + nodelevel(parent(node))
  end
end


abstract type AbstractDecisionTree{T<:AbstractTreeNode} end

function tree_type(tree::T) where {T<:AbstractDecisionTree}
  if T <: ClassificationTree
    return "Classification"
  else
    return "Regression"
  end
end

function tree_type(::Type{T}) where {T<:AbstractDecisionTree}
  if T <: ClassificationTree
    return "Classification"
  else
    return "Regression"
  end
end


function create_attribute_randomiser(tree::T) where {T<:AbstractDecisionTree}
  function random_attribute()
    attrindices = 1:ncol(tree.features)
    return rand(attrindices)
  end
  return random_attribute
end


function create_decision_randomiser(tree::T) where {T<:AbstractDecisionTree}
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


const NodeMapDetails{T<:AbstractTreeNode} = @NamedTuple begin
  node::T
  rowmask::BitVector
  parent::Union{Int,Nothing}
end

const NodeMap{T<:AbstractTreeNode} = Dict{Int,NodeMapDetails{T}}

function Base.copy(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  root_copy = copy(tree.root)
  new_nodemap = NodeMap{V}()
  current_num_nodes_processed = 0
  function process_node(node::V)
    current_num_nodes_processed += 1
    previous_nmi = tree.nodemap[current_num_nodes_processed]
    new_nodemap[current_num_nodes_processed] = (node=node, rowmask=previous_nmi.rowmask, parent=previous_nmi.parent)
    if isbranch(node)
      process_node(node.left[])
      process_node(node.right[])
    end
  end
  process_node(root_copy)
  new_tree = V <: ClassificationTreeNode ? ClassificationTree(
    root_copy,
    tree.features,
    tree.targets,
    new_nodemap,
    tree.max_depth
  ) : RegressionTree(
    root_copy,
    tree.features,
    tree.targets,
    new_nodemap,
    tree.max_depth,
    tree.leaf_predictor
  )
  reset_nodemap!(new_tree)
  return new_tree
end


function reset_nodemap!(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  new_nodemap = NodeMap{V}()

  current_num_nodes_processed = 0
  function process_node(node::V, prev_mask::BitVector, parent_idx::Union{Int,Nothing}=nothing)
    current_num_nodes_processed += 1
    if !isroot(node)
      parent_attr = parent(node).attribute
      parent_threshold = parent(node).threshold

      additional_mask = isleftchild(node) ? ((@view tree.features[!, parent_attr]) .< parent_threshold) : ((@view tree.features[!, parent_attr]) .>= parent_threshold)
    else
      additional_mask = BitVector(ones(nrow(tree.features)))
    end

    current_mask = prev_mask .& additional_mask
    new_nodemap[current_num_nodes_processed] = (node=node, rowmask=current_mask, parent=parent_idx)

    if isbranch(node)
      process_node(node.left[], current_mask, current_num_nodes_processed)
      process_node(node.right[], current_mask, current_num_nodes_processed)
    end
  end

  process_node(tree.root, BitVector(ones(length(tree.targets))), nothing)

  empty!(tree.nodemap)

  for nmi in new_nodemap
    tree.nodemap[nmi[1]] = nmi[2]
  end

  return tree
end



attributes(tree::AbstractDecisionTree) = Symbol.(names(tree.features))



function nodenumber(tree::T, node::V) where {T<:AbstractDecisionTree{V}} where {V<:AbstractTreeNode}
  for ni in tree.nodemap
    idx = ni[1]
    nodeinfo = ni[2]

    if nodeinfo.node == node
      return idx
    end
  end

  return nothing
end

struct NodeConstraints
  constraints::ConstraintDict
end

function Base.getindex(nc::NodeConstraints, index::Int)
  return nc.constraints[index]
end


function Base.iterate(nc::NodeConstraints)
  return Base.iterate(nc.constraints)
end

function Base.iterate(nc::NodeConstraints, state)
  return Base.iterate(nc.constraints, state)
end


function merge(c1::ConstraintDict, c2::ConstraintDict)
  new_constraints = ConstraintDict()
  all_attributes = keys(c1)

  for attrindex in all_attributes
    old_lbs = [c1[attrindex].lower_bound, c2[attrindex].lower_bound]
    filter!(x -> !isnothing(x), old_lbs)
    new_lb = length(old_lbs) == 0 ? nothing : maximum(old_lbs)
    new_lb_closed = isnothing(new_lb) ?
                    false :
                    new_lb == c1[attrindex].lower_bound ?
                    c1[attrindex].closed_lower :
                    c2[attrindex].closed_lower
    old_ubs = [c1[attrindex].upper_bound, c2[attrindex].upper_bound]
    filter!(x -> !isnothing(x), old_ubs)
    new_ub = length(old_ubs) == 0 ? nothing : minimum(old_ubs)
    new_ub_closed = isnothing(new_ub) ?
                    false :
                    new_ub == c1[attrindex].upper_bound ?
                    c1[attrindex].closed_upper :
                    c2[attrindex].closed_upper
    new_constraints[attrindex] = (lower_bound=new_lb, upper_bound=new_ub, closed_lower=new_lb_closed, closed_upper=new_ub_closed)
  end

  return new_constraints
end


function create_empty_constraints_generator(tree::T) where {T<:AbstractDecisionTree}
  attrs = 1:ncol(tree.features)
  function empty_constraints()
    constraints = ConstraintDict()
    for attr in attrs
      constraints[attr] = (lower_bound=nothing, upper_bound=nothing, closed_lower=false, closed_upper=false)
    end
    return constraints
  end
  return empty_constraints
end

function create_parent_constraints_generator(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  empty_constraints_generator = create_empty_constraints_generator(tree)
  function get_parent_constraints(node::V)
    if isroot(node)
      return empty_constraints_generator()
    else
      node_is_left_child = isleftchild(node)
      parent_attribute = parent(node).attribute
      parent_threshold = parent(node).threshold
      parent_constraints = empty_constraints_generator()
      if node_is_left_child
        parent_constraints[parent_attribute] = (lower_bound=nothing, upper_bound=parent_threshold, closed_lower=false, closed_upper=false)
      else
        parent_constraints[parent_attribute] = (lower_bound=parent_threshold, upper_bound=nothing, closed_lower=true, closed_upper=false)
      end
      return merge(parent_constraints, get_parent_constraints(parent(node)))
    end
  end
  return get_parent_constraints
end

function create_left_constraints_generator(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function left_constraints(node::V)
    if isleaf(node)
      return merge(empty_constraints_generator(), parent_constraints_generator(node))
    else
      new_constraints = empty_constraints_generator()
      new_constraints[node.attribute] = (lower_bound=nothing, upper_bound=node.threshold, closed_lower=false, closed_upper=false)
      return merge(new_constraints, parent_constraints_generator(node))
    end
  end
  return left_constraints
end


function create_right_constraints_generator(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function right_constraints(node::V)
    if isleaf(node)
      return merge(empty_constraints_generator(), parent_constraints_generator(node))
    else
      new_constraints = empty_constraints_generator()
      new_constraints[node.attribute] = (lower_bound=node.threshold, upper_bound=nothing, closed_lower=true, closed_upper=false)
      return merge(new_constraints, parent_constraints_generator(node))
    end
  end
  return right_constraints
end



function create_constraints_generator(tree::AbstractDecisionTree{V}) where {V<:AbstractTreeNode}
  parent_constraints_generator = create_parent_constraints_generator(tree)
  left_constraints_generator = create_left_constraints_generator(tree)
  right_constraints_generator = create_right_constraints_generator(tree)
  function constraints(node::V, constraint_type::Union{Nothing,Type{<:ChildDirection}}=nothing)
    if isnothing(constraint_type)
      return NodeConstraints(parent_constraints_generator(node))
    elseif constraint_type <: LeftChild
      return NodeConstraints(left_constraints_generator(node))
    else
      constraint_type <: RightChild
      return NodeConstraints(right_constraints_generator(node))
    end
  end
end


function Base.print(io::IO, tree::T) where {T<:AbstractDecisionTree{V}} where {V<:AbstractTreeNode}
  nleaves = length(collect(AT.Leaves(tree.root)))
  nbranches = length(collect(AT.PreOrderDFS(tree.root))) - nleaves
  treetype = tree_type(tree)
  print(io, "$treetype tree with $nleaves leaf nodes and $nbranches branch nodes")
end

function Base.print(tree::T) where {T<:AbstractDecisionTree{V}} where {V<:AbstractTreeNode}
  io = stdout
  print(io, tree)
end


function Base.show(io::IO, tree::AbstractDecisionTree{V}; without_constraints::Bool=false, kw...) where {V<:AbstractTreeNode}
  if get(io, :in_forest, false)
    print(io, tree)
  elseif without_constraints
    AT.print_tree(io, tree.root; kw...)
  else
    constraint_generator = create_constraints_generator(tree)
    function node_constraint_printer(io::IO, node::V; kw2...)
      decimals = get(kw, :decimals, 3)
      nn = nodenumber(tree, node)
      print(io, "#$nn: ")
      print(io, (isbranch(node) ? "$(attrlabel(node)) < $(round(node.threshold, digits=decimals))" : node))
      if isbranch(node) && !(isroot(node))
        nodetype = nothing
        print(io, " | ")
      elseif isleaf(node)
        nodetype = childdir(node)
        print(io, " ⟺ ")
      else
        nodetype = nothing

      end

      if !isroot(node)
        constraints = constraint_generator(node, nodetype).constraints
        all_consts = Vector{String}()
        for constraint in pairs(constraints)
          feature = names(tree.features)[constraint[1]]
          lower_bound = constraint[2].lower_bound
          upper_bound = constraint[2].upper_bound
          left_set_bracket = constraint[2].closed_lower ? "[" : "("
          right_set_bracket = constraint[2].closed_upper ? "]" : ")"
          lb = isnothing(lower_bound) ? "-∞" : "$(round(lower_bound, digits=decimals))"
          rb = isnothing(upper_bound) ? "∞" : "$(round(upper_bound, digits=decimals))"
          push!(all_consts, feature * " ∈ " * left_set_bracket * lb * " , " * rb * right_set_bracket)
        end
        print(io, join(all_consts, " ∧ "))
      end
    end
    AT.print_tree(node_constraint_printer, io, tree.root)
  end
end



function Base.display(io::IO, trees::Vector{<:AbstractDecisionTree{V}}) where {V<:AbstractTreeNode}
  tt = tree_type(eltype(trees))
  println(io, "$(length(trees))-element $(tt) Tree Vector")
  if length(trees) > 10
    for t in trees[1:5]
      ctx = IOContext(io, :in_forest => true)
      print(ctx, " ")
      show(ctx, t)
      println(ctx)
    end
    println(" ⋮")
    for t in trees[end-4:end]
      ctx = IOContext(io, :in_forest => true)
      print(ctx, " ")
      show(ctx, t)
      println(ctx)
    end
  elseif length(trees) > 10
    for t in trees[1:end]
      ctx = IOContext(io, :in_forest => true)
      print(ctx, " ")
      show(ctx, t)
      println(ctx)
    end
  else

  end
end

function Base.display(trees::Vector{<:AbstractDecisionTree{V}}) where {V<:AbstractTreeNode}
  io = stdout
  display(io, trees)
end

function Base.show(io::IO, trees::Vector{<:AbstractDecisionTree{V}}) where {V<:AbstractTreeNode}
  display(io, trees)
end

function Base.show(trees::Vector{<:AbstractDecisionTree{V}}) where {V<:AbstractTreeNode}
  io = stdout
  display(io, trees)
end

function Base.print(io::IO, trees::Vector{T}) where {T<:AbstractDecisionTree{V}} where {V<:AbstractTreeNode}
  display(io, trees)
end

function Base.print(trees::Vector{T}) where {T<:AbstractDecisionTree{V}} where {V<:AbstractTreeNode}
  io = stdout
  print(io, trees)
end



function create_nodemap(tree::AbstractDecisionTree{V}, Xnew::AbstractDataFrame) where {V<:AbstractTreeNode}

  new_nodemap = NodeMap{V}()

  current_node_number = 0
  function process_node(node::V, prev_mask::BitVector, parent_idx::Union{Nothing,Int})
    current_node_number += 1
    if isroot(node)
      additional_mask = BitVector(ones(nrow(Xnew)))
    else
      parent_threshold = parent(node).threshold
      parent_attribute = parent(node).attribute
      additional_mask = isleftchild(node) ? ((@view Xnew[!, parent_attribute]) .< parent_threshold) : ((@view Xnew[!, parent_attribute]) .>= parent_threshold)
    end
    new_mask = prev_mask .& additional_mask
    new_nodemap[current_node_number] = (node=node, rowmask=new_mask, parent=parent_idx)

    if isbranch(node)
      process_node(node.left[], new_mask, current_node_number)
      process_node(node.right[], new_mask, current_node_number)
    end
  end

  process_node(tree.root, BitVector(ones(nrow(Xnew))), nothing)

  return new_nodemap
end
