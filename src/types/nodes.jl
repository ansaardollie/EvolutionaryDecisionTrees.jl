const RefValue = Base.RefValue
const LabelDict = Dict{Int,String}
const ConstraintDict = Dict{Int,@NamedTuple{lower_bound::Union{Nothing,Float64}, upper_bound::Union{Nothing,Float64}, closed_lower::Bool, closed_upper::Bool}}

abstract type ChildDirection end

struct LeftChild <: ChildDirection end
struct RightChild <: ChildDirection end

abstract type TreeNode end

mutable struct ClassificationTreeNode <: TreeNode
  attribute::Union{Nothing,Int}
  threshold::Union{Nothing,Float64}
  outcome::Union{Nothing,CategoricalValue}
  parent::RefValue{Union{Nothing,ClassificationTreeNode}}
  left::RefValue{Union{Nothing,ClassificationTreeNode}}
  right::RefValue{Union{Nothing,ClassificationTreeNode}}
  attribute_labels::Union{Nothing,LabelDict}
end

const OptionalNode = Union{Nothing,ClassificationTreeNode}

parent(node::ClassificationTreeNode) = node.parent[]
left(node::ClassificationTreeNode) = node.left[]
right(node::ClassificationTreeNode) = node.right[]

isleaf(node::ClassificationTreeNode) = isnothing(node.attribute)
isbranch(node::ClassificationTreeNode) = !isnothing(node.attribute) && !isnothing(node.threshold)
isroot(node::ClassificationTreeNode) = isnothing(parent(node))
isleftchild(node::ClassificationTreeNode) = isroot(node) ? false : left(parent(node)) == node
isrightchild(node::ClassificationTreeNode) = !isleftchild(node)
attrlabels(node::ClassificationTreeNode) = node.attribute_labels
attrlabel(node::ClassificationTreeNode) = isnothing(node.attribute_labels) ? "Feature $(attribute)" : node.attribute_labels[node.attribute]
outcomelabel(node::ClassificationTreeNode) = "$(node.outcome)"
childdir(node::ClassificationTreeNode) = isleftchild(node) ? LeftChild : isrightchild(node) ? RightChild : nothing

AT.nodevalue(node::ClassificationTreeNode) = node

function Base.show(io::IO, node::ClassificationTreeNode)
  if isleaf(node)
    outcome = outcomelabel(node)
    print(io, outcome)
  elseif isbranch(node)
    attribute = isnothing(attrlabels(node)) ? "Feature $(node.attribute)" : attrlabels(node)[node.attribute]
    print(io, "$attribute < $(node.threshold)")
  end
end

function Base.display(node::ClassificationTreeNode)
  AT.print_tree(node)
end

AT.parent(node::ClassificationTreeNode) = parent(node)
Base.parent(node::ClassificationTreeNode) = parent(node)

AT.childtype(::Type{ClassificationTreeNode}) = ClassificationTreeNode
AT.ParentLinks(::Type{ClassificationTreeNode}) = AT.StoredParents()

AT.NodeType(::Type{ClassificationTreeNode}) = AT.HasNodeType()
AT.nodetype(::Type{ClassificationTreeNode}) = ClassificationTreeNode

function AT.children(node::ClassificationTreeNode)
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

function AT.nextsibling(node::ClassificationTreeNode)
  isleftchild(node) && return parent(node).right
  return nothing
end

function AT.prevsibling(node::ClassificationTreeNode)
  isrightchild(node) && return left(node)
  return nothing
end

function branch(
  attribute::Int,
  threshold::Float64;
  parent::OptionalNode=nothing,
  left::OptionalNode=nothing,
  right::OptionalNode=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing)

  return ClassificationTreeNode(
    attribute,
    threshold,
    nothing,
    RefValue{OptionalNode}(parent),
    RefValue{OptionalNode}(left),
    RefValue{OptionalNode}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict
  )
end

function leaf(
  outcome::CategoricalValue;
  parent::OptionalNode=nothing,
  left::OptionalNode=nothing,
  right::OptionalNode=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing)

  return ClassificationTreeNode(
    nothing,
    nothing,
    outcome,
    RefValue{OptionalNode}(parent),
    RefValue{OptionalNode}(left),
    RefValue{OptionalNode}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict
  )
end


function leftchild!(parent::ClassificationTreeNode, outcome::CategoricalValue)
  isnothing(left(parent)) || error("Left child already assigned.")
  child = leaf(outcome, parent=parent)
  parent.left[] = child
end

function leftchild!(parent::ClassificationTreeNode, attribute::Int, threshold::Float64)
  isnothing(left(parent)) || error("Left child already assigned.")
  child = branch(attribute, threshold, parent=parent)
  parent.left[] = child
end

function leftchild!(parent_node::ClassificationTreeNode, child_node::ClassificationTreeNode)
  isnothing(left(parent_node)) || error("Left child already assigned.")
  isnothing(parent(child_node)) || error("Parent already assigned.")
  child_node.parent[] = parent_node
  parent_node.left[] = child_node
end

function rightchild!(parent::ClassificationTreeNode, outcome::CategoricalValue)
  isnothing(right(parent)) || error("Right child already assigned.")
  child = leaf(outcome, parent=parent)
  parent.right[] = child
end

function rightchild!(parent::ClassificationTreeNode, attribute::Int, threshold::Float64)
  isnothing(right(parent)) || error("Right child already assigned.")
  child = branch(attribute, threshold, parent=parent)
  parent.right[] = child
end

function rightchild!(parent_node::ClassificationTreeNode, child_node::ClassificationTreeNode)
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

function Base.copy(node::OptionalNode)
  isnothing(node) && return nothing
  if isroot(node)
    root = branch(node.attribute, node.threshold, attribute_labels_dict=node.attribute_labels)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(root, lc)
    rightchild!(root, rc)
    return root
  elseif isleaf(node)
    leaf_node = leaf(node.outcome, attribute_labels_dict=node.attribute_labels)
    return leaf_node
  else
    isbranch(node)
    branch_node = branch(node.attribute, node.threshold, attribute_labels_dict=node.attribute_labels)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(branch_node, lc)
    rightchild!(branch_node, rc)
    return branch_node
  end
end

function subtree(node::ClassificationTreeNode, depth::Int)
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

function nodelevel(node::ClassificationTreeNode)
  if isroot(node)
    return 0
  else
    return 1 + nodelevel(parent(node))
  end
end
