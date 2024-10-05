module EvolutionaryDecisionTrees

using
  Distributions,
  ProgressMeter,
  StatisticalMeasures,
  Random,
  DataFrames,
  GLMakie,
  Graphs,
  GraphMakie,
  NetworkLayout,
  Base.Threads,
  ProgressMeter,
  CategoricalArrays,
  CategoricalDistributions

import GraphMakie.graphplot
import Makie.plot!
import AbstractTrees as AT
import Base.convert

export random_trees, outcome_class_predictions
#=====================================================================
Types
=====================================================================#
const RefValue = Base.RefValue
const LabelDict = Dict{Int,String}
const ConstraintDict = Dict{Int,@NamedTuple{lower_bound::Union{Nothing,Float64}, upper_bound::Union{Nothing,Float64}, closed_lower::Bool, closed_upper::Bool}}



abstract type TreeNode end

abstract type ChildDirection end

struct LeftChild <: ChildDirection end
struct RightChild <: ChildDirection end

mutable struct ClassifierTreeNode <: TreeNode
  attribute::Union{Nothing,Int}
  threshold::Union{Nothing,Float64}
  outcome::Union{Nothing,Int}
  parent::RefValue{Union{Nothing,ClassifierTreeNode}}
  left::RefValue{Union{Nothing,ClassifierTreeNode}}
  right::RefValue{Union{Nothing,ClassifierTreeNode}}
  attribute_labels::Union{Nothing,LabelDict}
  outcome_labels::Union{Nothing,LabelDict}
end

const OptionalNode = Union{Nothing,ClassifierTreeNode}
parent(node::ClassifierTreeNode) = node.parent[]
left(node::ClassifierTreeNode) = node.left[]
right(node::ClassifierTreeNode) = node.right[]

isleaf(node::ClassifierTreeNode) = isnothing(node.attribute)
isbranch(node::ClassifierTreeNode) = !isnothing(node.attribute) && !isnothing(node.threshold)
isroot(node::ClassifierTreeNode) = isnothing(parent(node))
isleftchild(node::ClassifierTreeNode) = isroot(node) ? false : left(parent(node)) == node
isrightchild(node::ClassifierTreeNode) = !isleftchild(node)
attrlabels(node::ClassifierTreeNode) = node.attribute_labels
attrlabel(node::ClassifierTreeNode) = isnothing(node.attribute_labels) ? "Feature $(attribute)" : node.attribute_labels[node.attribute]
outcomelabels(node::ClassifierTreeNode) = node.outcome_labels
outcomelabel(node::ClassifierTreeNode) = isnothing(node.outcome_labels) ? "Class $(node.outcome)" : node.outcome_labels[node.outcome]
childdir(node::ClassifierTreeNode) = isleftchild(node) ? LeftChild : isrightchild(node) ? RightChild : nothing

AT.nodevalue(node::ClassifierTreeNode) = node

function Base.show(io::IO, node::ClassifierTreeNode)
  if isleaf(node)
    outcome = isnothing(outcomelabels(node)) ? "Class $(node.outcome)" : outcomelabels(node)[node.outcome]
    print(io, outcome)
  elseif isbranch(node)
    attribute = isnothing(attrlabels(node)) ? "Feature $(node.attribute)" : attrlabels(node)[node.attribute]
    print(io, "$attribute < $(node.threshold)")
  end
end

function Base.display(node::ClassifierTreeNode)
  AT.print_tree(node)
end

AT.parent(node::ClassifierTreeNode) = parent(node)
Base.parent(node::ClassifierTreeNode) = parent(node)

AT.childtype(::Type{ClassifierTreeNode}) = ClassifierTreeNode
AT.ParentLinks(::Type{ClassifierTreeNode}) = AT.StoredParents()

AT.NodeType(::Type{ClassifierTreeNode}) = AT.HasNodeType()
AT.nodetype(::Type{ClassifierTreeNode}) = ClassifierTreeNode

function AT.children(node::ClassifierTreeNode)
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

function AT.nextsibling(node::ClassifierTreeNode)
  isleftchild(node) && return parent(node).right
  return nothing
end

function AT.prevsibling(node::ClassifierTreeNode)
  isrightchild(node) && return left(node)
  return nothing
end

function branch(
  attribute::Int,
  threshold::Float64;
  parent::OptionalNode=nothing,
  left::OptionalNode=nothing,
  right::OptionalNode=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing,
  outcome_labels_dict::Union{Nothing,LabelDict}=nothing)

  return ClassifierTreeNode(
    attribute,
    threshold,
    nothing,
    RefValue{OptionalNode}(parent),
    RefValue{OptionalNode}(left),
    RefValue{OptionalNode}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict,
    isnothing(outcome_labels_dict) ? (isnothing(parent) ? nothing : outcomelabels(parent)) : outcome_labels_dict
  )
end

function leaf(
  outcome::Int;
  parent::OptionalNode=nothing,
  left::OptionalNode=nothing,
  right::OptionalNode=nothing,
  attribute_labels_dict::Union{Nothing,LabelDict}=nothing,
  outcome_labels_dict::Union{Nothing,LabelDict}=nothing)

  return ClassifierTreeNode(
    nothing,
    nothing,
    outcome,
    RefValue{OptionalNode}(parent),
    RefValue{OptionalNode}(left),
    RefValue{OptionalNode}(right),
    isnothing(attribute_labels_dict) ? (isnothing(parent) ? nothing : attrlabels(parent)) : attribute_labels_dict,
    isnothing(outcome_labels_dict) ? (isnothing(parent) ? nothing : outcomelabels(parent)) : outcome_labels_dict
  )
end


function leftchild!(parent::ClassifierTreeNode, outcome::Int)
  isnothing(left(parent)) || error("Left child already assigned.")
  child = leaf(outcome, parent=parent)
  parent.left[] = child
end

function leftchild!(parent::ClassifierTreeNode, attribute::Int, threshold::Float64)
  isnothing(left(parent)) || error("Left child already assigned.")
  child = branch(attribute, threshold, parent=parent)
  parent.left[] = child
end

function leftchild!(parent::ClassifierTreeNode, child::ClassifierTreeNode)
  isnothing(left(parent)) || error("Left child already assigned.")
  isnothing(parent(child)) || error("Parent already assigned.")
  child.parent[] = parent
  parent.left[] = child
end


function rightchild!(parent::ClassifierTreeNode, outcome::Int)
  isnothing(right(parent)) || error("Right child already assigned.")
  child = leaf(outcome, parent=parent)
  parent.right[] = child
end

function rightchild!(parent::ClassifierTreeNode, attribute::Int, threshold::Float64)
  isnothing(right(parent)) || error("Right child already assigned.")
  child = branch(attribute, threshold, parent=parent)
  parent.right[] = child
end

function rightchild!(parent::ClassifierTreeNode, child::ClassifierTreeNode)
  isnothing(right(parent)) || error("Right child already assigned.")
  isnothing(parent(child)) || error("Parent already assigned.")
  child.parent[] = parent
  parent.right[] = child
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
    root = branch(node.attribute, node.threshold)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(root, lc)
    rightchild!(root, rc)
  elseif isleaf(node)
    leaf_node = leaf(node.outcome)
    return leaf_node
  else
    isbranch(node)
    branch_node = branch(node.attribute, node.threshold)
    lc = copy(left(node))
    rc = copy(right(node))
    leftchild!(branch_node, lc)
    rightchild!(branch_node, rc)
    return branch_node
  end
  return nothing
end

function subtree(node::ClassifierTreeNode, depth::Int)
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


function nodelevel(node::ClassifierTreeNode)
  if isroot(node)
    return 0
  else
    return 1 + nodelevel(parent(node))
  end
end







# function Base.show(io::IO, nc::NodeConstraints)
#   constraints = nc.constraints
#   all_consts = Vector{String}()
#   for constraint in pairs(constraints)
#     feature = String(constraint[1])
#     lower_bound = constraint[2].lower_bound
#     upper_bound = constraint[2].upper_bound
#     left_set_bracket = constraint[2].closed_lower ? "[" : "("
#     right_set_bracket = constraint[2].closed_upper ? "]" : ")"
#     lb = isnothing(lower_bound) ? "-∞" : "$lower_bound"
#     rb = isnothing(upper_bound) ? "∞" : "$upper_bound"
#     push!(all_consts, feature * " ∈ " * left_set_bracket * lb * " , " * rb * right_set_bracket)
#   end
#   print(io, join(all_consts, " ∧ "))
# end

const NodeMap = Dict{Int,@NamedTuple{node::ClassifierTreeNode, rowmask::BitVector}}

struct TreeClassifier
  root::ClassifierTreeNode
  features::DataFrame
  targets::Vector{Int}
  pool::CategoricalPool
  nodemap::NodeMap
end

attributes(tree::TreeClassifier) = Symbol.(names(tree.features))


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


function create_empty_constraints_generator(tree::TreeClassifier)
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

function create_parent_constraints_generator(tree::TreeClassifier)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  function get_parent_constraints(node::ClassifierTreeNode)
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

function create_left_constraints_generator(tree::TreeClassifier)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function left_constraints(node::ClassifierTreeNode)
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


function create_right_constraints_generator(tree::TreeClassifier)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function right_constraints(node::ClassifierTreeNode)
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



function create_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  left_constraints_generator = create_left_constraints_generator(tree)
  right_constraints_generator = create_right_constraints_generator(tree)
  function constraints(node::ClassifierTreeNode, constraint_type::Union{Nothing,Type{<:ChildDirection}})
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


function Base.show(io::IO, tree::TreeClassifier; without_constraints::Bool=false, kw...)
  if get(io, :in_forest, false)
    nleaves = length(collect(AT.Leaves(tree.root)))
    nbranches = length(collect(AT.PreOrderDFS(tree.root))) - nleaves
    print(io, "Tree Classifier with $nleaves leaf nodes and $nbranches branch nodes")
  elseif without_constraints
    AT.print_tree(io, tree.root; kw...)
  else
    constraint_generator = create_constraints_generator(tree)
    function node_constraint_printer(io::IO, node::ClassifierTreeNode; kw2...)
      decimals = get(kw, :decimals, 3)
      print(io, isbranch(node) ? "$(attrlabel(node)) < $(round(node.threshold, digits=decimals))" : node)
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

function create_outcome_randomiser(tree::TreeClassifier)
  targets = tree.targets
  unique_targets = unique(targets)
  function random_outcome(mask::Union{Nothing,BitArray{1}})
    indices = isnothing(mask) ? BitVector(repeat([1], length(targets))) : mask
    available_values = @view targets[indices]
    return length(available_values) > 0 ? rand(available_values) : rand(unique_targets)
  end
  return random_outcome
end

function create_attribute_randomiser(tree::TreeClassifier)
  function random_attribute()
    attrindices = 1:ncol(tree.features)
    return rand(attrindices)
  end
  return random_attribute
end

function create_decision_randomiser(tree::TreeClassifier)
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

# function create_leaf_mask_generator(tree::TreeClassifier)
#   X = tree.features

#   function leaf_mask(node::ClassifierTreeNode)
#     if isroot(node)
#       return BitVector()
#   end

# end

function random_tree(X, y; outcome_labels::LabelDict, pool::CategoricalPool, maxdepth=5, probsplit=0.5)
  attribute_labels = Dict(pairs(names(X)))

  root_attr = rand(1:ncol(X))
  root_threshold = rand(@view X[!, root_attr])

  root = branch(root_attr, root_threshold, outcome_labels_dict=outcome_labels, attribute_labels_dict=attribute_labels)
  leftmask = (@view X[!, root_attr]) .< root_threshold
  rightmask = (@view X[!, root_attr]) .>= root_threshold

  children_to_randomise = Vector{@NamedTuple begin
    parent::ClassifierTreeNode
    mask::BitVector
    direction::DataType
  end}()

  push!(children_to_randomise, (parent=root, mask=rightmask, direction=RightChild))
  push!(children_to_randomise, (parent=root, mask=leftmask, direction=LeftChild))

  tree = TreeClassifier(root, X, y, pool, NodeMap(1 => (node=root, rowmask=BitVector(ones(nrow(X))))))

  random_outcome = create_outcome_randomiser(tree)
  random_decision = create_decision_randomiser(tree)
  current_num_nodes = 1
  while length(children_to_randomise) > 0
    current_num_nodes += 1
    parent, mask, direction = pop!(children_to_randomise)
    if (nodelevel(parent) == maxdepth - 1) || rand() >= probsplit
      leaf_node = child!(direction, parent, random_outcome(mask))
      tree.nodemap[current_num_nodes] = (node=leaf_node, rowmask=mask)
    else
      decision = random_decision(mask, prevattr=parent.attribute, prevthreshold=parent.threshold)
      branch_node = child!(direction, parent, decision.attribute, decision.threshold)

      tree.nodemap[current_num_nodes] = (node=branch_node, rowmask=mask)
      push!(children_to_randomise, (parent=branch_node, mask=decision.leftmask, direction=LeftChild))
      push!(children_to_randomise, (parent=branch_node, mask=decision.rightmask, direction=RightChild))
    end
  end



  return tree
end


function random_trees(numtrees, X, y; outcome_labels::LabelDict, pool::CategoricalPool, maxdepth=5, probsplit=0.5)
  treechan = Channel{Any}(numtrees)

  @showprogress desc = "Generating Trees..." barglyphs = BarGlyphs("[=> ]") for i in 1:numtrees
    t = random_tree(X, y; outcome_labels=outcome_labels, pool=pool, maxdepth=maxdepth, probsplit=probsplit)

    put!(treechan, t)
  end

  all_trees = [take!(treechan) for i in 1:numtrees]

  return all_trees
end


function Base.display(trees::Vector{TreeClassifier})
  io = stdout
  println(io, "$(length(trees))-element Classifier Tree Forest")
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
end



function outcome_class_predictions(tree::TreeClassifier)
  ŷ = Vector{Int}(undef, length(tree.targets))

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
  for leaf = leaf_nodes
    ŷ[leaf.rowmask] .= leaf.node.outcome
  end

  return ŷ
end


# function outcome_probability_predictions(tree::TreeClassifier)
#   leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))

#   map(leaf_nodes) do leaf_info
#     ysubset = @view tree.targets[leaf_info.rowmask]
#     all_levels = collect(eachindex(tree.pool.levels))

#     probs = map(x -> sum(ysubset .== x) / length(ysubset), all_levels)
#     return UnivariateFinite(levels())
#   end
# end




function Base.convert(::Type{SimpleDiGraph}, tree::TreeClassifier; maxdepth=AT.treeheight(tree))
  if maxdepth == -1
    maxdepth = AT.treeheight(tree)
  end

  g = SimpleDiGraph()
  properties = Any[]
  edge_props = Any[]
  walk_tree!(tree.root, g, maxdepth, properties)
  return g, properties, edge_props

end

function walk_tree!(node::ClassifierTreeNode, g, depthleft, properties)


  if isleaf(node)
    add_vertex!(g)
    push!(properties, (node, outcomelabel(node)))
    return vertices(g)[end]
  else
    add_vertex!(g)

    if depthleft == 0
      push!(properties, (Nothing, "..."))
      return vertices(g)[end]
    else
      depthleft -= 1
    end


    current_vertex = vertices(g)[end]
    threshold = node.threshold
    attribute = attrlabel(node)

    push!(properties, (node, "$(attribute) < $(round(threshold, digits=3))"))
    child = walk_tree!(node.left[], g, depthleft, properties)

    add_edge!(g, current_vertex, child)

    child = walk_tree!(node.right[], g, depthleft, properties)

    add_edge!(g, current_vertex, child)

    return current_vertex
  end
end

@recipe(PlotClassifierTree) do scene
  Attributes(
    nodecolormap=:rainbow,
    textcolor=RGBf(0, 0, 0),
    leafcolor=:darkgreen,
    nodecolor=:white,
    maxdepth=-1
  )
end


function treeplot(tree::TreeClassifier; kwargs...)
  f, ax, h = plotclassifiertree(tree; kwargs...)
  hidedecorations!(ax)
  hidespines!(ax)
  return f
end

function Makie.plot!(plt::PlotClassifierTree{<:Tuple{TreeClassifier}})

  @extract plt leafcolor, textcolor, nodecolormap, nodecolor, maxdepth

  @show leafcolor, textcolor, nodecolormap, nodecolor, maxdepth

  tree = plt[1]

  tmpObs = @lift convert(SimpleDiGraph, $tree; maxdepth=$maxdepth)
  graph = @lift $tmpObs[1]
  properties = @lift $tmpObs[2]

  leaf_labels = @lift [string(p[2]) for p in $properties]

  nlabels_color = map(properties, leaf_labels, leafcolor, textcolor, nodecolormap) do properties, leaf_labels, leafcolor, textcolor, nodecolormap
    leaf_ix = findall([isleaf(p[1]) for p in properties])
    leaf_label_texts = [p[1] for p in split.(leaf_labels[leaf_ix], " : ")]

    unique_labels = unique(leaf_label_texts)
    inidividual_leaf_colors = resample_cmap(nodecolormap, length(unique_labels))
    nlabels_color = Any[isbranch(p[1]) ? textcolor : leafcolor for p in properties]
    for (ix, uLV) = enumerate(unique_labels)
      ixV = leaf_label_texts .== uLV
      nlabels_color[leaf_ix[ixV]] .= inidividual_leaf_colors[ix]
    end
    return nlabels_color
  end




  graphplot!(
    plt,
    graph;
    layout=Buchheim(),
    nlabels=leaf_labels,
    node_size=100,
    node_color=nodecolor,
    nlabels_color=nlabels_color,
    nlabels_align=(:center, :center),
    nlabels_attr=(
      font="Open Sans Bold",
    )
  )

  return plt
end
































for n in names(@__MODULE__; all=true)
  if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval)
    @eval export $n
  end
end

# export
#   LabelDict,
#   ConstraintDict,
#   TreeNode,
#   ChildDirection,
#   LeftChild,
#   RightChild,
#   ClassifierTreeNode,
#   OptionalNode,
#   parent,
#   left,
#   right,
#   isleaf, 
#   isbranch,
#   isroot,








































end
