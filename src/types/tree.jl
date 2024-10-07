const NodeMap = Dict{Int,@NamedTuple{node::ClassificationTreeNode, rowmask::BitVector, parent::Union{Int,Nothing}}}

struct ClassificationTree
  root::ClassificationTreeNode
  features::DataFrame
  targets::CategoricalVector
  nodemap::NodeMap
  maxdepth::Int
end

function Base.copy(tree::ClassificationTree)
  root_copy = copy(tree.root)
  new_nodemap = NodeMap()
  current_num_nodes_processed = 0
  function process_node(node::ClassificationTreeNode)
    current_num_nodes_processed += 1
    previous_nmi = tree.nodemap[current_num_nodes_processed]
    new_nodemap[current_num_nodes_processed] = (node=node, rowmask=previous_nmi.rowmask, parent=previous_nmi.parent)
    if isbranch(node)
      process_node(node.left[])
      process_node(node.right[])
    end
  end
  process_node(root_copy)
  return ClassificationTree(
    root_copy,
    tree.features,
    tree.targets,
    new_nodemap,
    tree.maxdepth
  )
end

function reset_nodemap!(tree::ClassificationTree)
  new_nodemap = NodeMap()

  current_num_nodes_processed = 0
  function process_node(node::ClassificationTreeNode; prev_mask::BitVector, parent_idx::Union{Int,Nothing}=nothing)
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
      process_node(node.left[]; prev_mask=current_mask, parent_idx=current_num_nodes_processed)
      process_node(node.right[]; prev_mask=current_mask, parent_idx=current_num_nodes_processed)
    end
  end

  process_node(tree.root; prev_mask=BitVector(ones(length(tree.targets))), parent_idx=nothing)

  empty!(tree.nodemap)

  for nmi in new_nodemap
    tree.nodemap[nmi[1]] = nmi[2]
  end

  return tree
end

attributes(tree::ClassificationTree) = Symbol.(names(tree.features))
CategoricalArrays.levels(tree::ClassificationTree) = levels(tree.targets)


function nodenumber(tree::ClassificationTree, node::ClassificationTreeNode)
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


function create_empty_constraints_generator(tree::ClassificationTree)
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

function create_parent_constraints_generator(tree::ClassificationTree)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  function get_parent_constraints(node::ClassificationTreeNode)
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

function create_left_constraints_generator(tree::ClassificationTree)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function left_constraints(node::ClassificationTreeNode)
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


function create_right_constraints_generator(tree::ClassificationTree)
  empty_constraints_generator = create_empty_constraints_generator(tree)
  parent_constraints_generator = create_parent_constraints_generator(tree)
  function right_constraints(node::ClassificationTreeNode)
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
  function constraints(node::ClassificationTreeNode, constraint_type::Union{Nothing,Type{<:ChildDirection}}=nothing)
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


function Base.show(io::IO, tree::ClassificationTree; without_constraints::Bool=false, kw...)
  if get(io, :in_forest, false)
    nleaves = length(collect(AT.Leaves(tree.root)))
    nbranches = length(collect(AT.PreOrderDFS(tree.root))) - nleaves
    print(io, "Tree Classifier with $nleaves leaf nodes and $nbranches branch nodes")
  elseif without_constraints
    AT.print_tree(io, tree.root; kw...)
  else
    constraint_generator = create_constraints_generator(tree)
    function node_constraint_printer(io::IO, node::ClassificationTreeNode; kw2...)
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

function random_tree(X, y; maxdepth=5, probsplit=0.5)
  attribute_labels = Dict(pairs(names(X)))

  root_attr = rand(1:ncol(X))
  root_threshold = rand(@view X[!, root_attr])

  root = branch(root_attr, root_threshold, attribute_labels_dict=attribute_labels)
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

  tree = ClassificationTree(root, X, y, NodeMap(1 => (node=root, rowmask=BitVector(ones(nrow(X))), parent=nothing)), maxdepth)

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


function random_trees(numtrees, X, y; maxdepth=5, probsplit=0.5)
  treechan = Channel{Any}(numtrees)

  @showprogress desc = "Generating Trees..." barglyphs = BarGlyphs("[=> ]") for i in 1:numtrees
    t = random_tree(X, y; maxdepth=maxdepth, probsplit=probsplit)

    put!(treechan, t)
  end

  all_trees = [take!(treechan) for i in 1:numtrees]

  return all_trees
end


function Base.display(trees::Vector{ClassificationTree})
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



function outcome_class_predictions(tree::ClassificationTree)
  ŷ = similar(tree.targets)
  ŷ.pool = tree.targets.pool

  leaf_nodes = map(x -> x[2], collect(filter(xp -> isleaf(xp[2].node), tree.nodemap)))
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
