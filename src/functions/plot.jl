
function Base.convert(::Type{SimpleDiGraph}, tree::ClassificationTree; maxdepth=AT.treeheight(tree))
  if maxdepth == -1
    maxdepth = AT.treeheight(tree.root)
  end

  g = SimpleDiGraph()
  properties = Any[]
  edge_props = Any[]
  walk_tree!(tree, tree.root, g, maxdepth, properties)
  return g, properties, edge_props

end

function walk_tree!(tree::ClassificationTree, node::ClassificationTreeNode, g, depthleft, properties)

  nn = nodenumber(tree, node)
  nobs = sum(tree.nodemap[nn].rowmask)
  if isleaf(node)
    add_vertex!(g)
    push!(properties, (node, nn, "$(outcomelabel(node))", nobs))
    return vertices(g)[end]
  else
    add_vertex!(g)

    if depthleft == 0
      push!(properties, (nothing, nothing, "...", nothing))
      return vertices(g)[end]
    else
      depthleft -= 1
    end


    current_vertex = vertices(g)[end]
    threshold = node.threshold
    attribute = attrlabel(node)

    push!(properties, (node, nn, "$(attribute) < $(round(threshold, digits=3))", nobs))
    child = walk_tree!(tree, node.left[], g, depthleft, properties)

    add_edge!(g, current_vertex, child)

    child = walk_tree!(tree, node.right[], g, depthleft, properties)

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


function treeplot(tree::ClassificationTree; kwargs...)
  f, ax, h = plotclassifiertree(tree; kwargs...)
  hidedecorations!(ax)
  hidespines!(ax)
  return f
end

function Makie.plot!(plt::PlotClassifierTree{<:Tuple{ClassificationTree}})

  @extract plt leafcolor, textcolor, nodecolormap, nodecolor, maxdepth

  tree = plt[1]

  tmpObs = @lift convert(SimpleDiGraph, $tree; maxdepth=$maxdepth)
  graph = @lift $tmpObs[1]
  properties = @lift $tmpObs[2]

  all_labels = @lift [string(p[3]) for p in $properties]

  nlabels_color = map(properties, all_labels, leafcolor, textcolor, nodecolormap) do properties, all_labels, leafcolor, textcolor, nodecolormap
    leaf_ix = findall([isnothing(p[1]) ? false : isleaf(p[1]) for p in properties])
    leaf_label_texts = all_labels[leaf_ix]   #[p[1] for p in split.(leaf_labels[leaf_ix], ":")]
    unique_labels = sort(unique(leaf_label_texts))
    inidividual_leaf_colors = resample_cmap(nodecolormap, length(unique_labels))
    nlabels_color = Any[isnothing(p[1]) ? textcolor : isbranch(p[1]) ? textcolor : leafcolor for p in properties]
    for (ix, uLV) = enumerate(unique_labels)
      ixV = leaf_label_texts .== uLV
      nlabels_color[leaf_ix[ixV]] .= inidividual_leaf_colors[ix]
    end
    return nlabels_color
  end

  numbered_labels = @lift ["$(p[2]): $(p[3])\n# obs: $(p[4])" for p in $properties]


  graphplot!(
    plt,
    graph;
    layout=Buchheim(),
    nlabels=numbered_labels,
    node_size=120,
    node_color=nodecolor,
    nlabels_color=nlabels_color,
    nlabels_align=(:center, :center),
    nlabels_attr=(
      font="Open Sans Bold",
    )
  )

  return plt
end

