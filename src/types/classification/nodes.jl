

struct ClassificationTreeNode <: AbstractTreeNode
  attribute::Union{Nothing,Int}
  threshold::Union{Nothing,Float64}
  outcome::Union{Nothing,CategoricalValue}
  parent::RefValue{Union{Nothing,ClassificationTreeNode}}
  left::RefValue{Union{Nothing,ClassificationTreeNode}}
  right::RefValue{Union{Nothing,ClassificationTreeNode}}
  attribute_labels::Union{Nothing,LabelDict}
end

const OptionalClassificationNode = Union{Nothing,ClassificationTreeNode}

outcomelabel(node::ClassificationTreeNode) = "$(node.outcome)"

