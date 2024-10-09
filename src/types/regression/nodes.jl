


mutable struct RegressionTreeNode <: AbstractTreeNode
  attribute::Union{Nothing,Int}
  threshold::Union{Nothing,Float64}
  outcome::Union{Nothing,Float64}
  parent::RefValue{Union{Nothing,RegressionTreeNode}}
  left::RefValue{Union{Nothing,RegressionTreeNode}}
  right::RefValue{Union{Nothing,RegressionTreeNode}}
  attribute_labels::Union{Nothing,LabelDict}
end

const OptionalRegressionNode = Union{Nothing,RegressionTreeNode}

outcomelabel(node::RegressionTreeNode) = "$(node.outcome)"

