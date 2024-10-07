module EvolutionaryDecisionTrees

include("requirements.jl")

# for n in names(@__MODULE__; all=true)
#   if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval)
#     @eval export $n
#   end
# end

include("types/nodes.jl")
include("types/tree.jl")
include("types/metrics.jl")

include("functions/genetics.jl")
include("functions/plot.jl")

export
  ChildDirection,
  ClassifierAccuracy,
  ClassifierBalancedAccuracy,
  ClassifierF1Score,
  ClassifierInformedness,
  ClassifierMarkedness,
  ClassifierMatthewsCorrelation,
  ClassifierTreeFitnessMetric,
  ClassifierTreeNode,
  ClassifierTreePenalty,
  ClassifierTreeTargetType,
  ConstraintDict,
  EvolutionaryDecisionTrees,
  LabelDict,
  LeafDistribution,
  LeafLabel,
  LeftChild,
  NodeConstraints,
  NodeMap,
  OptionalNode,
  PlotClassifierTree,
  RefValue,
  RightChild,
  TreeClassifier,
  TreeDepthPenalty,
  TreeNode,
  TreeNodesPenalty,
  attributes,
  attrlabel,
  attrlabels,
  branch,
  child!,
  childdir,
  create_attribute_randomiser,
  create_constraints_generator,
  create_decision_randomiser,
  create_empty_constraints_generator,
  create_left_constraints_generator,
  create_outcome_randomiser,
  create_parent_constraints_generator,
  create_right_constraints_generator,
  crossover,
  eval,
  evaluate_fitness,
  evolve,
  fitness,
  include,
  informedness,
  isbranch,
  isleaf,
  isleftchild,
  isrightchild,
  isroot,
  leaf,
  left,
  leftchild!,
  markedness,
  merge,
  metric_function,
  mutate!,
  nodelevel,
  nodenumber,
  outcome_class_predictions,
  outcome_probability_predictions,
  outcomelabel,
  parent,
  penalised_fitness,
  plotclassifiertree,
  plotclassifiertree!,
  prune,
  prune!,
  random_tree,
  random_trees,
  reset_nodemap!,
  right,
  rightchild!,
  subtree,
  train,
  treeplot,
  walk_tree!

end
