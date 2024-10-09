module EvolutionaryDecisionTrees

include("requirements.jl")

include("types/global.jl")
include("types/classification/nodes.jl")
include("types/classification/tree.jl")
include("types/classification/metrics.jl")

include("types/regression/nodes.jl")
include("types/regression/tree.jl")
include("types/regression/metrics.jl")

include("functions/genetics.jl")
include("functions/plot.jl")


include("mlj/mlj.jl")

export
  AbstractDecisionTree,
  AbstractTreeNode,
  AccuracyFitness,
  AdjustedRSquaredFitness,
  BalancedAccuracyFitness,
  BinaryF1ScoreFitness,
  ChildDirection,
  ClassificationFitnessMetric,
  ClassificationTree,
  ClassificationTreeNode,
  ConstraintDict,
  DepthPenalty,
  DeterministicEvolutionaryDecisionTreeClassifier,
  InformednessFitness,
  LabelDict,
  LeafDistribution,
  LeafLabel,
  LeafPredictionType,
  LeftChild,
  MarkednessFitness,
  MatthewsCorrelationFitness,
  MeanPrediction,
  MedianPrediction,
  MidpointPrediction,
  MultiF1ScoreFitness,
  NodeConstraints,
  NodeMap,
  NodeMapDetails,
  NodePenalty,
  OptionalClassificationNode,
  OptionalRegressionNode,
  PenaltyType,
  PlotClassifierTree,
  RSquaredFitness,
  RandomPrediction,
  RefValue,
  RegressionFitnessMetric,
  RegressionTree,
  RegressionTreeNode,
  RightChild,
  R²,
  TargetType,
  attributes,
  attrlabel,
  attrlabels,
  branch,
  child!,
  childdir,
  create_adjusted_R²,
  create_attribute_randomiser,
  create_constraints_generator,
  create_decision_randomiser,
  create_empty_constraints_generator,
  create_left_constraints_generator,
  create_nodemap,
  create_outcome_randomiser,
  create_parent_constraints_generator,
  create_prediction_selector,
  create_right_constraints_generator,
  crossover,
  evaluate_fitness,
  evolve,
  fitness,
  fitness_function,
  informedness,
  isbranch,
  isleaf,
  isleftchild,
  isrightchild,
  isroot,
  leaf,
  leaf_predictor,
  left,
  leftchild!,
  markedness,
  merge,
  mutate!,
  nodelevel,
  nodenumber,
  outcome_class_predictions,
  outcome_predictions,
  outcome_probability_predictions,
  outcomelabel,
  parent,
  penalised_fitness,
  plotclassifiertree,
  plotclassifiertree!,
  prune,
  prune!,
  random_classification_tree,
  random_classification_trees,
  random_regression_tree,
  random_regression_trees,
  reset_nodemap!,
  right,
  rightchild!,
  subtree,
  train,
  tree_type,
  treeplot,
  walk_tree!
end
