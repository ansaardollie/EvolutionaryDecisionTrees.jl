
mutable struct DeterministicEvolutionaryDecisionTreeClassifier <: MMI.Deterministic
  generation_size::Int
  num_generations::Int
  max_depth::Int
  split_probability::Float64
  mutation_probability::Float64
  num_mutations::Int
  fitness_function::Any
  penalty_type::Type{<:PenaltyType}
  penalty_weight::Float64
  label_type::Type{<:ClassificationPredictionType}
  elite_proportion::Float64
  max_generations_stagnant::Int
  seed::Int
end

function MMI.clean!(model::DeterministicEvolutionaryDecisionTreeClassifier)
  warning = ""

  if model.generation_size < 1
    warning *= "Need generation_size >= 1. Resetting generation_size = 100\n"
    model.generation_size = 100
  end

  if model.num_generations < 1
    warning *= "Need num_generations >= 1. Resetting num_generations = 100\n"
    model.num_generations = 100
  end

  if model.max_depth < 1
    warning *= "Need max_depth >= 1. Resetting max_depth = 5\n"
    model.max_depth = 5
  end

  if !(0 < model.split_probability <= 1)
    warning *= "Need split_probability ∈ (0, 1]. Resetting split_probability = 0.5\n"
    model.split_probability = 0.5
  end

  if !(0 < model.mutation_probability <= 1)
    warning *= "Need mutation_probability ∈ (0,1]. Resetting mutation_probability = 0.5\n"
    model.mutation_probability = 0.5
  end

  if model.num_mutations < 0
    warning *= "Need num_mutations > 0. Resetting num_mutations = 1\n"
    model.num_mutations = 1
  end

  if !(0 < model.elite_proportion < 1)
    warning += "Need elite_proportion ∈ (0,1). Resetting elite_proportion = 0.1\n"
    model.elite_proportion = 0.1
  end

  if !(0 < model.max_generations_stagnant <= model.num_generations)
    warning *= "Need max_generations_stagnant ∈ (0, $(model.num_generations)). Resetting num_generations = $(Int(floor(0.2 * model.num_generations)))"
    model.max_generations_stagnant = Int(floor(0.2 * model.num_generations))
  end

  return warning
end


function DeterministicEvolutionaryDecisionTreeClassifier(;
  generation_size::Int=1000,
  num_generations::Int=100,
  max_depth::Int=5,
  split_probability::Float64=0.5,
  mutation_probability::Float64=0.5,
  num_mutations::Int=1,
  fitness_function_type::DataType=InformednessFitness,
  fitness_function::Union{Nothing,Function}=nothing,
  penalty_type::Type{<:PenaltyType}=NodePenalty,
  penalty_weight::Float64=0.05,
  prediction_type::Type{<:ClassificationPredictionType}=LeafLabel,
  elite_proportion::Float64=0.2,
  max_generations_stagnant::Union{Nothing,Int}=nothing,
  seed::Int=-1
)

  penalty_type <: PenaltyType || error("`penalty_type` = $(penalty_type)` is not a recognised PenatlyType")
  prediction_type <: ClassificationPredictionType || error("`label_type` = $(prediction_type) is not a recognised ClassificationPredictionType")

  fitness_function = isnothing(fitness_function) ? fitness_type_function(fitness_function_type) : fitness_function

  SMB.is_measure(fitness_function) || error("Fitness metric is not a recognised measure")

  model = DeterministicEvolutionaryDecisionTreeClassifier(
    generation_size,
    num_generations,
    max_depth,
    split_probability,
    mutation_probability,
    num_mutations,
    fitness_function,
    penalty_type,
    penalty_weight,
    prediction_type,
    elite_proportion,
    max_generations_stagnant,
    seed
  )

  message = MMI.clean!(model)

  isempty(message) || @warn message

  return model
end

function MMI.reformat(::DeterministicEvolutionaryDecisionTreeClassifier, X, y)
  if typeof(X) <: AbstractDataFrame && typeof(y) <: CategoricalVector
    return (X, y)
  elseif typeof(X) <: AbstractDataFrame
    return (X, categorical(y))
  elseif typeof(y) <: CategoricalVector
    return (DataFrame(X), y)
  else
    return (DataFrame(X), categorical(y))
  end
end

function MMI.reformat(::DeterministicEvolutionaryDecisionTreeClassifier, X)
  if typeof(X) <: AbstractDataFrame
    return (X,)
  else
    return (DataFrame(X),)
  end
end

function MMI.selectrows(::DeterministicEvolutionaryDecisionTreeClassifier, I, Xdf, y)
  return (view(Xdf, I, :), y[I])
end

function MMI.selectrows(::DeterministicEvolutionaryDecisionTreeClassifier, I, Xdf)
  return (view(Xdx, I, :),)
end

function MMI.fit(model::DeterministicEvolutionaryDecisionTreeClassifier, verbosity, X, y)

  if model.seed != -1
    Random.seed!(model.seed)
  end
  initial_population = random_classification_trees(model.generation_size, X, y; max_depth=model.max_depth, split_probability=model.split_probability)

  train_config = Dict(
    :num_generations => model.num_generations,
    :max_generations_stagnant => model.max_generations_stagnant,
    :verbosity => verbosity,
    :fitness_function => model.fitness_function,
    :max_depth => model.max_depth,
    :label_type => model.label_type,
    :penalty_type => model.penalty_type,
    :penalty_weight => model.penalty_weight,
    :mutation_probability => model.mutation_probability,
    :num_mutations => model.num_mutations,
    :elite_propprtion => model.elite_proportion
  )

  out = train(initial_population; train_config...)

  fitresult = (out.best_tree,)
  cache = (out.final_population,)
  report = (out.best_tree_history, out.fitness_history)

  return (fitresult, cache, report)
end

function MMI.predict(model::DeterministicEvolutionaryDecisionTreeClassifier, fitresult, Xnew)
  if model.label_type == LeafLabel
    return outcome_class_predictions(fitresult[1], Xnew)
  else
    return mode.(outcome_probability_predictions(fitresult[1], Xnew))
  end
end
