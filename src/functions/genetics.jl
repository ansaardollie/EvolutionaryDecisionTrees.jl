function crossover(t1::AbstractDecisionTree, t2::AbstractDecisionTree; p1::Union{Int,Nothing}=nothing, p2::Union{Int,Nothing}=nothing)
  mother = copy(t1)
  father = copy(t2)
  m_crossover_idx = isnothing(p1) ? rand(filter(k -> k != 1, keys(mother.nodemap))) : p1

  f_crossover_idx = isnothing(p2) ? rand(filter(k -> k != 1, keys(father.nodemap))) : p2

  m_crossover_node = mother.nodemap[m_crossover_idx].node

  f_crossover_node = father.nodemap[f_crossover_idx].node

  if isleftchild(m_crossover_node)
    m_crossover_node.parent[].left[] = nothing
    f_crossover_node.parent[] = nothing

    leftchild!(m_crossover_node.parent[], f_crossover_node)
  else
    m_crossover_node.parent[].right[] = nothing
    f_crossover_node.parent[] = nothing

    rightchild!(m_crossover_node.parent[], f_crossover_node)
  end
  reset_nodemap!(mother)
  return mother
end

"""
    mutate(trees::ClassificationTree, node_idx::Int)

Mutate `tree` at node with index `node_idx`.
"""
function mutate!(tree::ClassificationTree, node_idx::Int)
  leaf_prediction = create_prediction_selector(tree)
  random_decision = create_decision_randomiser(tree)
  selected_node_info = tree.nodemap[node_idx]
  current_mask = selected_node_info.rowmask
  current_node = selected_node_info.node
  if isbranch(current_node)
    prevattr = isroot(current_node) ? nothing : parent(current_node).attribute
    prevthreshold = isroot(current_node) ? nothing : parent(current_node).threshold
    new_decision = random_decision(current_mask, prevattr=prevattr, prevthreshold=prevthreshold)

    left_child = current_node.left[]
    right_child = current_node.right[]

    left_child.parent[] = nothing
    right_child.parent[] = nothing

    new_node = branch(ClassificationTreeNode, new_decision.attribute, new_decision.threshold, attribute_labels_dict=current_node.attribute_labels)

    leftchild!(new_node, left_child)
    rightchild!(new_node, right_child)

    if isroot(current_node)
      tree.root = new_node
    else
      parent_node = current_node.parent[]
      if isleftchild(current_node)
        parent_node.left[] = nothing
        leftchild!(parent_node, new_node)
      else
        parent_node.right[] = nothing
        rightchild!(parent_node, new_node)
      end
    end
  else
    new_outcome = leaf_prediction(current_mask)
    new_node = leaf(ClassificationTreeNode, new_outcome, attribute_labels_dict=current_node.attribute_labels)
    parent_node = current_node.parent[]
    if isleftchild(current_node)
      parent_node.left[] = nothing
      leftchild!(parent_node, new_node)
    else
      parent_node.right[] = nothing
      rightchild!(parent_node, new_node)
    end

  end
  reset_nodemap!(tree)
  return tree
end


"""
    mutate(trees::RegressionTree, node_idx::Int)

Mutate `tree` at node with index `node_idx`.
"""
function mutate!(tree::RegressionTree, node_idx::Int)
  random_decision = create_decision_randomiser(tree)
  selected_node_info = tree.nodemap[node_idx]
  current_mask = selected_node_info.rowmask
  current_node = selected_node_info.node
  if isbranch(current_node)
    prevattr = isroot(current_node) ? nothing : parent(current_node).attribute
    prevthreshold = isroot(current_node) ? nothing : parent(current_node).threshold
    new_decision = random_decision(current_mask, prevattr=prevattr, prevthreshold=prevthreshold)

    left_child = current_node.left[]
    right_child = current_node.right[]

    left_child.parent[] = nothing
    right_child.parent[] = nothing

    new_node = branch(RegressionTreeNode, new_decision.attribute, new_decision.threshold, attribute_labels_dict=current_node.attribute_labels)

    leftchild!(new_node, left_child)
    rightchild!(new_node, right_child)

    if isroot(current_node)
      tree.root = new_node
    else
      parent_node = current_node.parent[]
      if isleftchild(current_node)
        parent_node.left[] = nothing
        leftchild!(parent_node, new_node)
      else
        parent_node.right[] = nothing
        rightchild!(parent_node, new_node)
      end
    end
  else
    error("Cannot mutate leaf node in RegressionTree")
  end
  reset_nodemap!(tree)
  return tree
end


"""
    mutate(trees::ClassificationTree; kwargs...)

# Keyword Arguments
- `num_mutations::Int=1`: number of nodes to mutate.
- `mutation_probability::Float64=0.5`: probability of randomly selected node being mutated.
"""
function mutate!(tree::ClassificationTree; kwargs...)
  num_mutations = get(kwargs, :num_mutations, 1)
  mutation_probability = get(kwargs, :mutation_probability, 0.5)
  total_num_mutations = 0

  while total_num_mutations < num_mutations
    selected_node_idx = rand(collect(keys(tree.nodemap)))
    if rand() < mutation_probability
      total_num_mutations += 1
      mutate!(tree, selected_node_idx)
    end
  end

  return tree
end

"""
    mutate(trees::RegressionTree; kwargs...)

# Keyword Arguments
- `num_mutations::Int=1`: number of nodes to mutate.
- `mutation_probability::Float64=0.5`: probability of randomly selected node being mutated.
"""
function mutate!(tree::RegressionTree; kwargs...)
  num_mutations = get(kwargs, :num_mutations, 1)
  mutation_probability = get(kwargs, :mutation_probability, 0.5)
  total_num_mutations = 0

  available_node_indices = collect(keys(filter(x -> !(isroot(x[2].node)) && isbranch(x[2].node), tree.nodemap)))
  while length(available_node_indices) > 0 && total_num_mutations < num_mutations
    selected_node_idx = rand(available_node_indices)
    if rand() < mutation_probability
      total_num_mutations += 1
      mutate!(tree, selected_node_idx)
    end
  end

  return tree
end

"""
    evolve(trees::Vector{T}; kwargs...) where {T<:AbstractDecisionTree}

# Keyword Arguments
- `fitness_function::Any`: appropriate fitness function for the type of decision trees provided, if `nothing` used `fitness_function_type` to create one.
- `fitness_function_type::Union{ClassificationFitnessType, RegressionFitnessType}`: which fitness metric to evaluate the tree with. 
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `elite_proportion::Float64=0.3`: proportion of top-ranked trees to propogate unchanged into next generation.
- `num_mutations::Int=1`: number of nodes to mutate.
- `mutation_probability::Float64=0.5`: probability of randomly selected node being mutated.
- `max_depth::Int`: max depth of tree used for penalty calculation, defaults to the tree's `max_depth`.

"""
function evolve(trees::Vector{T}; kwargs...) where {T<:AbstractDecisionTree}
  max_depth = get(kwargs, :max_depth, trees[1].max_depth)
  mutation_probability = get(kwargs, :mutation_probability, 0.4)
  num_mutations = get(kwargs, :num_mutations, 1)
  elite_proportion = get(kwargs, :elite_proportion, 0.3)
  seed = get(kwargs, :seed, nothing)
  if !isnothing(seed)
    Random.seed!(seed)
  end

  N = length(trees)
  new_populaton = T[]

  trees_fitness = evaluate_fitness(trees; kwargs...)
  rescaled_fitness = map(fv -> fv.rescaled, trees_fitness)
  raw_fitness = map(fv -> fv.raw, trees_fitness)
  penalised_fitness = map(fv -> fv.penalised, trees_fitness)

  sort_order = sortperm(rescaled_fitness, rev=true)
  sorted_trees = trees[sort_order]
  sorted_rescaled_fitness = rescaled_fitness[sort_order]
  sorted_raw_fitness = raw_fitness[sort_order]
  sorted_penalised_fitness = penalised_fitness[sort_order]

  elite_pop_indices = 1:(Int(floor(N * elite_proportion)))
  append!(new_populaton, sorted_trees[elite_pop_indices])

  total_fitness = sum(sorted_rescaled_fitness)
  num_offspring = Int(ceil(N * (1 - elite_proportion)))

  roulette_pointer_distance = total_fitness / num_offspring
  fitness_boundaries = [0, cumsum(sorted_rescaled_fitness)...]

  mothers_roulette_start = rand(Uniform(0, roulette_pointer_distance))
  mothers_points = [mothers_roulette_start + (i - 1) * roulette_pointer_distance for i in 1:num_offspring]
  mothers_indices = [findfirst(f -> p < f, fitness_boundaries) - 1 for p in mothers_points]

  fathers_roulette_start = rand(Uniform(0, roulette_pointer_distance))
  fathers_points = [fathers_roulette_start + (i - 1) * roulette_pointer_distance for i in 1:num_offspring]
  fathers_indices = [findfirst(f -> p < f, fitness_boundaries) - 1 for p in fathers_points]

  mothers = sorted_trees[mothers_indices]
  fathers = sorted_trees[fathers_indices]

  children = [
    mutate!(
      prune!(
        crossover(m, f),
        max_depth=max_depth
      ),
      num_mutations=num_mutations,
      mutation_probability=mutation_probability
    )
    for (m, f) in zip(mothers, fathers)
  ]

  append!(new_populaton, children)

  return (
    offspring=new_populaton,
    starting_best_fitness=sorted_rescaled_fitness[1],
    starting_best_tree=sorted_trees[1],
    starting_population=sorted_trees,
    starting_population_raw_fitness=sorted_raw_fitness,
    starting_population_penalised_fitness=sorted_penalised_fitness,
    starting_population_rescaled_fitness=sorted_rescaled_fitness
  )
end

"""
  train(trees::Vector{<:AbstractDecisionTree}; kwargs...)

# Keyword Arguments
- `num_generations::Int=200`: number of generations to train for.
- `max_generations_stagnant::Int=Int(floor(num_generations * 0.2))`: stopping criterion, stop training if best tree's fitness does not improve for this number of generations.
- `fitness_function::Any`: appropriate fitness function for the type of decision trees provided, if `nothing` used `fitness_function_type` to create one.
- `fitness_function_type::Union{ClassificationFitnessType, RegressionFitnessType}`: which fitness metric to evaluate the tree with. 
- `penalty_type::PenaltyType=DepthPenalty`: what type of penalty is applied to fitness calculation.
- `penalty_weight::Float64=0.5`: how much weight is assigned to penalty in fitness calculation.
- `elite_proportion::Float64=0.3`: proportion of top-ranked trees to propogate unchanged into next generation.
- `num_mutations::Int=1`: number of nodes to mutate.
- `mutation_probability::Float64=0.5`: probability of randomly selected node being mutated.
- `max_depth::Int`: max depth of tree used for penalty calculation, defaults to the tree's `max_depth`.

"""
function train(
  trees::Vector{T};
  kwargs...
) where {T<:AbstractDecisionTree}
  num_generations = get(kwargs, :num_generations, 200)
  max_generations_stagnant = get(kwargs, :max_generations_stagnant, Int(floor(num_generations * 0.2)))
  verbosity = get(kwargs, :verbosity, 1)


  current_population = trees
  fitness_history = Vector{FitnessValues}()
  best_tree_history = Vector{T}()
  prevfitness = 0
  num_stagnant_generations = 0
  num_generations_evolved = 0
  current_best_fitness = 0

  if verbosity == 0
    for i in 1:num_generations
      num_generations_evolved += 1
      evolution_results = evolve(current_population; kwargs...)
      offspring = evolution_results.offspring
      starting_best_fitness = evolution_results.starting_best_fitness
      if i == 1
        current_best_fitness = starting_best_fitness
      end
      current_population = offspring
      push!(fitness_history, (
        raw=evolution_results.starting_population_raw_fitness[1],
        penalised=evolution_results.starting_population_penalised_fitness[1],
        rescaled=evolution_results.starting_population_rescaled_fitness[1]
      ))

      push!(best_tree_history, evolution_results.starting_best_tree)

      if prevfitness >= starting_best_fitness
        num_stagnant_generations += 1
      else
        num_stagnant_generations = 0
      end

      if num_stagnant_generations == max_generations_stagnant
        break
      end
      prevfitness = starting_best_fitness
    end
  else
    @showprogress desc = "Training..." barglyphs = BarGlyphs("[=> ]") for i in 1:num_generations
      num_generations_evolved += 1
      evolution_results = evolve(current_population; kwargs...)
      offspring = evolution_results.offspring
      starting_best_fitness = evolution_results.starting_best_fitness
      if i == 1
        current_best_fitness = starting_best_fitness
      end
      current_population = offspring
      push!(fitness_history, (
        raw=evolution_results.starting_population_raw_fitness[1],
        penalised=evolution_results.starting_population_penalised_fitness[1],
        rescaled=evolution_results.starting_population_rescaled_fitness[1]
      ))

      push!(best_tree_history, evolution_results.starting_best_tree)

      if prevfitness >= starting_best_fitness
        num_stagnant_generations += 1
      else
        num_stagnant_generations = 0
      end

      if num_stagnant_generations == max_generations_stagnant
        break
      end
      prevfitness = starting_best_fitness
    end
  end


  if num_generations_evolved < num_generations && verbosity > 0
    println("Stopped training due to stagnations (Total generations trained: $(num_generations_evolved))")
  end

  final_population_fitness_values = evaluate_fitness(current_population; kwargs...)
  final_population_rescaled_fitness = map(fv -> fv.rescaled, final_population_fitness_values)
  final_population_raw_fitness = map(fv -> fv.raw, final_population_fitness_values)
  final_population_penalised_fitness = map(fv -> fv.penalised, final_population_fitness_values)

  best_tree_index = argmax(final_population_rescaled_fitness)
  final_best_tree = current_population[best_tree_index]
  push!(fitness_history, (
    raw=final_population_raw_fitness[best_tree_index],
    penalised=final_population_penalised_fitness[best_tree_index],
    rescaled=final_population_rescaled_fitness[best_tree_index]
  ))

  push!(best_tree_history, final_best_tree)
  # final_order = sortperm(final_population_rescaled_fitness, rev=true)
  # final_population = current_population[final_order]
  # final_sorted_fitnesses = final_population_rescaled_fitness[final_order]
  # push!(fitness_history, (raw = final))

  return (
    best_tree=final_best_tree,
    fitness_history=fitness_history,
    best_tree_history=best_tree_history,
    final_population=current_population
  )
end
