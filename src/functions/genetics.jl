function crossover(t1::ClassificationTree, t2::ClassificationTree; p1::Union{Int,Nothing}=nothing, p2::Union{Int,Nothing}=nothing)
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

function mutate!(tree::ClassificationTree, node_idx)
  random_outcome = create_outcome_randomiser(tree)
  random_decision = create_decision_randomiser(tree)
  selected_node_info = tree.nodemap[node_idx]
  current_mask = selected_node_info.rowmask
  current_node = selected_node_info.node
  if isbranch(current_node)
    prevattr = isroot(current_node) ? nothing : parent(current_node).attribute
    prevthreshold = isroot(current_node) ? nothing : parent(current_node).threshold
    new_decision = random_decision(current_mask, prevattr=prevattr, prevthreshold=prevthreshold)
    current_node.attribute = new_decision.attribute
    current_node.threshold = new_decision.threshold
  else
    new_outcome = random_outcome(current_mask)
    current_node.outcome = new_outcome
  end
  reset_nodemap!(tree)
  return tree
end

function mutate!(tree::ClassificationTree; num_mutations::Int=1, probmutation=0.4)
  total_num_mutations = 0

  while total_num_mutations < num_mutations
    selected_node_idx = rand(collect(keys(tree.nodemap)))
    if rand() < probmutation
      total_num_mutations += 1
      mutate!(tree, selected_node_idx)
    end
  end

  return tree
end

function evolve(
  trees::Vector{ClassificationTree};
  kwargs...
)
  metric_type = get(kwargs, :metric_type, InformednessFitness)
  metric = get(kwargs, :metric, metric_function(trees[1], metric_type))
  maxdepth = get(kwargs, :maxdepth, trees[1].maxdepth)
  target = get(kwargs, :target, LeafLabel)
  penalty = get(kwargs, :penalty, DepthPenalty)
  penalty_weight = get(kwargs, :penalty_weight, 0.05)
  probmutation = get(kwargs, :probmutation, 0.4)
  num_mutations = get(kwargs, :num_mutations, 1)
  eliteprop = get(kwargs, :eliteprop, 0.3)
  seed = get(kwargs, :seed, nothing)
  verbose = get(kwargs, :verbose, false)
  if !isnothing(seed)
    Random.seed!(seed)
  end

  N = length(trees)
  new_populaton = Vector{ClassificationTree}()

  fitnesses = evaluate_fitness(trees; kwargs...)

  sort_order = sortperm(fitnesses, rev=true)
  sorted_trees = trees[sort_order]
  sorted_fitnesses = fitnesses[sort_order]

  elite_pop_indices = 1:(Int(floor(N * eliteprop)))
  append!(new_populaton, sorted_trees[elite_pop_indices])

  total_fitness = sum(sorted_fitnesses)
  num_offspring = Int(ceil(N * (1 - eliteprop)))

  roulette_pointer_distance = total_fitness / num_offspring
  fitness_boundaries = [0, cumsum(sorted_fitnesses)...]

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
        maxdepth=maxdepth
      ),
      num_mutations=num_mutations,
      probmutation=probmutation
    )
    for (m, f) in zip(mothers, fathers)
  ]

  append!(new_populaton, children)

  return (
    offspring=new_populaton,
    best_fitness=sorted_fitnesses[1],
    best_tree=sorted_trees[1]
  )
end


function train(
  trees::Vector{ClassificationTree};
  generations::Int=100,
  kwargs...
)
  max_generations_stagnant = get(kwargs, :max_generations_stagnant, Int(floor(generations * 0.2)))
  current_population = trees
  fitness_history = Vector{Float64}()

  prevfitness = 0
  num_stagnant_generations = 0
  num_generations_evolved = 0
  current_best_fitness = 0

  @showprogress desc = "Training..." barglyphs = BarGlyphs("[=> ]") for i in 1:generations
    num_generations_evolved += 1
    offspring, best_fitness, best_tree = evolve(current_population; kwargs...)
    if i == 1
      current_best_fitness = best_fitness
    end
    current_population = offspring
    push!(fitness_history, best_fitness)

    if prevfitness >= best_fitness
      num_stagnant_generations += 1
    else
      num_stagnant_generations = 0
    end

    if num_stagnant_generations == max_generations_stagnant
      break
    end
    prevfitness = best_fitness
  end

  if num_generations_evolved < generations
    println("Stopped training due to stagnations (Total generations trained: $(num_generations_evolved))")
  end

  final_fitnesses = evaluate_fitness(current_population; kwargs...)
  final_order = sortperm(final_fitnesses, rev=true)
  final_population = current_population[final_order]
  final_sorted_fitnesses = final_fitnesses[final_order]
  push!(fitness_history, final_sorted_fitnesses[1])

  return (
    best_tree=final_population[1],
    final_population=final_population,
    final_fitnesses=final_fitnesses,
    fitness_history=fitness_history
  )
end
