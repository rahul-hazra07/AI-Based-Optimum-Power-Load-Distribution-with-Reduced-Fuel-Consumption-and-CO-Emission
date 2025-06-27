import pandas as pd
import random
import matplotlib.pyplot as plt

# Load dataset
generators = pd.read_csv("generatorDataset.csv")
print("Number of available generators:", len(generators))

# Input total demand
total_demand = float(input("Enter total power demand (MW): "))

# Genetic Algorithm Parameters
population_size = 150
generations = 200
mutation_rate = 0.3
top_k = 10
max_tolerance_percent = 0.1  # 0.1% power mismatch tolerance
diversity_injection_rate = 0.1

# Fitness function using quadratic fuel and emission cost formulas
def fitness(individual):
    total_power = sum(g['power'] for g in individual)
    total_cost = 0

    for g in individual:
        Pi = g['power']
        # Fuel Cost: a*P^2 + b*P + c
        fuel_cost = g['a'] * Pi**2 + g['b'] * Pi + g['c']
        # Emission Cost: alpha*P^2 + beta*P + gamma
        emission_cost = g['alpha'] * Pi**2 + g['beta'] * Pi + g['gamma']
        total_cost += fuel_cost + emission_cost

    power_diff = abs(total_demand - total_power)
    power_diff_percent = (power_diff / total_demand) * 100 if total_demand > 0 else 0

    if power_diff_percent > max_tolerance_percent:
        return float('inf')  # Penalize infeasible solutions

    fitness_penalty = power_diff * 100
    return total_cost + fitness_penalty

# Create individual using Random Equal Utilization method
def create_individual_random_equal_utilization(generators, demand):
    # Calculate total maximum capacity
    total_max_capacity = generators['max_power_limit'].sum()

    # Check if demand is feasible
    if demand > total_max_capacity:
        print(f"Warning: Demand ({demand} MW) exceeds total capacity ({total_max_capacity} MW)")
        return None

    # Calculate minimum utilization percentage needed
    min_utilization = demand / total_max_capacity

    # Choose random utilization percentage between minimum and 150% of minimum
    max_utilization = min(1.0, min_utilization * 1.5)
    utilization_percentage = random.uniform(min_utilization, max_utilization)

    individual = []
    total_assigned = 0

    for _, row in generators.iterrows():
        # Each generator operates at the same random utilization percentage
        assigned_power = row['max_power_limit'] * utilization_percentage

        individual.append({
            'generator_id': row['generator_id'],
            'power': assigned_power,
            'a': row['a'],
            'b': row['b'],
            'c': row['c'],
            'alpha': row['alpha'],
            'beta': row['beta'],
            'gamma': row['gamma'],
            'max_power_limit': row['max_power_limit']
        })
        total_assigned += assigned_power

    # Scale to exactly match demand
    if total_assigned > 0:
        scale_factor = demand / total_assigned
        for gene in individual:
            gene['power'] *= scale_factor

    return individual

# Create individual using original random method
def create_individual_original(generators, demand):
    selected = generators.sample(frac=1).reset_index(drop=True)
    total_power = 0
    individual = []

    for _, row in selected.iterrows():
        if total_power >= demand:
            break
        available = min(row['max_power_limit'], demand - total_power)
        if available > 0:
            individual.append({
                'generator_id': row['generator_id'],
                'power': available,
                'a': row['a'],
                'b': row['b'],
                'c': row['c'],
                'alpha': row['alpha'],
                'beta': row['beta'],
                'gamma': row['gamma'],
                'max_power_limit': row['max_power_limit']
            })
            total_power += available
    return individual

# Standard crossover
def crossover(parent1, parent2):
    child = []
    used_ids = set()
    for parent in [parent1, parent2]:
        for gene in parent:
            if gene['generator_id'] not in used_ids:
                used_ids.add(gene['generator_id'])
                child.append(gene.copy())
    return child

# Mutation
def mutate(individual, mutation_rate, demand):
    for gene in individual:
        if random.random() < mutation_rate:
            change = random.uniform(-0.1, 0.1) * gene['power']
            gene['power'] = min(max(gene['power'] + change, 0), gene['max_power_limit'])

    # Scale to match demand
    total_power = sum(g['power'] for g in individual)
    if total_power == 0:
        return individual
    scale = demand / total_power
    for gene in individual:
        gene['power'] = min(gene['max_power_limit'], gene['power'] * scale)
    return individual

# Initialize population with hybrid approach
population = []
methods = ['random_equal', 'original']
method_weights = [0.7, 0.3]  # 70% random_equal, 30% original

for i in range(population_size):
    # Randomly choose method based on weights
    method = random.choices(methods, weights=method_weights)[0]
    
    if method == 'random_equal':
        individual = create_individual_random_equal_utilization(generators, total_demand)
    else:
        individual = create_individual_original(generators, total_demand)

    if individual is not None:
        population.append(individual)

if len(population) == 0:
    print("Failed to create any feasible individuals!")
    exit()

best_fitness = []

# Evolution loop
for gen in range(generations):
    population.sort(key=fitness)
    best_fitness.append(fitness(population[0]))

    # Keep top performers
    new_population = population[:top_k]

    # Add diversity injection every 10 generations
    if gen % 10 == 0 and gen > 0:
        num_random = int(population_size * diversity_injection_rate)
        for _ in range(num_random):
            method = random.choice(['original', 'random_equal'])
            if method == 'random_equal':
                new_individual = create_individual_random_equal_utilization(generators, total_demand)
            else:
                new_individual = create_individual_original(generators, total_demand)
            if new_individual is not None:
                new_population.append(new_individual)

    # Generate offspring
    while len(new_population) < population_size:
        # Tournament selection
        tournament_size = 3
        tournament = random.sample(population[:min(50, len(population))], tournament_size)
        p1 = min(tournament, key=fitness)

        tournament = random.sample(population[:min(50, len(population))], tournament_size)
        p2 = min(tournament, key=fitness)

        child = crossover(p1, p2)
        child = mutate(child, mutation_rate, total_demand)

        new_population.append(child)

    population = new_population

    if gen % 50 == 0:
        best_fitness_val = best_fitness[-1] if best_fitness[-1] != float('inf') else 'Infeasible'
        print(f"Generation {gen}: Best fitness = {best_fitness_val}")

        # Calculate population diversity
        if gen % 100 == 0:
            costs = [fitness(ind) for ind in population[:10] if fitness(ind) != float('inf')]
            if costs:
                diversity = max(costs) - min(costs)
                print(f"  Population diversity: {diversity:.2f}")

# Extract top 5 unique feasible solutions
unique_costs = set()
printed = 0
i = 0

print("\nTop 5 Unique Optimal Solutions:\n")
population.sort(key=fitness)

while printed < 5 and i < len(population):
    sol = population[i]
    total_power = sum(g['power'] for g in sol)
    total_cost = 0
    for g in sol:
        Pi = g['power']
        fuel_cost = g['a'] * Pi**2 + g['b'] * Pi + g['c']
        emission_cost = g['alpha'] * Pi**2 + g['beta'] * Pi + g['gamma']
        total_cost += fuel_cost + emission_cost

    power_diff = abs(total_demand - total_power)
    power_diff_percent = (power_diff / total_demand) * 100

    if power_diff_percent <= max_tolerance_percent:
        rounded_cost = round(total_cost, 2)
        if rounded_cost not in unique_costs:
            unique_costs.add(rounded_cost)
            printed += 1
            print(f"Solution {printed}:")
            if printed == 1:
                print("Optimal Solution")
            print("Number of generators used:", len(sol))
            print("Generator Details:")

            # Calculate costs for each generator
            total_fuel_cost = 0
            total_emission_cost = 0
            
            for g in sol:
                Pi = g['power']
                fuel_cost = g['a'] * Pi**2 + g['b'] * Pi + g['c']
                emission_cost = g['alpha'] * Pi**2 + g['beta'] * Pi + g['gamma']
                total_fuel_cost += fuel_cost
                total_emission_cost += emission_cost
                
                utilization = (g['power'] / g['max_power_limit']) * 100
                print(f"  Generator {g['generator_id']}: Power = {g['power']:.2f} MW, "
                      f"Utilization = {utilization:.1f}%")

            print(f"Total Power: {total_power:.2f} MW")
            print(f"Total Fuel Cost: {total_fuel_cost:.2f}")
            print(f"Total Emission Cost: {total_emission_cost:.2f}")
            print(f"Total Cost: {total_cost:.2f}")
            print()
    i += 1

# Run 30 rounds with the same demand to test consistency
print("\n" + "="*50)
print("RUNNING 30 ROUNDS FOR CONSISTENCY ANALYSIS")
print("="*50)

round_costs = []
for round_num in range(0, 30):
    print(f"Running Round {round_num}...")
    
    # Initialize population for this round
    round_population = []
    for i in range(population_size):
        method = random.choices(methods, weights=method_weights)[0]
        
        if method == 'random_equal':
            individual = create_individual_random_equal_utilization(generators, total_demand)
        else:
            individual = create_individual_original(generators, total_demand)

        if individual is not None:
            round_population.append(individual)

    if len(round_population) == 0:
        print(f"Failed to create feasible individuals for round {round_num}!")
        continue

    round_best_fitness = []

    # Evolution loop for this round
    for gen in range(generations):
        round_population.sort(key=fitness)
        round_best_fitness.append(fitness(round_population[0]))

        # Keep top performers
        new_population = round_population[:top_k]

        # Add diversity injection every 10 generations
        if gen % 10 == 0 and gen > 0:
            num_random = int(population_size * diversity_injection_rate)
            for _ in range(num_random):
                method = random.choice(['original', 'random_equal'])
                if method == 'random_equal':
                    new_individual = create_individual_random_equal_utilization(generators, total_demand)
                else:
                    new_individual = create_individual_original(generators, total_demand)
                if new_individual is not None:
                    new_population.append(new_individual)

        # Generate offspring
        while len(new_population) < population_size:
            tournament_size = 3
            tournament = random.sample(round_population[:min(50, len(round_population))], tournament_size)
            p1 = min(tournament, key=fitness)

            tournament = random.sample(round_population[:min(50, len(round_population))], tournament_size)
            p2 = min(tournament, key=fitness)

            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, total_demand)

            new_population.append(child)

        round_population = new_population

    # Get the best cost for this round
    round_population.sort(key=fitness)
    best_round_cost = fitness(round_population[0])
    round_costs.append(best_round_cost)
    print(f"Round {round_num} Best Cost: {best_round_cost:.2f}")

# Display round results
print(f"\n30 ROUNDS RESULTS:")
print(f"Average Cost: {sum(round_costs)/len(round_costs):.2f}")
print(f"Best Cost: {min(round_costs):.2f}")
print(f"Worst Cost: {max(round_costs):.2f}")
print(f"Standard Deviation: {(sum((c - sum(round_costs)/len(round_costs))**2 for c in round_costs) / len(round_costs))**0.5:.2f}")

# Plot fitness trend for original run
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(generations), best_fitness)
plt.xlabel("Generation")
plt.ylabel("Optimal Cost")
plt.title("Generation vs Optimal Cost (Single Run)")
plt.grid(True)

# Plot 30 rounds cost variation
plt.subplot(1, 2, 2)
plt.plot(range(1, 31), round_costs, 'bo-', linewidth=2, markersize=6)
plt.xlabel("Round")
plt.ylabel("Best Cost")
plt.title("Cost Variation Across 30 Rounds")
plt.grid(True)
plt.xticks(range(0, 31, 5))

plt.tight_layout()
plt.show()

# Display results
print("\nMethod Used: Hybrid (70% Random Equal Utilization + 30% Original)")
print("This provides good population diversity while exploring efficient equal utilization solutions")

print(f"\nOriginal Run - Final Best Fitness: {best_fitness[-1]:.2f}")
improvement = best_fitness[0] - best_fitness[-1] if len(best_fitness) > 1 else 0
print(f"Total Improvement: {improvement:.2f}")
print(f"Generations with improvement: {sum(1 for i in range(1, len(best_fitness)) if best_fitness[i] < best_fitness[i-1])}")