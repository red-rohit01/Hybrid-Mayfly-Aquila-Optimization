import numpy as np
import matplotlib.pyplot as plt

# Constants
population_size = 100  # Number of candidate solutions
num_sensors = 100      # Number of sensor nodes
network_dimensions = 100  # Size of the network area
initial_energy = 0.5   # Initial energy of each sensor node in Joules

# Energy model parameters
E_elec = 50e-9         # Energy consumed by the electronics to transmit or receive 1 bit (J/bit)
E_fs = 10e-12          # Free space model amplification energy (J/bit/m^2)
E_mp = 0.0013e-12      # Multipath model amplification energy (J/bit/m^4)
s0 = np.sqrt(E_fs / E_mp)  # Threshold distance for energy model

# Sink node location
sink_node = np.array([50, 50])  # Sink located at the center of the 100x100 grid

# Define bounds for solutions
LB = 0  # Lower bound
UB = network_dimensions  # Upper bound

# Number of bits transmitted per message
num_bits = 1000  # Number of bits per packet

# Initialize population of sensor nodes
def initialize_population(pop_size, dimensions):
    return np.round(np.random.rand(pop_size, dimensions) * network_dimensions, 2)

# Initialize energy levels
def initialize_energy_levels(num_sensors, initial_energy):
    return np.array([initial_energy] * num_sensors)

# Update energy levels based on transmission and reception
def update_energy_levels(energy_levels, distances, num_bits):
    for i, d in enumerate(distances):
        if energy_levels[i] > 0:  # Only update if node is alive
            transmit_cost = transmit_energy(num_bits, d)
            receive_cost = receive_energy(num_bits)
            total_cost = transmit_cost + receive_cost
            energy_levels[i] = max(0, energy_levels[i] - total_cost)
    return energy_levels

# Energy model functions
def transmit_energy(k, s):
    """Calculate the energy required to transmit k bits over distances s with dynamic power adjustment."""
    s = np.array(s)  # Ensure s is a NumPy array
    E_amp = np.where(s <= s0, E_fs, E_mp)
    path_loss_exponent = np.where(s <= s0, 2, 4)
    energy = k * E_elec + k * E_amp * s**path_loss_exponent
    return energy

def receive_energy(k):
    """Calculate the energy required to receive k bits."""
    return k * E_elec

# Fitness function for CH selection with adaptive weighting
def mayfly_fitness_function(population, sensors, sink_node, energy_levels, node_degrees):
    # For each candidate CH position in the population, compute fitness
    num_candidates = len(population)
    fitness = np.zeros(num_candidates)
    
    # Adaptive weights based on average residual energy
    avg_residual_energy = np.mean(energy_levels)
    energy_ratio = avg_residual_energy / initial_energy  # between 0 and 1

    # Adjust weights dynamically
    adaptive_alpha1 = 0.33 * (1 + (1 - energy_ratio))  # Increase as energy decreases
    adaptive_alpha2 = 0.25 * energy_ratio              # Decrease as energy decreases
    adaptive_alpha3 = 0.20
    adaptive_alpha4 = 0.12
    adaptive_alpha5 = 0.10

    total_alpha = adaptive_alpha1 + adaptive_alpha2 + adaptive_alpha3 + adaptive_alpha4 + adaptive_alpha5
    # Normalize alphas
    adaptive_alpha1 /= total_alpha
    adaptive_alpha2 /= total_alpha
    adaptive_alpha3 /= total_alpha
    adaptive_alpha4 /= total_alpha
    adaptive_alpha5 /= total_alpha

    for i in range(num_candidates):
        candidate_CH = population[i]

        # Compute metrics for this candidate CH
        g1 = residual_energy_metric(candidate_CH, sensors, energy_levels)
        g2 = distance_metric_to_bs(candidate_CH, sink_node)
        g3 = energy_consumption_rate(candidate_CH, sensors, energy_levels)
        g4 = node_degree_metric(candidate_CH, sensors, node_degrees)
        g5 = node_centrality_with_dimensions(candidate_CH, sensors)

        # Compute total metric and fitness
        total_metric = (adaptive_alpha1 * g1 + adaptive_alpha2 * g2 + adaptive_alpha3 * g3 +
                        adaptive_alpha4 * g4 + adaptive_alpha5 * g5)
        fitness[i] = 1 / (total_metric + 1e-6)  # Add epsilon to prevent divide by zero

    return fitness

# Supporting functions for fitness components
def residual_energy_metric(candidate_CH, sensors, energy_levels):
    # Compute distance from sensors to candidate CH
    distances = np.linalg.norm(sensors - candidate_CH, axis=1)
    # Select nodes within a certain range (e.g., 30 units)
    range_radius = 30
    in_range_indices = np.where(distances <= range_radius)[0]
    # Sum residual energy of in-range nodes
    total_residual_energy = np.sum(energy_levels[in_range_indices])
    # Return inverse to minimize
    return 1 / (total_residual_energy + 1e-6)

def distance_metric_to_bs(candidate_CH, sink_node):
    distance = np.linalg.norm(candidate_CH - sink_node)
    return distance

def energy_consumption_rate(candidate_CH, sensors, energy_levels):
    # Compute energy required for sensors to transmit to candidate CH
    distances = np.linalg.norm(sensors - candidate_CH, axis=1)
    energy_consumed = transmit_energy(num_bits, distances)
    # Prevent division by zero
    safe_energy_levels = energy_levels.copy()
    safe_energy_levels[safe_energy_levels == 0] = 1e-6
    # Compute consumption rate
    consumption_rate = np.sum(energy_consumed / safe_energy_levels)
    return consumption_rate

def node_degree_metric(candidate_CH, sensors, node_degrees):
    # Compute distance from sensors to candidate CH
    distances = np.linalg.norm(sensors - candidate_CH, axis=1)
    # Select nodes within a certain range (e.g., 30 units)
    range_radius = 30
    in_range_indices = np.where(distances <= range_radius)[0]
    # Sum node degrees of in-range nodes
    total_node_degree = np.sum(node_degrees[in_range_indices])
    # Return inverse to minimize
    return 1 / (total_node_degree + 1e-6)

def node_centrality_with_dimensions(candidate_CH, sensors):
    # Compute distances from candidate CH to all sensors
    distances = np.linalg.norm(sensors - candidate_CH, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance / (network_dimensions + 1e-6)

# Mayfly velocity and position updates
def update_male_velocity(males, pbest, gbest, velocities, h=0.9, y1=1.0, y2=1.0, alpha=0.1):
    for i in range(len(males)):
        r2 = np.random.rand()
        velocities[i] = h * velocities[i] + y1 * np.exp(-alpha * r2) * (pbest[i] - males[i]) \
                        + y2 * np.exp(-alpha * r2) * (gbest - males[i])
        males[i] += velocities[i]
        # Ensure bounds are respected
        males[i] = np.clip(males[i], LB, UB)
    return males, velocities

def update_female_velocity(females, males, velocities, h=0.9, y2=1.0, alpha=0.1):
    for i in range(len(females)):
        r_mf = np.linalg.norm(females[i] - males[i])
        velocities[i] = h * velocities[i] + y2 * np.exp(-alpha * r_mf) * (males[i] - females[i])
        females[i] += velocities[i]
        # Ensure bounds are respected
        females[i] = np.clip(females[i], LB, UB)
    return females, velocities

# Perform crossover between male and female mayflies
def crossover(male, female, roff=0.5):
    """Perform crossover to generate offspring."""
    offspring1 = roff * female + (1 - roff) * male
    offspring2 = (1 - roff) * female + roff * male
    return offspring1, offspring2

# Mutation function to enhance diversity
def mutate_population(population, mutation_rate=0.1, mutation_strength=0.05):
    """Apply mutation to the population to increase diversity."""
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:
            mutation = np.random.uniform(-mutation_strength, mutation_strength, size=population.shape[1])
            population[i] += mutation
            # Ensure bounds are respected
            population[i] = np.clip(population[i], LB, UB)
    return population

# Aquila Algorithm Fitness Function (for Routing) with energy awareness
def aquila_fitness_function(CH, SNs, BS, residual_energy):
    """Evaluate fitness for Aquila routing optimization."""
    distances_to_CH = np.linalg.norm(SNs - CH, axis=1)
    distances_to_BS = np.linalg.norm(CH - BS)

    # Energy consumption for transmitting to CH
    energy_consumption_CH = transmit_energy(num_bits, distances_to_CH)
    # Energy consumption for transmitting from CH to BS
    energy_consumption_BS = transmit_energy(num_bits, distances_to_BS)

    # Total energy consumption
    total_energy_consumption = np.sum(energy_consumption_CH) + energy_consumption_BS

    # Penalize routes involving nodes with low residual energy
    low_energy_penalty = np.sum(1 / (residual_energy + 1e-6))

    # Fitness function combining distance and energy considerations
    return total_energy_consumption + low_energy_penalty

# Aquila optimization phases
def extended_exploration(Y, Ybest, YM, t, T):
    rand_vals = np.random.rand(*Y.shape)
    return Ybest * (1 - t / T) + YM - Ybest * rand_vals

def narrowed_exploration(Y, Ybest, Y_random):
    return Ybest + (Y_random - Ybest) * np.random.rand(*Y.shape)

def extended_exploitation(Y, Ybest, Y_random, LB, UB):
    return (Ybest - Y_random) + (UB - LB) * np.random.rand(*Y.shape) + LB

def narrowed_exploitation(Y, Ybest, Y_random):
    return Ybest - (Ybest - Y_random) * np.random.rand(*Y.shape)

# Hybrid Mayfly-Aquila algorithm
def hybrid_mayfly_aquila_algorithm(sensors, base_station):
    # Initialize variables to track node deaths
    first_node_dead = None
    half_nodes_dead = None
    all_nodes_dead = None

    # Arrays to store the number of alive and dead nodes per round
    alive_nodes_per_round = []
    dead_nodes_per_round = []

    # List to store average residual energy per round
    average_residual_energy_per_round = []

    # Initialize energy levels
    energy_levels = initialize_energy_levels(num_sensors, initial_energy)

    # Node degrees (for the fitness function)
    node_degrees = np.random.randint(1, 5, size=num_sensors)

    round_number = 0
    max_rounds = 10000  # Set a high number for maximum rounds

    # Variables to store the best CH and route
    best_CH = None
    best_route = None

    while True:
        round_number += 1

        # Mayfly algorithm for CH selection
        male_population = initialize_population(population_size // 2, sensors.shape[1])
        female_population = initialize_population(population_size // 2, sensors.shape[1])

        male_velocities = np.zeros_like(male_population)
        female_velocities = np.zeros_like(female_population)

        # Initialize pbest and fitness_pbest
        pbest = male_population.copy()
        fitness_pbest = mayfly_fitness_function(pbest, sensors, base_station, energy_levels, node_degrees)

        # Initialize gbest
        gbest = pbest[np.argmax(fitness_pbest)]

        max_iterations = 1 # Increased for better convergence
        for iteration in range(max_iterations):
            male_population, male_velocities = update_male_velocity(male_population, pbest, gbest, male_velocities)
            female_population, female_velocities = update_female_velocity(female_population, male_population, female_velocities)

            # Apply mutation to enhance diversity
            male_population = mutate_population(male_population)
            female_population = mutate_population(female_population)

            offspring = [crossover(m, f) for m, f in zip(male_population, female_population)]
            male_population, female_population = np.array(offspring)[:, 0], np.array(offspring)[:, 1]

            # Compute fitness for male_population
            fitness = mayfly_fitness_function(male_population, sensors, base_station, energy_levels, node_degrees)

            # Update pbest and fitness_pbest where fitness has improved
            update_indices = fitness > fitness_pbest
            pbest[update_indices] = male_population[update_indices]
            fitness_pbest[update_indices] = fitness[update_indices]

            # Update gbest based on the updated pbest
            gbest = pbest[np.argmax(fitness_pbest)]

        best_CH = gbest

        # Aquila optimization for routing
        population = initialize_population(population_size, sensors.shape[1])
        t_max = int(2 / 3 * max_iterations)
        best_route = None
        best_fitness = float('inf')

        for t in range(max_iterations):
            fitness_values = [aquila_fitness_function(CH, sensors, base_station, energy_levels) for CH in population]
            Ybest = population[np.argmin(fitness_values)]
            if t <= t_max:
                YM = np.mean(population, axis=0)
                for i in range(population_size):
                    if np.random.rand() < 0.4:
                        population[i] = extended_exploration(population[i], Ybest, YM, t, max_iterations)
                    else:
                        random_idx = np.random.randint(0, population_size)
                        population[i] = narrowed_exploration(population[i], Ybest, population[random_idx])
            else:
                for i in range(population_size):
                    if np.random.rand() < 0.4:
                        random_idx = np.random.randint(0, population_size)
                        population[i] = extended_exploitation(population[i], Ybest, population[random_idx], LB, UB)
                    else:
                        random_idx = np.random.randint(0, population_size)
                        population[i] = narrowed_exploitation(population[i], Ybest, population[random_idx])

            # Apply mutation to enhance diversity
            population = mutate_population(population)

            # Ensure population is within bounds
            population = np.clip(population, LB, UB)

            fitness_values = [aquila_fitness_function(CH, sensors, base_station, energy_levels) for CH in population]
            min_fitness = np.min(fitness_values)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_route = population[np.argmin(fitness_values)]

        # Energy loss during routing
        distances = np.linalg.norm(sensors - best_route, axis=1)
        energy_levels = update_energy_levels(energy_levels, distances, num_bits)

        # Record alive and dead nodes
        alive_nodes = np.count_nonzero(energy_levels > 0)
        dead_nodes = num_sensors - alive_nodes

        alive_nodes_per_round.append(alive_nodes)
        dead_nodes_per_round.append(dead_nodes)

        # Calculate average residual energy
        average_residual_energy = np.mean(energy_levels[energy_levels > 0]) if alive_nodes > 0 else 0
        average_residual_energy_per_round.append(average_residual_energy)

        # Check for first node dead
        if first_node_dead is None and dead_nodes >= 1:
            first_node_dead = round_number
            print(f"First node died at round {round_number}")

        # Check for 50% nodes dead
        if half_nodes_dead is None and dead_nodes >= num_sensors / 2:
            half_nodes_dead = round_number
            print(f"50% nodes died at round {round_number}")

        # Check for all nodes dead
        if all_nodes_dead is None and dead_nodes == num_sensors:
            all_nodes_dead = round_number
            print(f"All nodes died at round {round_number}")
            break  # Stop the simulation when all nodes are dead

        # Optional: Print progress every 100 rounds
        if round_number % 100 == 0:
            print(f"Round {round_number}: {alive_nodes} nodes alive, {dead_nodes} nodes dead")

    # Plotting the required graphs
    # 1. Graph showing the round number when 1st node dies, 50% nodes die, and all nodes die
    plt.figure(figsize=(10, 6))
    events = ['First Node Dead', '50% Nodes Dead', 'All Nodes Dead']
    rounds = [first_node_dead, half_nodes_dead, all_nodes_dead]
    plt.bar(events, rounds, color=['green', 'orange', 'red'])
    plt.title('Round Numbers for Node Death Events')
    plt.ylabel('Round Number')
    plt.show()

    # 2. Graph showing the number of alive and dead nodes after each round
    rounds_list = range(1, round_number + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(rounds_list, alive_nodes_per_round, label='Alive Nodes')
    plt.plot(rounds_list, dead_nodes_per_round, label='Dead Nodes')
    plt.title('Number of Alive and Dead Nodes per Round')
    plt.xlabel('Round Number')
    plt.ylabel('Number of Nodes')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Graph showing the average residual energy per round
    plt.figure(figsize=(12, 6))
    plt.plot(rounds_list, average_residual_energy_per_round, color='blue')
    plt.title('Average Residual Energy vs. Round Number')
    plt.xlabel('Round Number')
    plt.ylabel('Average Residual Energy (Joules)')
    plt.grid(True)
    plt.show()

    # Return the final results
    return best_CH, best_route

# Example usage
sensors = initialize_population(num_sensors, 2)
base_station = np.array([50, 50])
best_CH, best_route = hybrid_mayfly_aquila_algorithm(sensors, base_station)

print("Simulation completed with improved algorithm.")





