import numpy as np

# Constants
alpha1, alpha2, alpha3, alpha4, alpha5 = 0.33, 0.25, 0.20, 0.12, 0.10
max_iterations = 100
population_size = 100  # Number of nodes used for implementation
num_sensors = 100  # Number of sensor nodes
network_dimensions = 1000  # Size of the network area
initial_energy = 0.5  # Initial energy of each sensor node in Joules
t_max = int(2 / 3 * max_iterations)  # Threshold for exploration vs exploitation phases

# Energy model parameters
E_elec = 5e-9  # Energy consumed by the electronics for 1 bit (J)
E_amp = 100e-12  # Free space model amplification (J/m^2)
E_DA = 50e-9  # Data aggregation energy (J/bit/signal)
s0 = np.sqrt(E_amp / E_elec)  # Threshold distance for energy model

# Sink node location
sink_node = np.array([50, 50])  # Sink located at the center of the 1000x1000 grid

# Define bounds for solutions
LB = 0  # Lower bound
UB = network_dimensions  # Upper bound

# Initialize population of sensor nodes
def initialize_population(pop_size, dimensions):
    return np.round(np.random.rand(pop_size, dimensions) * network_dimensions, 2)

# Initialize energy levels
def initialize_energy_levels(num_sensors, initial_energy):
    return np.array([initial_energy] * num_sensors)

# Update energy levels based on transmission and reception
def update_energy_levels(energy_levels, distances, num_bits):
    for i, d in enumerate(distances):
        transmit_cost = transmit_energy(num_bits, d)
        receive_cost = receive_energy(num_bits)
        energy_levels[i] = max(0, energy_levels[i] - (transmit_cost + receive_cost))
    return energy_levels

# Energy model functions
def transmit_energy(k, s):
    """Calculate the energy required to transmit k bits over a distance s."""
    if s <= s0:
        return k * E_elec + k * E_amp * s**2
    else:
        return k * E_elec + k * E_amp * s**4

def receive_energy(k):
    """Calculate the energy required to receive k bits."""
    return k * E_elec

# Fitness function for CH selection
def mayfly_fitness_function(population, sensors, sink_node, energy_levels, node_degrees):
    residual_energy = energy_levels
    g1 = residual_energy_metric(residual_energy)
    g2 = distance_metric_to_bs(sensors, sink_node)
    g3 = energy_consumption_rate(energy_levels)
    g4 = node_degree_metric(node_degrees)
    g5 = node_centrality_with_dimensions(sensors, population)
    return 1 / (alpha1 * g1 + alpha2 * g2 + alpha3 * g3 + alpha4 * g4 + alpha5 * g5)

# Supporting functions for fitness components
def residual_energy_metric(residual_energy):
    return 1 / np.sum(residual_energy + 1e-6)  # Avoid division by zero

def distance_metric_to_bs(sensors, sink_node):
    distances = np.linalg.norm(sensors - sink_node, axis=1)
    avg_distance = np.mean(distances)
    return np.sum(distances / (avg_distance + 1e-6))

def energy_consumption_rate(energy_levels):
    safe_energy_levels = np.maximum(energy_levels, 1e-6)  # Prevent divide by zero
    return np.sum(1 / safe_energy_levels)

def node_degree_metric(node_degrees):
    return np.sum(node_degrees)

def node_centrality_with_dimensions(sensors, population):
    distances = np.linalg.norm(sensors[:, None] - population, axis=2)
    avg_distances = np.mean(distances, axis=0)
    return np.sum(avg_distances / (network_dimensions + 1e-6))

# Mayfly velocity and position updates
def update_male_velocity(males, pbest, gbest, velocities, h=0.5, y1=0.5, y2=0.5, alpha=0.1):
    for i in range(len(males)):
        r2 = np.random.rand()
        velocities[i] = h * velocities[i] + y1 * np.exp(-alpha * r2) * (pbest[i] - males[i]) \
                        + y2 * np.exp(-alpha * r2) * (gbest - males[i])
        males[i] += velocities[i]
    return males, velocities

def update_female_velocity(females, males, velocities, h=0.5, y2=0.5, alpha=0.1):
    for i in range(len(females)):
        r_mf = np.linalg.norm(females[i] - males[i])
        velocities[i] = h * velocities[i] + y2 * np.exp(-alpha * r_mf) * (males[i] - females[i])
        females[i] += velocities[i]
    return females, velocities

# Perform crossover between male and female mayflies
def crossover(male, female, roff=0.5):
    """Perform crossover to generate offspring."""
    offspring1 = roff * female + (1 - roff) * male
    offspring2 = (1 - roff) * female + roff * male
    return offspring1, offspring2

# Aquila Algorithm Fitness Function (for Routing)
def aquila_fitness_function(CH, SNs, BS, residual_energy):
    """Evaluate fitness for Aquila routing optimization."""
    PsCHSN = np.sum(np.linalg.norm(SNs - CH, axis=1))
    PsSNBS = np.sum(np.linalg.norm(SNs - BS, axis=1))
    PRESN = 1 / (np.sum(residual_energy) + 1e-6)  # Avoid division by zero
    return 0.5 * (PsCHSN + PsSNBS + PRESN)

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
def hybrid_mayfly_aquila_algorithm(sensors, base_station, max_iterations=100):
    # Mayfly algorithm for CH selection
    male_population = initialize_population(population_size // 2, sensors.shape[1])
    female_population = initialize_population(population_size // 2, sensors.shape[1])
    energy_levels = initialize_energy_levels(num_sensors, initial_energy)
    node_degrees = np.random.randint(1, 5, size=num_sensors) # This generates an array of size 'num_sensors' with with random values between [1,5]
    fitness = mayfly_fitness_function(male_population, sensors, base_station, energy_levels, node_degrees)
    pbest = male_population.copy()
    gbest = male_population[np.argmax(fitness)]

    for iteration in range(max_iterations):
        male_population, male_velocities = update_male_velocity(male_population, pbest, gbest, np.zeros_like(male_population))
        female_population, female_velocities = update_female_velocity(female_population, male_population, np.zeros_like(female_population))

        distances = np.linalg.norm(sensors - base_station, axis=1)
        energy_levels = update_energy_levels(energy_levels, distances, num_bits=1)

        offspring = [crossover(m, f) for m, f in zip(male_population, female_population)]
        male_population, female_population = np.array(offspring)[:, 0], np.array(offspring)[:, 1]

        fitness = mayfly_fitness_function(male_population, sensors, base_station, energy_levels, node_degrees)
        pbest = np.where(fitness > mayfly_fitness_function(pbest, sensors, base_station, energy_levels, node_degrees), male_population, pbest)
        gbest = male_population[np.argmax(fitness)]

    # Aquila optimization for routing
    best_CH = gbest
    population = initialize_population(population_size, sensors.shape[1])
    best_route = None
    best_fitness = float('inf')

    for t in range(max_iterations):
        if t <= t_max:
            YM = np.mean(population, axis=0)
            Ybest = population[np.argmin([aquila_fitness_function(best_CH, sensors, base_station, energy_levels)])]
            for i in range(population_size):
                if np.random.rand() < 0.4:
                    population[i] = extended_exploration(population[i], Ybest, YM, t, max_iterations)
                else:
                    random_idx = np.random.randint(0, population_size)
                    population[i] = narrowed_exploration(population[i], Ybest, population[random_idx])
        else:
            Ybest = population[np.argmin([aquila_fitness_function(best_CH, sensors, base_station, energy_levels)])]
            for i in range(population_size):
                if np.random.rand() < 0.4:
                    random_idx = np.random.randint(0, population_size)
                    population[i] = extended_exploitation(population[i], Ybest, population[random_idx], LB, UB)
                else:
                    random_idx = np.random.randint(0, population_size)
                    population[i] = narrowed_exploitation(population[i], Ybest, population[random_idx])

        fitness_values = [aquila_fitness_function(best_CH, sensors, base_station, energy_levels) for CH in population]
        min_fitness = np.min(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_route = population[np.argmin(fitness_values)]

        # Energy loss during routing
        distances = np.linalg.norm(sensors - best_route, axis=1)
        energy_levels = update_energy_levels(energy_levels, distances, num_bits=1)

    return best_CH, best_route

# Example usage
sensors = initialize_population(num_sensors, 2)
base_station = np.array([50, 50])
best_CH, best_route = hybrid_mayfly_aquila_algorithm(sensors, base_station)
print("Best Cluster Heads:", best_CH)
print("Best Route:", best_route)