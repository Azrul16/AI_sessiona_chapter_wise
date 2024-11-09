import math
import random


def objective_function(x):
    return x ** 2


def generate_neighbor(current_solution, temperature):
    return current_solution + random.uniform(-temperature, temperature)


def acceptance_probability(current_energy, new_energy, temperature):
    if new_energy < current_energy:
        return 1.0
    return math.exp((current_energy - new_energy) / temperature)


def simulated_annealing(initial_solution, max_iterations, initial_temperature, cooling_rate):
    current_solution = initial_solution
    current_energy = objective_function(current_solution)

    for iteration in range(max_iterations):
        temperature = initial_temperature * math.exp(-cooling_rate * iteration)

        new_solution = generate_neighbor(current_solution, temperature)
        new_energy = objective_function(new_solution)

        if acceptance_probability(current_energy, new_energy, temperature) > random.random():
            current_solution = new_solution
            current_energy = new_energy

    return current_solution, current_energy


# Example usage:
initial_solution = 2.0  # Replace with your initial solution
max_iterations = 1000
initial_temperature = 1.0
cooling_rate = 0.003

final_solution, final_energy = simulated_annealing(initial_solution, max_iterations, initial_temperature, cooling_rate)

print(f"Initial Solution: {initial_solution}")
print(f"Final Solution: {final_solution}")
print(f"Objective Function Value at Final Solution: {final_energy}")