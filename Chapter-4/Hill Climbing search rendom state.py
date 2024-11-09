import random


def hill_climbing(problem, max_iterations=1000):


    current_state = generate_random_state()  # Initial random state
    current_cost = problem(current_state)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_state)  # Generate neighboring states
        best_neighbor = min(neighbors, key=problem)  # Choose the neighbor with the lowest cost

        if problem(best_neighbor) >= current_cost:
            break  # Stop if no neighbor has a lower cost

        current_state = best_neighbor
        current_cost = problem(current_state)

    return current_state, current_cost


# Example usage:
def example_problem(state):
    """
    Example problem function. Replace this with your specific problem function.
    """
    return sum(state)  # In this example, the cost is the sum of the elements in the state


def generate_random_state():
    """
    Example function to generate a random state. Replace this with your specific state generation logic.
    """
    return [random.randint(1, 10) for _ in range(5)]  # In this example, a state is a list of 5 random integers


def generate_neighbors(state):
    """
    Example function to generate neighboring states. Replace this with your specific neighbor generation logic.
    """
    neighbors = []
    for i in range(len(state)):
        neighbor = state.copy()
        neighbor[i] += random.randint(-1, 1)  # In this example, perturb each element of the state randomly
        neighbors.append(neighbor)
    return neighbors


# Solve the example problem using Hill Climbing
optimal_state, optimal_cost = hill_climbing(example_problem)

print(f"Optimal State: {optimal_state}")
print(f"Optimal Cost: {optimal_cost}")
