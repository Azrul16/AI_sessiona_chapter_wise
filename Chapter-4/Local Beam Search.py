import random


def generate_initial_states(num_states, state_size):
    return [[random.randint(0, 1) for _ in range(state_size)] for _ in range(num_states)]


def evaluate_states(states, evaluation_function):
    return [(state, evaluation_function(state)) for state in states]


def generate_neighbors(state):
    """
    Generate neighboring states by flipping a random bit.

    :param state: Current state.
    :return: List of neighboring states.
    """
    neighbors = [state.copy() for _ in range(5)]
    for neighbor in neighbors:
        bit_to_flip = random.randint(0, len(state) - 1)
        neighbor[bit_to_flip] = 1 - neighbor[bit_to_flip]  # Flip the bit
    return neighbors


def local_beam_search(num_states, state_size, max_iterations, evaluation_function):
    current_states = generate_initial_states(num_states, state_size)

    for iteration in range(max_iterations):
        evaluated_states = evaluate_states(current_states, evaluation_function)

        # Sort states by their evaluation function
        evaluated_states.sort(key=lambda x: x[1])

        # Select the top states to be the current states
        current_states = [state for state, _ in evaluated_states[:num_states]]

        # Check if the best state satisfies the termination condition
        if evaluation_function(current_states[0]) == 0:
            return current_states[0]

        # Generate neighbors for the current states
        neighbors = [generate_neighbors(state) for state in current_states]
        neighbors = [neighbor for sublist in neighbors for neighbor in sublist]  # Flatten the list

        # Evaluate the neighbors
        evaluated_neighbors = evaluate_states(neighbors, evaluation_function)

        # Combine the current states and the neighbors
        all_states = evaluated_states + evaluated_neighbors

        # Select the top states to be the current states
        all_states.sort(key=lambda x: x[1])
        current_states = [state for state, _ in all_states[:num_states]]

    # If the termination condition is not met, return the best state found so far
    return current_states[0]


# Example Usage:
def example_evaluation_function(state):
    return sum(state)


num_states = 5
state_size = 10
max_iterations = 100

best_state = local_beam_search(num_states, state_size, max_iterations, example_evaluation_function)
print(f"Best State: {best_state}")
print(f"Evaluation at Best State: {example_evaluation_function(best_state)}")
