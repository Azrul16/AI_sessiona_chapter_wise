import random

def sliding_puzzle_problem(state):
    """
    Sliding puzzle problem function. Calculates the number of misplaced tiles.

    :param state: Current state of the puzzle.
    :return: The number of misplaced tiles.
    """
    goal_state = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, None]  # None represents the empty space
    ]

    misplaced_tiles = sum(state_i != goal_i for row_state, row_goal in zip(state, goal_state) for state_i, goal_i in zip(row_state, row_goal))
    return misplaced_tiles

def generate_initial_state():
    """
    Generates a random initial state for the sliding puzzle.

    :return: A random initial state.
    """
    numbers = list(range(1, 9))
    random.shuffle(numbers)
    numbers.append(None)  # Representing the empty space
    initial_state = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
    return initial_state

def generate_neighbors(state):
    """
    Generates neighboring states for the sliding puzzle.

    :param state: Current state of the puzzle.
    :return: A list of neighboring states.
    """
    neighbors = []
    empty_row, empty_col = find_empty_space(state)

    # Move empty space up
    if empty_row > 0:
        neighbor = [row.copy() for row in state]
        neighbor[empty_row][empty_col], neighbor[empty_row - 1][empty_col] = neighbor[empty_row - 1][empty_col], None
        neighbors.append(neighbor)

    # Move empty space down
    if empty_row < 2:
        neighbor = [row.copy() for row in state]
        neighbor[empty_row][empty_col], neighbor[empty_row + 1][empty_col] = neighbor[empty_row + 1][empty_col], None
        neighbors.append(neighbor)

    # Move empty space left
    if empty_col > 0:
        neighbor = [row.copy() for row in state]
        neighbor[empty_row][empty_col], neighbor[empty_row][empty_col - 1] = neighbor[empty_row][empty_col - 1], None
        neighbors.append(neighbor)

    # Move empty space right
    if empty_col < 2:
        neighbor = [row.copy() for row in state]
        neighbor[empty_row][empty_col], neighbor[empty_row][empty_col + 1] = neighbor[empty_row][empty_col + 1], None
        neighbors.append(neighbor)

    return neighbors

def find_empty_space(state):
    """
    Finds the row and column indices of the empty space in the puzzle state.

    :param state: Current state of the puzzle.
    :return: Row and column indices of the empty space.
    """
    for row_index, row in enumerate(state):
        if None in row:
            return row_index, row.index(None)

def hill_climbing(problem, max_iterations=1000):
    """
    Performs Hill Climbing search to find the optimal solution to a problem.

    :param problem: A function representing the problem. It should take a state as input and return its cost.
    :param max_iterations: Maximum number of iterations to perform.
    :return: The optimal state and its cost.
    """

    current_state = generate_initial_state()
    current_cost = problem(current_state)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_state)
        best_neighbor = min(neighbors, key=problem)

        if problem(best_neighbor) >= current_cost:
            break

        current_state = best_neighbor
        current_cost = problem(current_state)

    return current_state, current_cost

# Solve the sliding puzzle using Hill Climbing
initial_state = generate_initial_state()
optimal_state, optimal_cost = hill_climbing(sliding_puzzle_problem)

print("Initial State:")
for row in initial_state:
    print(row)

print("\nOptimal State:")
for row in optimal_state:
    print(row)

print("\nGoal State:")
goal_state = [
    [1, 2, 3],
    [4, 5, 7],
    [6, 8, None]
]
for row in goal_state:
    print(row)

print(f"\nOptimal Cost: {optimal_cost}")