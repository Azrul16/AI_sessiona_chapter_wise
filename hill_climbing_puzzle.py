import numpy as np
from queue import PriorityQueue

# Define the initial and goal states
initial_state = np.array([[3, 8, 5], [7, 0, 4], [2, 1, 6]])
goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

# Calculate Manhattan Distance (heuristic function)
def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:
                x, y = divmod(state[i, j] - 1, 3)
                distance += abs(x - i) + abs(y - j)
    return distance

# Get valid neighboring states by swapping the empty tile (0)
def get_neighbors(state):
    neighbors = []
    x, y = np.argwhere(state == 0)[0]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = state.copy()
            new_state[x, y], new_state[nx, ny] = new_state[nx, ny], new_state[x, y]
            neighbors.append(new_state)

    return neighbors

# A* Search Algorithm
def a_star_search(start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {str(start): None}
    g_score = {str(start): 0}
    f_score = {str(start): manhattan_distance(start)}

    while not open_set.empty():
        _, current = open_set.get()
        current = np.array(current)

        # Check if the current state is the goal state
        if np.array_equal(current, goal):
            path = []
            while current is not None:
                path.append(current)
                current = came_from[str(current)]
            return path[::-1]  # Return the solution path

        # Explore neighboring states
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[str(current)] + 1
            neighbor_str = str(neighbor)

            if neighbor_str not in g_score or tentative_g_score < g_score[neighbor_str]:
                came_from[neighbor_str] = current
                g_score[neighbor_str] = tentative_g_score
                f_score[neighbor_str] = tentative_g_score + manhattan_distance(neighbor)
                open_set.put((f_score[neighbor_str], neighbor.tolist()))

    return None  # Return None if no solution is found

# Main block to execute the A* search
if __name__ == "__main__":
    solution_path = a_star_search(initial_state, goal_state)
    if solution_path:
        print("Solution Path:")
        for step, state in enumerate(solution_path):
            print(f"Step {step}:\n{state}\n")
    else:
        print("No solution found.")
