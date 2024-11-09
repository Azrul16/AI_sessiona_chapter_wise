def dls(graph, start, goal, max_depth):
    """
    Finds a path from the start node to the goal node in a graph using Depth-Limited Search.

    :param graph: The graph represented as an adjacency list.
    :param start: The starting node.
    :param goal: The goal node.
    :param max_depth: The maximum depth to explore.
    :return: The path from start to goal, or None if no path exists within the depth limit.
    """

    def recursive_dls(current_node, path, depth):
        if depth == 0:
            return None

        if current_node == goal:
            return path

        for neighbor in graph[current_node]:
            result = recursive_dls(neighbor, path + [neighbor], depth - 1)
            if result:
                return result

        return None

    return recursive_dls(start, [start], max_depth)

# Example usage:
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F', 'G'},
    'D': {'B'},
    'E': {'B', 'H'},
    'F': {'C'},
    'G': {'C'},
    'H': {'E', 'I', 'J'},
    'I': {'H'},
    'J': {'H'}
}

start_node = 'A'
goal_node = 'E'
max_depth = 3

# Find the path using DLS
dls_path = dls(graph, start_node, goal_node, max_depth)

if dls_path:
    print(f"Path from {start_node} to {goal_node} (within depth {max_depth}): {dls_path}")
else:
    print(f"No path found from {start_node} to {goal_node} within depth {max_depth}")
