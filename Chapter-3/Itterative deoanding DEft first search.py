def dls(graph, start, goal, max_depth):

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


def iddfs(graph, start, goal, max_depth):

    for depth_limit in range(1, max_depth + 1):
        result = dls(graph, start, goal, depth_limit)
        if result:
            return result

    return None

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
goal_node = 'I'
max_depth = 5

# Find the path using IDDFS
iddfs_path = iddfs(graph, start_node, goal_node, max_depth)

if iddfs_path:
    print(f"Path from {start_node} to {goal_node} (within max depth {max_depth}): {iddfs_path}")
else:
    print(f"No path found from {start_node} to {goal_node} within max depth {max_depth}")