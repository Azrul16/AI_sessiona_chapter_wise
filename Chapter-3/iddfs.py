def iterative_deepening_dfs(graph, start, goal):
    def dls(node, depth, limit):
        if depth > limit:
            return False
        if node == goal:
            return True
        for neighbor in graph[node]:
            if dls(neighbor, depth + 1, limit):
                return True
        return False

    depth = 0
    while True:
        if dls(start, 0, depth):
            return True
        depth += 1

# Main function
if __name__ == '__main__':
    # Create the graph as an adjacency list
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }

    # Define start and goal
    start_node = 'A'
    goal_node = 'F'

    # Perform Iterative Deepening DFS
    found = iterative_deepening_dfs(graph, start_node, goal_node)
    if found:
        print(f"Goal node '{goal_node}' found.")
    else:
        print(f"Goal node '{goal_node}' not found.")
