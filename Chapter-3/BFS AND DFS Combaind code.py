from collections import deque

def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [])])

    while queue:
        current_node, path = queue.popleft()

        if current_node == goal:
            return path + [current_node]  # Goal node found

        if current_node not in visited:
            visited.add(current_node)
            queue.extend((neighbor, path + [current_node]) for neighbor in set(graph[current_node]) - visited)

    return []  # Goal node not found

def dfs(graph, start, goal):
    visited = set()
    stack = [(start, [])]

    while stack:
        current_node, path = stack.pop()

        if current_node == goal:
            return path + [current_node]  # Goal node found

        if current_node not in visited:
            visited.add(current_node)
            stack.extend((neighbor, path + [current_node]) for neighbor in set(graph[current_node]) - visited)

    return []  # Goal node not found

# Define the graph
graph = {
    'A': {'B', 'C'},
    'B': {'D', 'E'},
    'C': {},
    'D': {'F'},
    'E': {'H', 'G'},
    'F': {},
    'G': {'I', 'J'},
    'H': {'K'},
    'I': {'L'},
    'J': {'L'},
    'K': {},
    'L': {}
}

# Define the start and goal nodes
start_node = 'A'
goal_node = 'L'

# Run BFS
bfs_path = bfs(graph, start_node, goal_node)
print(f"BFS Result: {'Goal found' if bfs_path else 'Goal not found'}")
print(f"Traversal Path: {bfs_path}")

# Run DFS
dfs_path = dfs(graph, start_node, goal_node)
print(f"DFS Result: {'Goal found' if dfs_path else 'Goal not found'}")
print(f"Traversal Path: {dfs_path}")