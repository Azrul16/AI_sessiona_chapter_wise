def is_safe(graph, node, color, assignment):
    # Check if it's safe to color the node with the given color
    for neighbor in graph[node]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

def color_map_util(graph, nodes, m, assignment):
    # Base case: If all nodes are assigned a color, return True
    if len(assignment) == len(nodes):
        return True

    # Try coloring the current node with each color
    for color in ['Red', 'Green', 'Blue']:
        if is_safe(graph, nodes[len(assignment)], color, assignment):
            assignment[nodes[len(assignment)]] = color

            # Recur to color the next node
            if color_map_util(graph, nodes, m, assignment):
                return True

            # If coloring the current node doesn't lead to a solution, backtrack
            assignment.pop(nodes[len(assignment)])

    # If no color is found for the current node, return False
    return False

def color_map(graph, m):
    # Get the list of nodes in the graph
    nodes = list(graph.keys())

    # Initialize an empty assignment
    assignment = {}

    # Call the utility function to color the map
    if not color_map_util(graph, nodes, m, assignment):
        print("Solution does not exist.")
        return

    # Print the solution
    print_solution(assignment)

def print_solution(assignment):
    # Print the colored map
    for node, color in assignment.items():
        print(f"{node}: {color}")

# Example: Color the map with 3 colors (Red, Green, Blue)
# Define the graph as an adjacency list
graph = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q': ['NT', 'SA', 'NSW'],
    'NSW': ['Q', 'SA', 'V'],
    'V': ['SA', 'NSW']
}

color_map(graph, 3)
