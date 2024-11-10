import heapq

# Define the graph with edges and heuristic values
graph = {
    'A': [('B', 4), ('C', 3)],
    'B': [('D', 11)],
    'C': [('E', 5)],
    'D': [('F', 9)],
    'E': [('F', 8)],
    'F': [('G', 2)],
    'G': []
}

heuristic = {
    'A': 10, 'B': 8, 'C': 5,
    'D': 7, 'E': 9, 'F': 0, 'G': 0
}

def a_star_search(start, goal):
    priority_queue = [(0, start, [])]
    visited = set()

    while priority_queue:
        cost, current, path = heapq.heappop(priority_queue)

        if current in visited:
            continue

        path = path + [current]
        visited.add(current)

        if current == goal:
            return path, cost

        for neighbor, weight in graph[current]:
            if neighbor not in visited:
                g = cost + weight
                f = g + heuristic[neighbor]
                heapq.heappush(priority_queue, (f, neighbor, path))

    return None, float('inf')

if __name__ == "__main__":
    path, total_cost = a_star_search('A', 'G')
    print(f"A* Search Path: {path}")
    print(f"Total Cost: {total_cost}")
