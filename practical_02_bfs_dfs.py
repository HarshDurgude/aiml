# Practical 2: Route Planning using BFS (Breadth-First Search) and DFS (Depth-First Search)
# AIML Laboratory

from collections import deque
import time

# Sample route map (Graph) as an adjacency list
route_graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}


def bfs_route(graph, start, goal):
    """Breadth-First Search: Explores shallowest nodes first (guarantees shortest path in unweighted graph)."""
    start_time = time.time()
    queue = deque([[start]])
    visited = {start}
    nodes_explored = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_explored += 1

        if node == goal:
            return path, nodes_explored, (time.time() - start_time)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None, nodes_explored, (time.time() - start_time)


def dfs_route(graph, start, goal):
    """Depth-First Search: Explores deepest nodes first using stack / LIFO."""
    start_time = time.time()
    stack = [[start]]
    visited = set()
    nodes_explored = 0

    while stack:
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited.add(node)
            nodes_explored += 1

            if node == goal:
                return path, nodes_explored, (time.time() - start_time)

            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)

    return None, nodes_explored, (time.time() - start_time)


if __name__ == "__main__":
    start_node = 'A'
    goal_node = 'F'
    print("=" * 60)
    print(f"Route Planning from '{start_node}' to '{goal_node}'")
    print("=" * 60)

    bfs_path, bfs_explored, bfs_time = bfs_route(route_graph, start_node, goal_node)
    print(f"BFS Result:")
    print(f"  - Optimal Path     : {' -> '.join(bfs_path) if bfs_path else 'No path'}")
    print(f"  - Nodes Explored   : {bfs_explored}")
    print(f"  - Execution Time   : {bfs_time:.6f} seconds")

    print("-" * 60)

    dfs_path, dfs_explored, dfs_time = dfs_route(route_graph, start_node, goal_node)
    print(f"DFS Result:")
    print(f"  - Path Found       : {' -> '.join(dfs_path) if dfs_path else 'No path'}")
    print(f"  - Nodes Explored   : {dfs_explored}")
    print(f"  - Execution Time   : {dfs_time:.6f} seconds")
    print("=" * 60)
