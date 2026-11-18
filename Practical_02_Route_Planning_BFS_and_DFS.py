# Practical 2: Route Planning using BFS and DFS
from collections import deque

# Graph representation as adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}


# Breadth-First Search (Queue - FIFO)
def bfs(start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


# Depth-First Search (Stack - LIFO)
def dfs(start, goal):
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node not in visited:
            visited.add(node)
            if node == goal:
                return path
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(path + [neighbor])
    return None


if __name__ == "__main__":
    print("BFS Shortest Path (A to F):", bfs('A', 'F'))
    print("DFS Path          (A to F):", dfs('A', 'F'))
