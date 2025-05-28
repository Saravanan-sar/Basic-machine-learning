
from queue import PriorityQueue

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 6)],
    'C': [('F', 5)],
    'D': [],
    'E': [('G', 2)],
    'F': [('G', 2)],
    'G': []
}

heuristic = {'A': 10, 'B': 8, 'C': 5, 'D': 7, 'E': 3, 'F': 6, 'G': 0}

def a_star(start, goal):
    pq = PriorityQueue()
    pq.put((0 + heuristic[start], 0, start, [start]))

    while not pq.empty():
        f, cost, node, path = pq.get()
        if node == goal:
            print("Path found:", path)
            return
        for neighbor, weight in graph[node]:
            g = cost + weight
            h = heuristic[neighbor]
            pq.put((g + h, g, neighbor, path + [neighbor]))

a_star('A', 'G')
