import heapq

class AStar:
    def __init__(self, graph, heur):
        self.graph = graph
        self.heur = heur

    def find_path(self, start, goal):
        openset = []
        heapq.heappush(openset, (self.heur[start], start))
        camefrom = dict()
        cost = {start: 0}

        while openset:
            current_priority, current = heapq.heappop(openset)

            if current == goal:
                path = [current]
                while current in camefrom:
                    current = camefrom[current]
                    path.append(current)
                return path[::-1]

            for neighbor, weight in self.graph.get(current, []):
                newcost = cost[current] + weight
                if neighbor not in cost or newcost < cost[neighbor]:
                    cost[neighbor] = newcost
                    priority = newcost + self.heur.get(neighbor, 0)
                    heapq.heappush(openset, (priority, neighbor))
                    camefrom[neighbor] = current

        return None

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []
}

heur = {
    'A': 3,
    'B': 2,
    'C': 1,
    'D': 0
}

astar_solver = AStar(graph, heur)
path = astar_solver.find_path('A', 'D')
print("Path:", path)
