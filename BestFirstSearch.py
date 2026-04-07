import heapq

def best_first_search(graph, start, goal, heur):
    openlist = []
    heapq.heappush(openlist, (heur[start], start))
    
    came_from = {}
    visited = set()

    while openlist:
        _, curr = heapq.heappop(openlist)

        if curr == goal:
            path = [curr]
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            return path[::-1]

        if curr in visited:
            continue
        visited.add(curr)

        for neigh, _ in graph.get(curr, []):
            if neigh not in visited:
                came_from[neigh] = curr
                heapq.heappush(openlist, (heur[neigh], neigh))

    return None

def main():
    graph = {
    'A': [('B', 1), ('C', 4)],
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

    path = best_first_search(graph,'A', 'D', heur)
    if path:
        print("Path found:", " -> ".join(path))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
