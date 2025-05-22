from typing import List, Tuple
from collections import defaultdict
import heapq

# Edge = Tuple[int, int, int]

class UnionFind():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def is_union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        self.parent[py] = px
        return True
    
# def kruskals_mst(n: int, edges: List[Edge]) -> Tuple[int, List[Edge]]:
#     uf = UnionFind(n)
#     mst_cost = 0
#     mst_edges = []
    
#     edges.sort()

#     for weight, u, v in edges:
#         if uf.is_union(u, v):
#             mst_edges.append([weight, u, v])
#             mst_cost += weight
#             if len(mst_edges) == n - 1:
#                 break

#     return mst_cost, mst_edges

# def prims_mst(n: int, edges: List[Edge]) -> Tuple[int, List[Edge]]:
#     mst_cost = 0
#     mst_edges = []
#     graph = defaultdict(list)

#     # Build adjency list
#     for cost, u, v in edges:
#         graph[u].append((cost, v))
#         graph[v].append((cost, u))

#     visited = set()
#     min_heap = [(0, 1)]

#     while min_heap and len(visited) < n:
#         cost, u = heapq.heappop(min_heap)
#         if u in visited:
#             continue

#         visited.add(u)
#         mst_cost += cost
#         mst_edges.append((cost, u, ))

#         for n_cost, v in graph[u]:
#             if v not in visited:
#                 heapq.heappush(min_heap, (n_cost, v))



#     return mst_cost, mst_edges
    


"""
Challenge: 

You are tasked with setting up a global network of trading servers. Each server can be connected to others through fiber optic links, and each link has a latency cost in milliseconds.

Your goal is to connect all servers with the minimum total latency.

- Input
n: int — the number of servers, labeled from 0 to n - 1

edges: List[Tuple[int, int, int]] — each tuple (latency, u, v) represents a possible link between server u and server v with the given latency of that connection

- Output
Return an int — the minimum total latency to connect all servers

If the servers cannot be connected, return -1

- Constraints
1 <= n <= 10^4

0 <= latency <= 10^4

len(edges) <= 10^5
""" 
Edge = Tuple[int, int, int]


def minimum_latency(n: int, edges: List[Edge]) -> int:
    uf = UnionFind(n)
    mst_cost = 0
    mst_edges = []
    
    edges.sort()

    for weight, u, v in edges:
        if uf.is_union(u, v):
            mst_edges.append([weight, u, v])
            mst_cost += weight
            if len(mst_edges) == n - 1:
                break

    return mst_cost


n = 4
edges = [
    (1, 0, 1),
    (4, 0, 2),
    (3, 1, 2),
    (2, 1, 3),
    (5, 2, 3)
]


assert minimum_latency(n, edges) == 6

"""
More test cases below
"""

