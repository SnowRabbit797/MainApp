import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, LpStatus, value

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])

pos = {
  1: (0, 0),
  2: (1, 0),
  3: (0, -1),
  4: (1, -1)
}

prob = LpProblem("Minimum_Vertex_Cover", LpMinimize)
x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in G.nodes}
prob += lpSum(x[v] for v in G.nodes)

for u, v in G.edges:
    prob += x[u] + x[v] >= 1

prob.solve()


cover_nodes = [v for v in G.nodes if x[v].varValue == 1]
print("最小頂点被覆:", cover_nodes)

color_map = ['lightgreen' if node in cover_nodes else 'lightgray' for node in G.nodes]
fig, ax = plt.subplots()
nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color='gray', node_size=800, font_size=14)

