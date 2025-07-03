import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go


file_path = "assets/csv/G_set1.csv"

df = pd.read_csv(file_path, skiprows=3)


G = nx.from_pandas_edgelist(df, source="from", target="to", edge_attr="weight")
pos = nx.spring_layout(G, k=3, seed=7)

x_nodes = [pos[i][0] for i in G.nodes()]
y_nodes = [pos[i][1] for i in G.nodes()]


edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig = go.Figure()

fig.add_trace(go.Scatter(
  x=edge_x, y=edge_y,
  line=dict(width=0.2, color="black")))

fig.add_trace(go.Scatter(
  x=x_nodes, y=y_nodes,
  marker=dict(size=5, color='blue'),
  mode="markers"
))

st.plotly_chart(fig)


# # エッジ描画
# fig.add_trace(go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines'))

# # ノード描画
# fig.add_trace(go.Scatter(
#     x=x_nodes, y=y_nodes,
#     mode='markers',
#     marker=dict(size=5, color='blue'),
#     hoverinfo='text',
#     text=[str(i) for i in G.nodes()]))

# fig.update_layout(
#     showlegend=False,
#     margin=dict(l=0, r=0, b=0, t=0),
#     xaxis=dict(showgrid=False, zeroline=False),
#     yaxis=dict(showgrid=False, zeroline=False)
# )

# # Streamlit で表示
# st.plotly_chart(fig, use_container_width=True)
