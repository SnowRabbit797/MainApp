import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary
import time

col1, col2 = st.columns(2)

with col1:
    n_nodes = st.slider("ノード数", min_value=5, max_value=300, value=15, step=1)
    edge_prob = st.slider("エッジの密度（確率）", min_value=0.20, max_value=1.0, value=0.3, step=0.05)

    if st.button("グラフを生成"):
        G = nx.gnp_random_graph(n=n_nodes, p=edge_prob, seed=None)

        x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in G.nodes}
        prob = LpProblem("Minimum_Vertex_Cover", LpMinimize)
        prob += lpSum(x[v] for v in G.nodes)
        for u, v in G.edges:
            prob += x[u] + x[v] >= 1

        start_time = time.time()
        prob.solve()
        end_time = time.time()
        wallclock_seconds = round(end_time - start_time, 4)

        cover_nodes = [v for v in G.nodes if x[v].varValue == 1]

        # --- 可視化 ---
        k = 1.5 / (n_nodes ** 0.5)
        pos = nx.spring_layout(G, seed=42, k=k)
        color_map = ['yellow' if v in cover_nodes else 'lightgray' for v in G.nodes]
        node_size = int(8000 / n_nodes)
        font_size = max(6, int(200 / n_nodes))

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color='gray',
                node_size=node_size, font_size=font_size, ax=ax)

        st.markdown(f"- ノード数: **{n_nodes}**")
        st.markdown(f"- エッジ数: **{G.number_of_edges()}**")
        st.markdown(f"- 最小頂点被覆サイズ: **{len(cover_nodes)}**")
        st.markdown(f"- Wallclock seconds: **{wallclock_seconds} 秒**")

        with col2:
            st.pyplot(fig)
