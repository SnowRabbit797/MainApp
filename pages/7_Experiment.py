import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY
from pulp import LpMaximize, LpProblem, LpVariable, value, LpMinimize, LpBinary, lpSum, LpStatus

st.markdown("<h2>5. 最小頂点被覆問題を整数計画法で解く③</h2>", unsafe_allow_html=True)
st.markdown("ノード数とエッジの密度を選択しグラフ生成を押すと、最小頂点被覆サイズとWallclock secondsを出力し、グラフを描画します。", unsafe_allow_html=True)

n_nodes = 5
n_nodes_start = n_nodes
n_nodes_end = 100
edge_prob = 0.5
wcs_list = []

if st.button("グラフを生成"):
    while n_nodes <= n_nodes_end:
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
        wcs_list.append(wallclock_seconds)

        cover_nodes = [v for v in G.nodes if x[v].varValue == 1]
        
        st.write(f"ノード数: {n_nodes}, 最小頂点被覆サイズ: {len(cover_nodes)}, Wallclock seconds: {wallclock_seconds}")
        n_nodes += 1
    
    st.line_chart(pd.Series(wcs_list, index=range(n_nodes_start , n_nodes_end+1)), use_container_width=True)
