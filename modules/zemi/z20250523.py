import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY
from pulp import LpMaximize, LpProblem, LpVariable, value




def main():

    st.markdown("""
        <style>
        .note-box {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 15px;
            padding: 20px;
            margin-top: 0;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .note-box hr {
          size: 1px;
          margin-top: 0;
        }
        .sline hr {
          size: 1px;
          margin-top: 0;
        }
    """, unsafe_allow_html=True)
    
    #----------------------------------------------------------
    st.title("5月23日(第3回)の発表")
    st.subheader("本日の発表内容")
    st.markdown("""
        <ol>
            <li>整数計画法を用いた最小頂点被覆問題(ノード:4, エッジ:5)の解法</li>
            <li></li>
            <li></li>
            <li></li>
        </ol>
    """, unsafe_allow_html=True)
    
    with st.container(border = True):
        st.subheader("最小頂点被覆問題", divider="orange")
        st.markdown(r"""与えられたグラフ
                    $$G=(V,E)$$について、$$G$$の頂点被覆のうち要素数が最小のものを求める問題
                    """)
    
    
    st.markdown("<br>", unsafe_allow_html=True)
    


    m = LpProblem(sense=LpMaximize)  # 数理モデル
    x = LpVariable("x", lowBound=0)  # 変数
    y = LpVariable("y", lowBound=0)  # 変数
    m += 100 * x + 100 * y  # 目的関数
    m += x + 2 * y <= 16  # 材料Aの上限の制約条件
    m += 3 * x + y <= 18  # 材料Bの上限の制約条件
    m.solve()  # ソルバーの実行
    st.write(value(x), value(y))  # 4.0 6.0

    with st.container(border = True):
        st.header("")
        with st.container(border = True):
            st.write("This is a second container")
#適当なグラフ
# G = nx.Graph()
# G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])

# pos = {
#   1: (0, 0),
#   2: (1, 0),
#   3: (0, -1),
#   4: (1, -1)
# }
# fig, ax = plt.subplots()
# nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=12, ax=ax)

# st.pyplot(fig)

#日本地図
# with st.container(border = True):
#     Array = np.loadtxt("assets/csv/admatrix.csv", delimiter=",")
#     G = nx.from_numpy_array(Array)
#     pos = kenchoXY.kenchoXY()
#     fig, ax = plt.subplots()
#     nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=30, font_size=3, ax=ax)
#     st.pyplot(fig)

# st.markdown("---")
