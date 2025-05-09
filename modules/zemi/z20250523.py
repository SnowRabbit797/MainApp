import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.sidebar.title("25/05/23 ゼミ発表資料")
    section = st.sidebar.radio("目次", ["section1", "section2", "section3", "section4", "section5"])

    if section == "section1":
        st.title("2025年5月23日 M2ゼミ発表(3回目)")
    elif section == "section2":
        st.title("(仮)グラフの作成")
        
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])

        pos = {
          1: (0, 0),
          2: (1, 0),
          3: (0, -1),
          4: (1, -1)
        }
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=12, ax=ax)
        
        st.pyplot(fig)
