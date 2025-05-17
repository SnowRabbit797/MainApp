import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY

def main():

    st.markdown("""
        <style>
        .note-box {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    """, unsafe_allow_html=True)
    
    #----------------------------------------------------------
    st.title("5月23日(第3回)の発表")
    
    st.markdown("""
        <div class="note-box">
            <h2>タイトル</h2>
            <hr>
            <p>ここに内容を書きます。説明文や要点を記述できます。</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border = True):
        Array = np.loadtxt("assets/csv/admatrix.csv", delimiter=",")
        G = nx.from_numpy_array(Array)
        pos = kenchoXY.kenchoXY()
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=30, font_size=3, ax=ax)
        st.pyplot(fig)
    
    
        
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
