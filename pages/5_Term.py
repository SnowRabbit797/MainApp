import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

st.title(("今までに出てきた用語一覧"))

with st.container(border = True):
    
    col1, col2 = st.columns(2, border=True)
    
    with col1:
        st.subheader("ノード", divider="orange")
        st.markdown(r"""頂点のこと。グラフ$$G=(V,E)$$においては$$V(vertices)$$の部分(下の図においての赤い点)。
                    """)
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])
        
        pos = {
          1: (0, 0),
          2: (1, 0),
          3: (0, -1),
          4: (1, -1)
        }
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="red", edge_color="lightblue", node_size=700, font_size=15, ax=ax)

        st.pyplot(fig)
    with col2:
        st.subheader("エッジ", divider="orange")
        st.markdown(r"""辺のこと。グラフ$$G=(V,E)$$においては$$E(edges)$$の部分(下の図においての赤い線)。
                    """)
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])
        
        pos = {
          1: (0, 0),
          2: (1, 0),
          3: (0, -1),
          4: (1, -1)
        }
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="red", node_size=700, font_size=15, ax=ax)

        st.pyplot(fig)
    

with st.container(border = True):
    st.subheader("頂点被覆", divider="orange")
    st.markdown(r"""与えられたグラフ
                $$G=(V,E)$$の頂点の部分集合を$$C$$とする。Eの全ての枝がCのいずれかの頂点と接続している(被覆している)とき、$$C$$を$$G$$の頂点被覆という。
                """)
    st.markdown(r"""下記のグラフは与えられたグラフにおいて、選ばれたノード(赤)によって被覆されているエッジ(赤)を示している。""")
    col1, col2 = st.columns(2, border=True)
    with col1:
        st.write("頂点被覆である例")
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])
        
        cover_nodes = [1, 4]
        
        covered_edges = set()
        for node in cover_nodes:
            for edge in G.edges(node):
                covered_edges.add(tuple(sorted(edge)))
                
        node_colors = []
        for node in G.nodes():
            if node in cover_nodes:
                node_colors.append("red")
            else:
                node_colors.append("lightblue")

        edge_colors = []
        for edge in G.edges():
            if tuple(sorted(edge)) in covered_edges:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")

        pos = {
          1: (0, 0),
          2: (1, 0),
          3: (0, -1),
          4: (1, -1)
        }
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=700, font_size=15, ax=ax)

        st.pyplot(fig)

    with col2:
        st.write("頂点被覆でない例")
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (3, 4), (2, 4), (1, 4)])
        
        cover_nodes = [1, 3]
        
        covered_edges = set()
        for node in cover_nodes:
            for edge in G.edges(node):
                covered_edges.add(tuple(sorted(edge)))
                
        node_colors = []
        for node in G.nodes():
            if node in cover_nodes:
                node_colors.append("red")
            else:
                node_colors.append("lightblue")

        edge_colors = []
        for edge in G.edges():
            if tuple(sorted(edge)) in covered_edges:
                edge_colors.append("red")
            else:
                edge_colors.append("gray")

        pos = {
          1: (0, 0),
          2: (1, 0),
          3: (0, -1),
          4: (1, -1)
        }
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=700, font_size=15, ax=ax)
        st.pyplot(fig)


with st.container(border = True):
    st.subheader("最小頂点被覆問題", divider="orange")
    st.markdown(r"""与えられたグラフ
                $$G=(V,E)$$について、$$G$$の頂点被覆のうち要素数が最小のものを求める問題
                """)
