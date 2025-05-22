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
    st.markdown("""<br>""", unsafe_allow_html=True)
    with st.container(border = True):
        st.subheader("本日の発表内容", divider="red")
        st.markdown("""
            <ol>
                <li>前回の復習</li>
                <li></li>
                <li></li>
                <li></li>
            </ol>
        """, unsafe_allow_html=True)
    #----------------------------------------------------------
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>1. 前回の復習</h2>", unsafe_allow_html=True)
    
    with st.container(border = True):
        with st.container(border = True):
            st.subheader("最小頂点被覆問題", divider="orange")
            st.markdown(r"""与えられたグラフ$$G=(V,E)$$について、$$G$$の頂点被覆のうち要素数が最小のものを求める問題
                        """, unsafe_allow_html=True)
        
        st.markdown(r"""前回は、貪欲法を用いて以下の手順で最小頂点被覆問題を解いた。""")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <ol>
                    <li>グラフのノードを全て選択肢に入れる。</li>
                    <li>選択肢の中から、最も多くのエッジをカバーするノードを選ぶ。</li>
                    <li>選んだノードをカバーセットに追加し、選択肢から削除する。</li>
                    <li>選択肢の中から、選んだノードに隣接するノードを全て削除する。</li>
                    <li>選択肢が空になるまで、2~4を繰り返す。</li>
                    <li>カバーセットを返す。</li>
                </ol>
            """, unsafe_allow_html=True)
        
        with col2:
            st.image("data/image/image0425/250425_page-0009.jpg")
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("""<u>今回は、<b>整数計画法</b>と<b>線形計画法</b>を用いて最小頂点被覆問題を解いてみようと思う。</u>""", unsafe_allow_html=True)
    
    #----------------------------------------------------------
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2>2. 整数計画法と線形計画法による最小頂点被覆問題の解法</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border = True):
            st.subheader("線形計画法", divider="orange")
            st.markdown(r"""さまざまな制約条件のもとで目的関数を最適化(最大化や最小化)する数理計画法のうち、制約条件が線形不等式・目的関数が線形関数・変数が<span style="color:red;"><b>正の実数値</b></span>を取る問題のこと。""", unsafe_allow_html=True)
    with col2:
        with st.container(border = True):
            st.subheader("整数計画法", divider="orange")
            st.markdown(r"""さまざまな制約条件のもとで目的関数を最適化(最大化や最小化)する数理計画法のうち、制約条件が線形不等式・目的関数が線形関数・変数が<span style="color:red;"><b>正の整数値</b></span>を取る問題のこと。""", unsafe_allow_html=True)
            
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("ここで、線形計画法について学部時代にやったものを例にしてやってみる。", unsafe_allow_html=True)
    st.markdown("""<br>""", unsafe_allow_html=True)
    
    with st.container(border = True):
        st.subheader("整数計画法", divider="orange")
        with st.expander("学部生時代に扱った簡単な問題。"):
            tab1, tab2 = st.tabs(["問題", "解答"])
            
            with tab1:
                st.image("data/image/image0523/250523-01.jpg")
            with tab2:
                st.image("data/image/image0523/250523-02.jpg")



#----------------------------------------------------------


# #pulp
# m = LpProblem(sense=LpMaximize)  # 数理モデル
# x = LpVariable("x", lowBound=0)  # 変数
# y = LpVariable("y", lowBound=0)  # 変数
# m += 100 * x + 100 * y  # 目的関数
# m += x + 2 * y <= 16  # 材料Aの上限の制約条件
# m += 3 * x + y <= 18  # 材料Bの上限の制約条件
# m.solve()  # ソルバーの実行
# st.write(value(x), value(y))  # 4.0 6.0
            
            
            
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
