import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.sidebar.title("10/17の資料")
    section = st.sidebar.radio("目次", ["学術の資料に載せた実験に関して", "次回の展望"])

    if section == "学術の資料に載せた実験に関して":
        st.header("10/17(後期第1回)の発表", divider="blue")
        st.markdown("""<br>""", unsafe_allow_html=True)
        st.image("data/image/image1017/スライド1.jpeg")
        st.image("data/image/image1017/スライド2.jpeg")
        st.image("data/image/image1017/スライド3.jpeg")
        st.image("data/image/image1017/スライド4.jpeg")
        st.image("data/image/image1017/スライド5.jpeg")
        st.image("data/image/image1017/スライド6.jpeg")
        st.image("data/image/image1017/スライド7.jpeg")
        st.image("data/image/image1017/スライド8.jpeg")
        st.image("data/image/image1017/スライド9.jpeg")
        st.image("data/image/image1017/スライド10.jpeg")
        st.image("data/image/image1017/スライド11.jpeg")
        st.image("data/image/image1017/スライド12.jpeg")
        st.image("data/image/image1017/スライド13.jpeg")
        st.image("data/image/image1017/スライド14.jpeg")
        st.image("data/image/image1017/スライド15.jpeg")
        st.image("data/image/image1017/スライド16.jpeg")
        
            
    
    elif section == "次回の展望":
        st.header("次回の展望", divider="blue")
        st.markdown("""予稿には以下のように記載した。<br>
                    <ul>
                      <li>提案手法の汎用性を検証するため，巡回セールスマン問題（TravelingSalesman Problem：TSP）など他の代表的な組合せ最適化問題への適用を行い，その有効性を評価する．</li>
                      <li>強い摂動戦略を焼きなまし法や粒子群最適化など他のメタヒューリスティクスにも導入し，GA との性能比較を行う．</li>
                      <li>摂動の強度や適用対象を適応的に調整する手法の導入や，個体単位での局所的な摂動など，より柔軟な制御戦略の設計についても検討を進める．</li>
                    </ul><br>
                    これらを踏まえて後期に取り組むことを考える。<br>
                    
                    """, unsafe_allow_html=True)
        st.subheader("修論に向けて", divider="orange")
        st.markdown("""
        ## 修論に向けて

        ### 修論の目標
        本研究の最終的な目標は、**様々な組合せ最適化問題に対して、Breakout Local Search（BLS）を適用したメタヒューリスティクス手法の有効性を体系的にまとめること**である。  
        具体的には、遺伝的アルゴリズム（GA）を中心に、BLSにおける「強い摂動」の概念を導入した各種手法を比較・検証し、その効果を定量的に評価することを目指す。

        ---

        ### 後期に取り組む内容
        - **巡回セールスマン問題（TSP）**をはじめとする、他の代表的なNP困難問題への適用を試みる。  
        - MVC（最小頂点被覆問題）に関しては、区間値フィットネス（interval-valued fitness）を導入し、評価の安定性と収束特性について追加実験を行う。  
        - これらの結果を通じて、**問題特性に応じたBLS戦略の有効性**を整理し、最終的に修士論文としてまとめる。

        ---

        ### 今後の展望
        """)

