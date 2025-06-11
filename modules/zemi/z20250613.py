import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY
from pulp import LpMaximize, LpProblem, LpVariable, value, LpMinimize, LpBinary, lpSum, LpStatus


def main():
    st.title("6月13日(第4回)の発表")
    st.markdown("""<br>""", unsafe_allow_html=True)
    with st.container(border = True):
        st.subheader("本日の発表内容", divider="red")
        st.markdown("""
            <ol>
                <li>進化計算アルゴリズムとは</li>
                <li></li>
                <li>最小頂点被覆問題を整数計画法で解く</li>
                <li>最小頂点被覆問題を整数計画法で解く②</li>
            </ol>
        """, unsafe_allow_html=True)
    # st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.container(border=False):
        st.markdown("""## 1.進化計算アルゴリズムとは""", unsafe_allow_html=True)
        st.markdown("""<br>""", unsafe_allow_html=True)
        with st.container(border = True):
            st.subheader("進化計算アルゴリズム", divider="orange")
            st.markdown("""地球上の生物は、それぞれの生息域に特有の環境や時間と共に変化する環境に適用するように進化を続けている。
                        各生物が「動的な環境に適応する」という困難な課題を解決すべく世代交代を繰り返し、解として現在の生態系を得てきた。<br><br>
                        このような生物の進化過程にヒントを得た最適解探索アルゴリズムが<span style="color:red;"><b>進化計算アルゴリズム</b></span>である。
                        """, unsafe_allow_html=True)
        st.markdown("""以下は進化計算アルゴリズムの例である""", unsafe_allow_html=True)
        st.markdown("""
        #### 遺伝的アルゴリズム（Genetic Algorithm, GA）
        - 自然界の「進化（淘汰・交叉・突然変異）」を模倣
        - 複数の個体（解候補）を世代ごとに進化させる
        - 最適化全般に広く使われる汎用的手法
        ---
        #### 粒子群最適化（Particle Swarm Optimization, PSO）
        - 鳥や魚の群れのように、粒子（個体）が移動しながら探索
        - 各粒子が「自身の最良解」と「全体の最良解」を参考に移動
        - パラメータ最適化など連続空間で特に強力
        ---
        #### アントコロニー最適化（Ant Colony Optimization, ACO）
        - アリのフェロモンによる経路探索行動をモデル化
        - フェロモンの強化・蒸発により、良い経路が強調される
        - 組合せ最適化（例：巡回セールスマン問題）に適している
        ---
        #### 人工蜂コロニー（Artificial Bee Colony, ABC）
        - ミツバチの採餌行動を模倣したアルゴリズム
        - 働きバチ、見張りバチ、偵察バチの役割分担で探索と活用を行う
        - バランスの取れた探索性能を持ち、数値最適化にも有効
        ---
        #### ホタルのアルゴリズム（Firefly Algorithm）
        - ホタルの光の強さに引き寄せられる性質を利用
        - 明るい個体（良い解）に他の個体が引き寄せられて移動
        - 局所最適からの脱出がしやすく、探索の多様性も高い
        ---
        """)
        st.markdown("""M1ではGA, PSO, ACOについて主に巡回セールスマン問題へのアプローチ法を考えてきた。<br>
                    今回(6/13)は最小頂点被覆問題に対する進化計算アルゴリズムの適用を考える。
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""""", unsafe_allow_html=True)
        
    with st.container(border=True):
        st.subheader("遺伝的アルゴリズム(GA)", divider="orange")
        st.markdown("""
                    <span style="color:red;"><b>遺伝的アルゴリズム</b></span>とは、生物の進化メカニズムを模倣した、特に以下の点に着目した最適化アルゴリズムである。<br>
                    <ul>
                        <li>環境に適応できる個体ほど次世代に自分の遺伝子を残せる</li>
                        <li>2個体の交叉により子を作る</li>
                        <li>ときどき突然変異が起こる</li>
                    </ul>
                    """, unsafe_allow_html=True)      
        st.markdown("""詳しくは[学部時代の資料](https://drive.google.com/file/d/1QZ8071fYYuhvX_yynp91OgrcQG5n4YwG/view?usp=drive_link)を参照してください。<br>""", unsafe_allow_html=True)
        
    st.markdown("""<h4>遺伝的アルゴリズムの実装の流れ</h4>""", unsafe_allow_html=True)
    
