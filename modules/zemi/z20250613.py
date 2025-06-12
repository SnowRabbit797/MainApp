import streamlit as st
st.set_page_config(page_title="ゼミの資料アプリ", layout="wide")
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
                <li>GAによるMVC問題への近似的アプローチ</li>
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
    st.markdown("""## 2.GAによるMVC問題への近似的アプローチ""", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("遺伝的アルゴリズム(GA)", divider="orange")
        
        st.markdown("""
        <span style="color:red;"><b>遺伝的アルゴリズム</b></span>とは、生物の進化メカニズムを模倣した、特に以下の点に着目した最適化アルゴリズムである。<br>
        <ul>
            <li>環境に適応できる個体ほど次世代に自分の遺伝子を残せる（<b>選択</b>）</li>
            <li>2個体の交叉により子を作る（<b>交叉</b>）</li>
            <li>ときどき特徴の違う個体が生まれる（<b>突然変異</b>）</li>
        </ul>
        """, unsafe_allow_html=True)
        
    st.markdown("""<h4>実装の流れ</h4>""", unsafe_allow_html=True)

    st.markdown("""
    1. **初期集団の生成**  
      ランダムに個体（解候補）を生成する。

    2. **適応度の評価**  
      各個体にスコアを与え、どれだけ「良い」かを評価する。

    3. **選択（Selection）**  
      良い個体ほど次世代に残りやすくなるように選ぶ。  
      - 例：ルーレット選択、トーナメント選択

    4. **交叉（Crossover）**  
      親個体の一部を組み合わせて新しい個体を生成する。  
      - 例：一点交叉、一様交叉

    5. **突然変異（Mutation）**  
      個体の遺伝子を一部ランダムに変化させ、多様性を保つ。  
      - 例：ビット反転、スワップ

    6. **次世代の更新**  
      新しく生まれた個体で次世代を構成する。必要に応じてエリート保存も行う。

    7. **終了条件の判定**  
      規定世代に達した、または解が十分良くなったら終了。
    """)
        
    st.markdown("""詳しくは[学部時代の資料①](https://drive.google.com/file/d/1QZ8071fYYuhvX_yynp91OgrcQG5n4YwG/view?usp=drive_link), 
                [学部時代の資料②](https://drive.google.com/file/d/1aCukTX3wn9Wrcr--U7mSUDIxUrfRV1eo/view?usp=drive_link)を参照してください。<br>""", unsafe_allow_html=True)
    st.markdown("""<br>""", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("選択(ルーレット選択)", divider="orange")
        st.markdown("""<h4>ルーレット選択</h4>""", unsafe_allow_html=True)
        st.markdown("""
        個体の適応度に比例して選択される確率を決定する方法。適応度が高いほど選ばれやすくなる。
        """)
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.write("学部時代の資料(ルーレット選択)")
            st.image("data/image/image0613/250612_01.jpg")
            st.image("data/image/image0613/250612_02.jpg")
        with col2:
            st.markdown("コード例")
            st.code("""
                    import random

                    population = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                    fitness = [80, 70, 60, 50, 40, 30, 20, 10]

                    def roulette_selection(population, fitness):
                        total_fitness = sum(fitness)
                        probs = []
                        r_sum = 0
                        for f in fitness:
                            r_sum += f
                            probs.append(r_sum / total_fitness)
                        r = random.random()
                        for i, prob in enumerate(probs):
                            if r <= prob:
                                return population[i]

                    for _ in range(10):
                        print(roulette_selection(population, fitness)))
            """)
            st.write("出力例")
            st.code("""
                    C
                    G
                    A
                    A
                    H
                    B
                    B
                    F
                    C
                    F
                    """)
        st.markdown("""<br>""", unsafe_allow_html=True)
        st.subheader("選択(エリート選択)", divider="orange")
        st.markdown("1番適応度が良い個体を無条件で次世代に残す方法")
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("交叉(一点交叉)", divider="orange")
        st.markdown("""<h4>一点交叉</h4>""", unsafe_allow_html=True)
        st.markdown("""
        2つの親個体からランダムに選んだ一点で遺伝子を交換し、新しい子個体を生成する方法。
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.image("data/image/image0613/250612_01.jpg")
            st.image("data/image/image0613/250612_02.jpg")
        with col2:
            st.markdown("コード例")
            st.code("""
                    import random

                    def one_point_crossover(parent1, parent2):
                        point = random.randint(1, len(parent1) - 1)
                        child1 = parent1[:point] + parent2[point:]
                        child2 = parent2[:point] + parent1[point:]
                        return child1, child2

                    parent1 = "110010"
                    parent2 = "101101"
                    child1, child2 = one_point_crossover(parent1, parent2)
                    print(child1, child2)
            """)
