import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm.MVCprogram import intervalDivision
from modules.algorithm.MVCprogram import BFSanimation
from modules.algorithm.MVCprogram import randomAnimation
from modules.algorithm.MVCprogram import jisuuAnimation


def main():
    st.sidebar.title("11/7の資料")
    section = st.sidebar.radio("目次", ["区間値フィットネスの導入", "適応度の評価方法と超個体(super-child)", "サブグラフについて", "実験", "次回"])
    
    if section == "区間値フィットネスの導入":
        st.header("11月7日の発表(区間値フィットネスの導入)")
        st.markdown("""<br><br>""", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.subheader("区間値フィットネスとは", divider="orange")
            st.markdown("""
            - グラフをいくつかのサブグラフに分け、個体を部分ごとに評価する方法。
            - 全体としては評価が低い個体でも、部分的には良い解を持つ場合がある。
            - 各個体の部分的に良い解を組み合わせることで、より良い解(super-child)を生成する。
            
            """, unsafe_allow_html=True)
            
        st.markdown("""
            
            ---
            
            <h3 style='color:black;'>1. 現状の問題点</h3>
            
            #### 1.1 評価が全体依存
            
              - 現状のGAでは個体の良さを全体の被覆サイズでしか測れない</li>
              - 1000ノードあったとして、その内の10ノード違うだけでも「どの部分が良くて、どの部分が悪いか」が全くわからない</li>
              - 結果として、「どの局所構造が良い解に寄与しているか」がわからない</li>
            
            →つまり、部分的に良い解を次世代に活かしにくい。
            
            
            #### 1.2 大規模グラフでは探索空間が膨大
            
            - 例えば、500ノード・2000エッジのグラフでは、頂点の選択パターンは $2^{500}$ 通り。</li>
            - 遺伝的オペレータやBLSを使ってもスケールの大きいグラフでは限界がある</li>
            
            →「良い部分構造」を取りこぼすリスクが大きい。
            
            ---
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <h3 style='color:black;'>2.区間値フィットネスを導入したGAの流れ</h3>
            
            """, unsafe_allow_html=True)
        st.image("data/image/image1107/12B7FFDC-3E78-48B4-AF48-4EACD7C42BFB.jpeg")

    if section == "適応度の評価方法と超個体(super-child)":
        st.markdown("""
            ### 3.適応度の評価と超個体(super-child)の生成(今回のメイン)
            """, unsafe_allow_html=True)
        st.image("data/image/image1107/6909D39E-03AA-4DB2-8927-B359B49D71B5.jpeg")
        st.image("data/image/image1107/1B578822-FFCA-4542-AE55-8AC38DC9BDCE.jpeg")
        st.image("data/image/image1107/FA743515-1177-4DB7-A00D-6194A77B94FC.jpeg")
        st.image("data/image/image1107/47982B5B-B46A-46A0-8524-F8FE1B2CA673_1_105_c.jpeg")
        
        st.markdown("""
            次世代に受け継ぐのは、この超個体と、既存の遺伝的オペレータで生成した個体群で構成する
            例えば、
            - 超個体：1体
            - 交叉・突然変異で生成した個体：39体
            - ランダム生成個体：10体
            """, unsafe_allow_html=True)

    
    if section == "サブグラフについて":
        st.header("4.サブグラフの作成方法")
        st.subheader("4.1.基本方針", divider="orange")
        st.markdown("まずは方針を確認するために、論文に即したサブグラフ分割を行う。")
        st.markdown("分割数は以下の式で定義する：")
        st.latex(r"m = \max \left\{ 2, \ \mathrm{round}\!\left( \frac{|V|^{0.6}}{3} \right) \right\}")
        
        st.markdown("""
            頂点数が増加しても、分割数の増加は抑制されるいい感じの式になっている。
            """, unsafe_allow_html=True)
        st.markdown("""
            #### 4.1.1.分割数
            
            
            | 頂点数  | 分割数 m |
            | :--- | :-------: |
            | 10   |     2     |
            | 50   |     3     |
            | 100  |     5     |
            | 500  |     14     |
            | 1000 |    20   |
            | 3000 |    40   |
            | 5000 |    55   |
            
            ---
            
            """, unsafe_allow_html=True)

        st.markdown("""
            #### 4.1.2.分割方法
            
            区間値フィットネスでは、グラフ $G=(V, E)$ の頂点集合 $V$を $m$ 個の部分集合に分割する。
            
            $$
            P = \{P_1, P_2, \ldots, P_m \}
            $$
            実際どの頂点をどのサブグラフに入れるかが重要となる。分割方法についてはいくつかあるが、ざっくり3つに分類できる。
            
            
            | 方法    | 構造の反映度 | 計算の速さ | 向いている場面     |
            | :---- | :----: | :---: | :---------- |
            | ランダム  |   ★☆☆  |  ★★★  | 初期テスト・軽量実験  |
            | 次数ロビン |   ★★☆  |  ★★☆  | 中規模・安定性を重視  |
            | BFS分割 |   ★★★  |  ★☆☆  | 構造を反映させたい場合 |
            
            """, unsafe_allow_html=True)
        st.markdown("""
            #### ランダム分割
            - 頂点をランダムにサブグラフに割り当てる。
            - 実装が簡単で計算も速いが、グラフの構造を反映しにくい。
            """, unsafe_allow_html=True)
        with st.expander("ランダム分割の例", expanded=False):
            randomAnimation.main()
        
        st.markdown("""
            #### 次数ロビン分割
            - 頂点を次数の大きい順にソートし、順番にサブグラフに割り当てる。
            - グラフの構造をある程度反映しつつ、計算も比較的速い。
            """, unsafe_allow_html=True)
        with st.expander("次数ロビンの例", expanded=False):
            jisuuAnimation.main()
        
        st.markdown("""
            #### BFS分割
            - グラフの連結成分ごとにBFSを行い、近接する頂点を同じサブグラフに割り当てる。
            - グラフの構造をよく反映できるが、計算コストが高い。
            """, unsafe_allow_html=True)
        with st.expander("BFS分割の例", expanded=False):
            BFSanimation.main()
            
        st.markdown("""
            -次数ロビンをベースに、BFS分割も試してみる(次数ロビンBFSハイブリッド)→ハブ起点でよくなりそう
            
            """, unsafe_allow_html=True)
        
        
    if section == "実験":
        st.markdown("""
            ### 5.実験
            実験設定
            - グラフ：G-set1_small (400ノード、1768エッジ)
            - 世代数：1000
            - 個体数：50
            - 突然変異率：0.5%
            """, unsafe_allow_html=True)
        st.image("data/image/image1107/newplot (10).png")
        st.markdown("""
            ### 考察
            - 区間値フィットネスの導入により、局所的な探索の粘り強さは確認できたが、最終的な改善幅は小さかった（527 → 525）。
            - 改善は70世代以降も続いたものの、評価関数が全体ノード数依存のままであったため、区間値の効果が十分に反映されなかった。
            - GreedyCollectionにより局所的な冗長選択が発生し、被覆サイズが大きくなりやすい傾向が見られた。
            - 500世代・突然変異率0.01では多様性が不足し、収束が早期に進む傾向が見られた。

            ### 改善点（次回の方針）
            - まずはコードを直す。
            - 現在は固定サブグラフ（初期集団生成時にサブグラフを固定）しているが、一定世代ごとに再分割を行う可変サブグラフを導入する。
            - ランダム分割または次数ロビン分割を「強い摂動」として導入し、多様性を確保する。
            - 補正処理（GreedyCollection）を改良し、利得ペナルティや1-hop削減を導入して冗長なノード選択を抑制する。
            
            | 世代 | 区間値なし 評価値 | 区間値あり 評価値 |
            | :- | :-------- | :-------- |
            | 1  | 552       | 552       |
            | 2  | 536       | 536       |
            | 3  | 535       | —         |
            | 5  | 533       | —         |
            | 6  | —         | 534       |
            | 7  | —         | 533       |
            | 8  | —         | 532       |
            | 9  | 531       | 531       |
            | 21 | —         | 530       |
            | 22 | —         | 528       |
            | 46 | 529       | —         |
            | 58 | —         | 527       |
            | 62 | 528       | —         |
            | 70 | 527       | —         |
            | 88 | —         | 526       |
            | 95 | —         | 525       |

            """, unsafe_allow_html=True)

