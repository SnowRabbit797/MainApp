import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.sidebar.title("07/11の資料一覧")
    section = st.sidebar.radio("目次", ["新しいGAのプログラム", "GreedyCollection", "GC付きの実験結果①", "補正処理の高速化", "全0ビット列の導入とその影響", "GreedyReduction"])
  
    st.title("7月11日(第5回)の発表")
    st.markdown("""<br>""", unsafe_allow_html=True)

    if section == "新しいGAのプログラム":
        with st.container(border=False):
            st.subheader("1. 今回のコードについて", divider="orange")
            st.markdown("""
            今回の実験では、従来のコードを再利用せず、**完全に一から新しい実装**を行った。<br><br>
            この方針にした理由は以下のとおりである：
            
            - 処理フローの本質的理解を深めるため
            - 古いコードに不要な依存やバグがあったため
            - よりシンプルで拡張性の高い設計を目指した
            """, unsafe_allow_html=True)
        
        st.markdown("""ということで、まずはコードがうまく動いているか調べるため、以下のような問題を作成した。<br><br><br>""", unsafe_allow_html=True)
            
        with st.container(border=True):
            st.subheader("1. 簡素なビット列最適化問題", divider="orange")

            st.markdown("""
            最小頂点被覆問題（MVC）に進む前の準備段階として、**極めてシンプルなGAモデル**を用いて挙動の確認を行った。<br>
            これはMVC問題とは全く関係ないが、基本的なGAの流れを確認し、GAの動作が正しく行われているか、個人的に確認するために作成した。<br>
            せっかく作成したので、ここで一応紹介する。摂動等を用いず、単純なことしかしていないので、多少は容赦いただきたい...<br><br>

            ---
            #### 問題設定
            - 各個体は800ビット（0 or 1）の配列
            - **適応度関数：ビット列中の「1の数」**
            - 最終的に **「すべて0」に近づける**ことが目標
            - 評価関数の形はMVCの「カバーサイズ最小化」に相当する簡易版

            ---
            #### 実験の目的
            - GAの基本的な流れ（選択・交叉・突然変異・エリート保存）の動作確認
            - 適応度の減少傾向を可視化し、アルゴリズムの収束特性を理解
            - 評価関数の単純化により、**GA全体の構造のデバッグが容易**

            ---
            #### 実行内容
            - 初期個体群をランダムに生成（個体数 = 1000）
            - 選択はルーレット方式（ただし逆数適応度で高スコアほど低確率）
            - 交叉は一点交叉、突然変異は無効化（mutationRate = 0）
            - エリート戦略を適用（上位10%を次世代へ無条件継承）
            - 世代数 = 1000、**最良適応度の推移を記録しグラフ化**

            ---
            #### 結果と考察
            - 初期世代では「1」の数はランダムに偏在（例：平均400前後）
            - 世代を経るごとに、ビット列の「1の数」が減少
            - 1000個体, 3000世代で回した結果、3000世代目には50以下に収束した。
            - 摂動を用いず、単純な選択と交叉のみである程度の値まで収束することが確認できた。<br>
            下の画像は1000個体、1000世代で回した結果のグラフである。<br>
            とりあえず、GAの基本的な流れは確認できたので、次のステップに進める。
            
            """, unsafe_allow_html=True)
            st.image("data/image/image0709/070901.jpeg")

    elif section == "GreedyCollection":
      
        with st.container(border=True):
            st.subheader("2. 貪欲補正（Greedy Correction）", divider="orange")

            st.markdown("""
            最小頂点被覆問題（MVC）において、ランダムで生成された初期個体（ビット列）や、交叉や突然変異によって生成された個体は、
            必ずしも**すべての辺をカバーしているとは限らない**。  
            そのため、今回は以下のような貪欲補正（Greedy Correction）なるものを導入した。

            ---
            ### 目的
            - 交叉や突然変異、ランダム生成により生じる「不完全な解（未被覆の辺が存在する）」を修正
            - GAの各世代において、**すべての個体が頂点被覆の解として有効**になるよう保証
            - これにより、**評価関数が「ノード数」だけで定義できる**ようになる（制約を満たす前提）  
            今までのように、適応度を、被覆ノード数と未被覆エッジ数の組み合わせで定義する必要がなくなる。

            ---

            ### 補正アルゴリズムの流れ（コード & 例付き）

            以下のような補正を各個体に対して行う：

            #### Step 1. 現在カバーされている辺を取得
            個体(individual) = `[1, 0, 0, 1, 0]`（ノード0と3が選ばれているとする）

            ```python
            for i, bit in enumerate(individual):
                if bit == 1:
                    u = node[i]
                    for v in G.neighbors(u):
                        covered_edges.add(tuple(sorted((u, v))))
            ```

            → ノード0と3の隣接辺を `covered_edges` に追加  
            → 残った未被覆の辺が `uncovered_edges` に入る

            #### Step 2. 各ノードの「スコア」を計算
            未被覆の各辺について、両端ノードに+1ずつ加点

            ```python
            scores = [0] * len(individual)
            for u, v in uncovered_edges:
                if individual[node.index(u)] == 0:
                    scores[node.index(u)] += 1
                if individual[node.index(v)] == 0:
                    scores[node.index(v)] += 1
            ```

            → 未被覆辺に最も多く接しているノードがスコア最大となる

            #### Step 3. スコア最大のノードを追加
            ```python
            max_idx = scores.index(max(scores))
            individual[max_idx] = 1
            ```

            → 最も効率よく未被覆辺をカバーできるノードを追加する

            #### Step 4. covered_edges と uncovered_edges を更新
            ```python
            u = node[max_idx]
            for v in G.neighbors(u):
                covered_edges.add(tuple(sorted((u, v))))
            ```

            → `uncovered_edges` を再計算し、Step 2 に戻る

            #### 終了条件
            未被覆の辺がすべてカバーされたら終了  
            → `while uncovered_edges:` のループを抜ける

            #### 補正後の個体（例）
            ```python
            Before: [1, 0, 0, 1, 0]
            After:  [1, 1, 0, 1, 1]
            ```

            ---
            ### 特徴と利点
            - 修正のたびに「最も効率的にエッジをカバーするノード」を選ぶため、**ノード追加数を最小限に抑える**
            - 実験では、**エリート保存・交叉・ランダム生成のいずれにも補正を適用**し、常に可行解を保つ

            ---
            ### 実験結果
            補正を適用した結果、以下のような挙動が確認
            
            """, unsafe_allow_html=True)
        
        st.markdown("""<br><br>""", unsafe_allow_html=True)
    
    elif section == "GC付きの実験結果①":
        with st.container(border=True):
            st.subheader("3. 実験結果と線形計画法との比較", divider="orange")

            st.markdown("""
            本章では、前章で構築した **Greedy補正付き遺伝的アルゴリズム（GA）** を用いて、  
            G_setのグラフ（ノード数800・エッジ数約4800）に対する**最小頂点被覆問題（MVC）**の近似解を求めた結果を示す。

            ---
            ### GAによる結果（Greedy補正付き）

            - ノード数：800  
            - エッジ数：約4800  
            - 個体数：100  
            - 世代数：50  
            - 突然変異率：0（無効）  
            - 貪欲補正（GreedyCorrection）：**全個体に適用**  
            - 計算時間：**約30分**  
            - 得られた最小頂点被覆サイズ：**598ノード**

            ---
            ### 線形計画法（ILP）との比較

            同じG_setのグラフに対して、前回行った**整数線形計画（ILP）**による厳密解との比較を行う：

            ```text
            Result - User ctrl-cuser ctrl-c
            Objective value:                179.00000000
            Lower bound:                    177.360
            Gap:                            0.01
            Enumerated nodes:               68221
            Total iterations:               10365924
            Time (Wallclock seconds):       1928.73
            ```

            - 最小頂点被覆サイズ（ほぼ厳密解）：**179ノード**
            - 計算時間：約32分
            - 相対ギャップ：約0.92%
            
            ---
            ### 比較まとめ

            | 方法           | 被覆ノード数 | 計算時間      | 制約充足 |
            |----------------|--------------|----------------|----------|
            | GA + Greedy補正 | **598**      | 約30分         | ◯       |
            | ILP（Pulp）     | **179**      | 約32分（打ち切り） | ◯       |
            """, unsafe_allow_html=True)
            
        st.markdown("""<br><br>""", unsafe_allow_html=True)
    elif section == "補正処理の高速化":
        with st.container(border=True):
            st.subheader("4. 補正処理の高速化とその効果", divider="orange")

            st.markdown("""
            最小頂点被覆問題に対する遺伝的アルゴリズム（GA）において、  個体の修正処理 `greedyCorrection()` は**毎世代すべての個体に対して呼び出される**ため、
            実行時間の大部分を占めるボトルネックとなっていた。

            ---
            ### 背景と目的

            - 先ほどの実装では、1回のGA実行に **約30分** を要していた
            - 主な原因は、毎回の `.index()` や `G.neighbors()` などの**非効率な操作**、つまり O(n) 操作が多発していたため
            - 計算効率を改善し、全体の実行時間を大幅に短縮することが目的

            ---
            ### 結果・効果
            
            毎回呼び出していた `greedyCorrection()` の実装を見直し、グローバル変数である `node_index` と `all_edges` を活用することで、高速化することに成功した。

            | 比較項目            | 高速化前        | 高速化後       |
            |---------------------|------------------|------------------|
            | 実行時間            | 約30分           | **約4分**         |
            | 計算条件            | 個体数100、世代数50 | 同左             |

            - 実行時間が **1/7以下** に短縮されたことで、今後の試行回数・実験数の増加が現実的に
            - 特にノード数・エッジ数が多いグラフでのGA適用が可能に

            """, unsafe_allow_html=True)       
            
        st.markdown("""<br><br>""", unsafe_allow_html=True)
    
    elif section == "全0ビット列の導入とその影響":
        with st.container(border=True):
            st.subheader("5. 全0ビット列の導入とその影響", divider="orange")

            st.markdown("""
            今回は、遺伝的アルゴリズム（GA）の初期集団に**すべて0のビット列**（すべてのノードを選ばない個体）を導入した。

            ---
            ### 実験設定

            - 対象グラフ：G_set（ノード数800、エッジ数約4800）
            - 個体数：100
            - 世代数：50
            - 初期集団：1体のみ `individual = [0, 0, ..., 0]` を含め、残りは通常のランダム生成
            - 補正：全て0ビットの個体を含めた全個体に対して `greedyCorrection()` を適用

            ---
            ### 結果

            - 実行時間や最良適応度の推移に**大きな変化は見られなかった**
            """, unsafe_allow_html=True)
        st.markdown("""<br><br>""", unsafe_allow_html=True)
        
        
    elif section == "GreedyReduction":
        with st.container(border=True):
            st.subheader("6. greedyReduction（貪欲削除）による局所改善", divider="orange")

            st.markdown("""
            本章では、新たに実装したgreedyReduction関数（貪欲削除）について、その背景、アルゴリズム、効率化の工夫を紹介する。  
            この関数は、既に頂点被覆となっている解に対して「冗長なノードを削除」し、より小さい頂点被覆（極小頂点被覆）を目指すための処理である。

            ---
            ### 1. 導入の目的

            これまで使用していた `greedyCorrection()` は、未被覆の辺をカバーするためにノードを追加する補正であったが、  
            一度追加されたノードがその後削除されることはなかった**。そのため、以下のような課題があった：

            - 補正により「過剰にノードが追加された」個体がそのまま評価対象になる
            - 結果として、より小さい頂点被覆に到達できないケースがある

            この問題を補うため、**交叉や突然変異の後に冗長なノードを削減する処理**として `greedyReduction()` を導入した。  
            これは、GA全体における**局所探索**の役割を担う。

            ---
            ### 2. アルゴリズムの流れ

            greedyReduction の処理は以下のように構成されている：

            #### （1）削除候補の選定とソート

            - 現在の頂点被覆（= individual のうち 1 のノード）を抽出
            - 各頂点の**次数（つながっている辺の本数）**をもとに、**小さい順にソート**
                - → 少ない方が冗長である可能性が高いため、削除候補として優先的に処理する

            #### （2）削除と検証

            - ソートされた頂点群に対して1つずつループ処理を行い、以下を実施：
                1. 一時的に削除（individual[i] = 0）
                2. その頂点に接続する辺が**他のノードでカバーされているか**を調べる
                    - グラフ全体ではなく、**削除対象の頂点 v の周辺のみに限定**
                    - 各隣接ノード u に対し、u もしくはその周辺に被覆ノードが存在すれば OK
                3. カバーできていない場合 → 削除を取り消して元に戻す（v は必要な頂点と判断）

            #### （3）繰り返し

            - 上記のチェックをすべての削除候補に対して行い、  
              不要な頂点だけを取り除いた**極小頂点被覆**を得る。

            ---
            ### 3. 効率化の工夫

            初期の実装では、頂点を1つ削除するたびにグラフ全体のすべての辺（4000本以上）を確認していた。  
            これは非常に非効率で、大規模グラフでは事実上使い物にならなかった。

            そこで次のような改善を加えた：

            - チェック範囲を「削除した頂点 v の周辺」のみに限定
            - v に隣接する各ノード u のみに注目し、その u が他のノードでカバーされているかだけを確認
            - それ以外の辺・ノードには一切触れない

            この改善により、削除判定のコストが劇的に下がり、  
            **実用的な速度で動作する貪欲削除アルゴリズム**となった。

            ---
            ### 4. 実装（抜粋）

            実際の実装は以下のようになっている¥：

            ```python
            def greedyReduction(individual, G, node, node_index):
                individual = individual.copy()
                current_cover_indices = [i for i, bit in enumerate(individual) if bit == 1]
                current_cover_indices.sort(key=lambda i: G.degree(node[i]))

                for i in current_cover_indices:
                    v = node[i]
                    individual[i] = 0
                    is_essential = False
                    for u in G.neighbors(v):
                        if individual[node_index[u]] == 1:
                            continue
                        for w in G.neighbors(u):
                            if individual[node_index[w]] == 1:
                                break
                        else:
                            is_essential = True
                            break
                    if is_essential:
                        individual[i] = 1
                return individual
            ```

            """, unsafe_allow_html=True)
