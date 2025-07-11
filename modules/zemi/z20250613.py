import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY
from pulp import LpMaximize, LpProblem, LpVariable, value, LpMinimize, LpBinary, lpSum, LpStatus


def main():
    st.title("6月13日(第4回)の発表")
    st.markdown("""<br>""", unsafe_allow_html=True)
    # with st.container(border = True):
    #     st.subheader("本日の発表内容", divider="red")
    #     st.markdown("""
    #         <ol>
    #             <li>進化計算アルゴリズムとは</li>
    #             <li>GAによるMVC問題への近似的アプローチ</li>
    #             <li>最小頂点被覆問題を整数計画法で解く</li>
    #             <li>最小頂点被覆問題を整数計画法で解く②</li>
    #         </ol>
    #     """, unsafe_allow_html=True)
    # # st.markdown("""<br>""", unsafe_allow_html=True)
    # st.markdown("---")
    
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
    st.markdown("""## 2.遺伝的アルゴリズム(GA)""", unsafe_allow_html=True)
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
    
    #-------------------------------選択---------------------------------------
    
    st.markdown("---")
    st.markdown("""### 2-1.選択""", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("ルーレット選択", divider="orange")
        st.markdown("""
        個体の適応度に比例して選択される確率を決定する方法。適応度が高いほど選ばれやすくなる。
        """)
    st.write("学部時代の資料(ルーレット選択)")
    st.image("data/image/image0613/250612_01.jpg")
    st.image("data/image/image0613/250612_02.jpg")
    col1, col2 = st.columns(2)
    
    with col1:
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
    with col2:
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
    with st.container(border=True):
        st.subheader("②選択(エリート選択)", divider="orange")
        st.markdown("1番適応度が良い個体を無条件で次世代に残す方法")
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<br>""", unsafe_allow_html=True)
        
    #-------------------------------交叉---------------------------------------
    
    st.markdown(""" ### 2-2.交叉""", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("交叉", divider="orange")
        st.markdown("""
        2つの親個体から新しい子個体を生成する。遺伝子の一部を交換することで、より良い解を探索する。
        """)
    st.write("学部時代の資料(交叉)")
    st.image("data/image/image0613/250612_03.jpg")
    st.image("data/image/image0613/250612_04.jpg")
    st.image("data/image/image0613/250612_05.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("コード例(二点交差)")
        st.code("""
            import random

            def two_point_crossover(parent1, parent2):
                length = len(parent1)
                point1 = random.randint(0, length - 2)
                point2 = random.randint(point1 + 1, length-1)
                child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
                return child1, child2, point1, point2
              
            parent1 = "110100"
            parent2 = "101011"
            child1, child2, point1, point2 = two_point_crossover(parent1, parent2)

            print("親1:", parent1)
            print("親2:", parent2)
            print("交差点1:", point1)
            print("交差点2:", point2)
            print("子1:", child1)
            print("子2:", child2)


        """)
        with col2:
            st.write("出力例")
            st.code("""
                    親1: 110100
                    親2: 101011
                    交差点1: 3
                    交差点2: 5
                    子1: 110010
                    子2: 101101
                    """)
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<br>""", unsafe_allow_html=True)
    
    #-------------------------------突然変異---------------------------------------
    st.markdown("""### 2-3.突然変異""", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("突然変異", divider="orange")
        st.markdown("""
        個体の遺伝子をランダムに変更することで、多様性を保つ。局所最適から脱出するために重要。
        """)
    st.write("学部時代の資料(突然変異)")
    st.image("data/image/image0613/250612_06.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("コード例(ビット反転)")
        st.markdown("今回は例のため確定で突然変異が起こるようにしている。")
        st.code("""
                import random
                def mutate(individual):
                    pos = random.randint(0, len(individual) - 1) 
                    if individual[pos] == '0':
                        mutated_num = '1'
                    else:
                        mutated_num = '0'
                    mutated = individual[:pos] + mutated_num + individual[pos+1:]
                    return mutated

                individual = "110100"
                mutated_individual = mutate(individual)
                print("Before:", individual)
                print("After: ", mutated_individual)
        """)
    with col2:
        st.write("出力例")
        st.code("""
                Before: 110100
                After:  110000
                """)
    
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<br><br>""", unsafe_allow_html=True)
    
    st.markdown("""## 2.GAによるMVC問題への近似的アプローチ""", unsafe_allow_html=True)
    st.markdown("""ここからは遺伝的アルゴリズム(GA)を用いて最小頂点被覆問題(MVC)を解く。<br>
                (一応)<b>最小頂点被覆問題</b>とは、与えられたグラフにおいて、全てのエッジを少なくとも1つの頂点で被覆するための頂点の最小集合を求める問題である。<br>
                以下では、GAを用いてMVC問題を近似的に解く方法を考えていく。<br>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""今回使うデータは以下の二種類""", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["オリジナルランダムグラフ", "G_setから引用したグラフ"])
        
        with tab1:
            st.markdown("""オリジナルランダムグラフ""", unsafe_allow_html=True)
            st.markdown("""<br>""", unsafe_allow_html=True)
            st.write("次回使用")
        
        with tab2:
            st.write("Gsetと呼ばれる最大カット問題のベンチマークセットをNVCに使用してみる。ダウンロードは以下のリンクから。")
            st.write("https://web.stanford.edu/~yyye/yyye/Gset/")
            uploaded_file = "assets/csv/G_set1.csv"

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, skiprows=3)

                G = nx.from_pandas_edgelist(df, source="from", target="to", edge_attr="weight", create_using=nx.Graph)

                st.write(f"ノード数: {G.number_of_nodes()}")
                st.write(f"エッジ数: {G.number_of_edges()}")

                pos = nx.spring_layout(G, seed=42)
                plt.figure(figsize=(8, 8))
                nx.draw(G, pos, node_size=10, edge_color="gray", with_labels=False)
                st.pyplot(plt)
            st.write("以上のグラフをPulpを用いて整数計画法で解いてみた結果がこちら。")
            st.code("""
                        Result - User ctrl-cuser ctrl-c
                        Objective value:                179.00000000
                        Lower bound:                    177.360
                        Gap:                            0.01
                        Enumerated nodes:               68221
                        Total iterations:               10365924
                        Time (CPU seconds):             1918.45
                        Time (Wallclock seconds):       1928.71

                        Option for printingOptions changed from normal to all
                        Total time (CPU seconds):       1918.46   (Wallclock seconds):       1928.73
                        """)
            
            st.write("""
                    つまりは、<br>
                    最小頂点被覆サイズ: 179<br>
                    下限値: 177.360  <br>
                    相対ギャップ: 約0.92%  <br>
                    計算時間: 約32分  <br>
                    となった。<br>
                    32分経った段階で、強制的に打ち切った。その際の最小頂点被覆サイズは179であった。<br>
                    しかし、下限値は177.360であり、相対ギャップは約0.92%である。<br>
                    相対ギャップとは、最小頂点被覆サイズと下限値の比率を表す指標であり、0に近いほど解が良いことを示す(らしい)。<br>
                    """, unsafe_allow_html=True)
    
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown("""再度GAの実装の流れを確認する。""", unsafe_allow_html=True)
    
    st.markdown("""
    1. **初期集団の生成**  
      ランダムに個体（解候補）を生成する。
      
      

    2. **適応度の評価**  
      各個体にスコアを与え、どれだけ「良い」かを評価する。

    3. **選択（Selection）**  
      良い個体ほど次世代に残りやすくなるように選ぶ。 

    4. **交叉（Crossover）**  
      親個体の一部を組み合わせて新しい個体を生成する。  

    5. **突然変異（Mutation）**  
      個体の遺伝子を一部ランダムに変化させ、多様性を保つ。  

    6. **次世代の更新**  
      新しく生まれた個体で次世代を構成する。必要に応じてエリート保存も行う。

    7. **終了条件の判定**  
      規定世代に達した、または解が十分良くなったら終了。
    """)
    st.markdown("""<br>""", unsafe_allow_html=True)

    st.markdown("""以下ではG_setを例として説明する。""", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("1. 初期集団の生成", divider="orange")

        st.markdown("""
        最小頂点被覆問題において、1つの個体は「各ノードをカバー集合に含めるかどうか」を表すビット列（0/1のリスト）として表現される。  
        ノード数が800であれば、各個体は長さ800の0/1の配列となる。

        - `1` → 該当ノードをカバー集合に含める  
        - `0` → 該当ノードを含めない

        ---
        #### 個体の例（ノード数 = 10 の場合）
        ```python
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        ```
        → ノード 0, 2, 3, 6, 9 がカバー集合に含まれていることを意味する。

        初期集団（population）は、こうした個体を**ランダムに複数生成**することで構成される。  
        今回は、**個体数（pop_size）= 100** として初期集団を構築する。
        """)

        st.code("""
            import random

            def init_population(pop_size, num_nodes):
                population = []
                for _ in range(pop_size):
                    individual = [random.randint(0, 1) for _ in range(num_nodes)]
                    population.append(individual)
                return population

            population = init_population(pop_size=100, num_nodes=800)
            """, language="python")

        st.markdown("""
        上記の関数では、100個体を生成し、それぞれが800ビット（0/1）のリストになる。  
        実際のnum_nodesの値は、csvファイルから読み込んだグラフのノード数に基づいて動的に設定する。
        """)
    st.markdown("""<br><br><br>""", unsafe_allow_html=True)
    
    #-------------------------------適応度---------------------------------------
    
    with st.container(border=True):
        st.subheader("2. 適応度の評価", divider="orange")

        st.markdown(r"""
        最小頂点被覆問題において、各個体がどれだけ「良い解」であるかを数値として評価するために、**適応度（fitness）** を計算する。  
        適応度関数は、以下の2つの要素に基づいて設計される：

        - **カバーされた辺の数（covered_edges）** が多いほど良い  
        - **使用したノード数（cover_node_count）** が少ないほど良い  

        ---
        ### 評価関数の設計

        適応度は以下の数式で定義する。

        $$
        \text{fitness} = \alpha \cdot \text{covered\_edges} - \beta \cdot \text{cover\_node\_count}
        $$

        - $$\alpha$$ ：カバーされた1本の辺あたりの貢献度（例：1000）  
        - $$\beta$$ ：使用された1つのノードあたりのコスト（例：1）
        
        

        この評価関数により、まずは制約（すべての辺をカバー）を満たす方向に進化が促され、  
        その後、使用するノード数を削減する方向へと最適化が進む。
        
        ※追記：次回の発表で行う「BLS」では、$$\alpha$$と$$\beta$$の値を変化させて局所最適に陥りにくくする工夫を行う。

        ---
        ### 評価例

        ノード数 = 800、エッジ数 = 4696 のとき：

        - 個体A：covered_edges = 4696、cover_node_count = 400  
          → fitness =  1000 $$\times$$ 4696 - 1 $$\times$$ 400 = **4,695,600**

        - 個体B：covered_edges = 4000、cover_node_count = 200  
          → fitness =  1000 $$\times$$ 4000 - 1 $$\times$$ 200 = **3,999,800**

        個体Aは全エッジをカバーしており、スコアが高いことが分かる。

        ---
        ### 評価関数の実装（Python）

        """, unsafe_allow_html=True)

        st.code("""
            def evaluate_fitness(individual, edge_list, alpha=1000, beta=1):
                covered_edges = 0
                for u, v in edge_list:
                    if individual[u] == 1 or individual[v] == 1:
                        covered_edges += 1
                cover_node_count = sum(individual)
                fitness = alpha * covered_edges - beta * cover_node_count
                return fitness

            fitness_list = [evaluate_fitness(ind, edge_list) for ind in population]

            """, language="python")

        st.markdown(r"""
        この関数は、**カバーされている辺の数**と**使用されたノード数**のバランスを考慮し、  
        **スコアが大きいほど優れた個体**と判断できるように設計される。
        """, unsafe_allow_html=True)

    st.markdown("""<br><br><br>""", unsafe_allow_html=True)
    
    #-------------------------------選択---------------------------------------
    
    with st.container(border=True):
        st.subheader("3. 選択", divider="orange")

        st.markdown(r"""
        先ほどのルーレット選択を用いることにする。
        
        高適応度の個体を優先しつつ、**他の個体にも選ばれるチャンス**があり、局所解を回避しつつ多様性を保った探索が可能となる。
        
        しかし、デメリットとしてルーレット選択はランダム性があるため。良い個体が選ばれない可能性もある。
        そのため、**エリート保存戦略** を併用し、最も適応度の高い個体を必ず次世代に残すようにする。

        ---
        ### 実装例（Python）
        """, unsafe_allow_html=True)

        st.code("""

            import random

            def roulette_selection(population, fitness_list, num_selected):
                total_fitness = sum(fitness_list)
                
                cumulative_probs = []
                cumulative_sum = 0
                for f in fitness_list:
                    cumulative_sum += f
                    cumulative_probs.append(cumulative_sum / total_fitness)

                selected = []
                for _ in range(num_selected):
                    r = random.random()
                    for i, prob in enumerate(cumulative_probs):
                        if r <= prob:
                            selected.append(population[i])
                            break
                return selected

        """, language="python")

        st.markdown(r"""
        この方法では、最も良い個体が必ず選ばれるとは限らず、**全体のバランスを保ちながら次世代を構成**できる。  
        そのため、**局所解に陥りにくい**という利点がある。
        """, unsafe_allow_html=True)
        
    st.markdown("""<br><br><br>""", unsafe_allow_html=True)
    #-------------------------------交叉---------------------------------------

    with st.container(border=True):
        st.subheader("4.交叉  &  5.突然変異", divider="orange")

        st.markdown(r"""
        次世代の個体を生成する際には、選択された親個体をもとに**交叉(Crossover)** と **突然変異(Mutation)** を適用する。  
        これにより、既存の優れた個体を活用しつつ、新しい探索領域への多様性も確保される。

        ---
        ### 交叉（Crossover）

        2つの親個体をペアとして、それぞれのビット列（0/1）を**50%の確率で交差(一様交叉)**させて子を生成する。

        - 親1: `[1, 0, 1, 0, 0]`  
        - 親2: `[0, 1, 0, 1, 1]`  
        → 子1: `[1, 1, 1, 0, 1]`  
        → 子2: `[0, 0, 0, 1, 0]`  
        （位置ごとに親からランダムに遺伝子を選ぶ）

        ---
        ### 突然変異（Mutation）

        子個体に対し、各ビットに**一定の確率（例: 1%）で反転操作**を行う。  
        これにより、個体群の多様性を維持し、局所解に陥るのを防ぐ。

        ---
        ### 実装例（Python）

        """, unsafe_allow_html=True)

        st.code("""
        import random

        def crossover_and_mutate(parent1, parent2, mutation_rate=0.01):
            length = len(parent1)
            child1 = []
            child2 = []

            for i in range(length):
                if random.random() < 0.5:
                    gene1 = parent1[i]
                    gene2 = parent2[i]
                else:
                    gene1 = parent2[i]
                    gene2 = parent1[i]

                if random.random() < mutation_rate:
                    gene1 = 1 - gene1
                if random.random() < mutation_rate:
                    gene2 = 1 - gene2

                child1.append(gene1)
                child2.append(gene2)

            return child1, child2
        """, language="python")

        st.markdown(r"""
        交叉と突然変異を組み合わせることで、親個体の良い特徴を引き継ぎつつ、  
        未探索の領域にも進出できるようになり、進化の効果が高まる。
        """, unsafe_allow_html=True)
    st.markdown("""<br><br><br>""", unsafe_allow_html=True)
    #-------------------------------実装---------------------------------------
    st.write("GAによるMVC問題の実装(Experiment2にて)")
    with st.container(border=True):
        st.subheader("振り返りと次回やること。", divider="orange")
        
        st.markdown(r"""
                    
                    
                    #### 区間値フィットネスの適用
                      
                    以下の論文では、MVC問題に対して区間値フィットネスを適用したGAを提案している。  
                    (自分の調べた限りでは)区間値フィットネスを用いた今回の問題は、グラフ全体をいくつかの部分グラフに分割し、  
                    各部分グラフに対して個別に適応度を計算し、各部分グラフで最も適応度の高い個体を組み合わせることでスーパー個体なるものを作成しているらしい。  
                    つまり、全体最適化ではなく、部分最適化を行うことで素早く最適な解を得ることができる。  
                      
                    https://acta.uni-obuda.hu/Nagy_Szokol_111.pdf  
                    Benedek Nagy, Péter Szokol. A Genetic Algorithm for the Minimum Vertex Cover Problem with Interval-Valued Fitness. Acta Polytechnica Hungarica. 2021, vol. 18, no. 4, pp. 105-123.  
                    
                    ---
                    
                    #### 初期集団の生成方法
                    
                    各パーテーション(部分グラフ)に分けたとしても、今回自分がやったようにランダムで作成した個体には未カバーのエッジが存在する可能性がある。  
                    それを防ぐために、「greedy error collection(貪欲的誤り修正)」を適用したい。  
                    具体的に何をするのかといえば、各パーテーションに対して未カバーのエッジをできるだけ少ないコストでカバーするために貪欲法でノードを追加する。という補正操作を行う。  
                    
                    ---
                    
                    #### 適応度の評価方法
                    パーテーションを小さく取り、またgreedy error collectionを適用した初期集団を生成することで、適応度の評価は「ノード数」だけで単純に決めることができそう。  
                    
                    ---
                    
                    #### (次回やること)
                    - MVCを区間値フィットネス, greedy error collection用いたGAで解く。
                    - それに追加してBLSを使い、局所最適に陥りにくいようにする。
                    - G_set含めた他のグラフでも試し、それらの結果を比較して、どのようなグラフに対して区間値フィットネスが有効かを考える。
                    - グリーディ戦略の使用有無
                    - アイディア、フローチャートの設計
                    - BLSの適用の際の各アルゴリズム内のメリットデメリットの把握
                    - MVCに対する各パーテーションの分け方
                      - 主要ノードに対してBFSを用いる
                      - BLSの適用の際の各アルゴリズム内のメリットデメリットの把握
                    - パーテーションを分けずに、全体的にerror collectionを実行してみる
                    
        """, unsafe_allow_html=True)
