import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.sidebar.title("25/04/25 ゼミ発表資料")
    section = st.sidebar.radio("目次", ["導入", "グラフの描画(脱線話)", "貪欲法", "networkxの説明", "次回は？"])

    if section == "導入":
        st.title("2025年4月25日 M2ゼミ発表(2回目)")
        st.title("北村竜嗣様、御誕生日おめでとうございます。")
        st.markdown("---")
        st.write("""今回からstreamlitを使用して、ゼミ発表の資料を作成しています。今回はお試しとして色々な機能を試してみています。""")
        st.write("まだversionは1.0なので、使いにくい部分もありますが、ご了承ください。")
        st.markdown("---")
        st.image("data/image/image0425/250425_page-0001.jpg")
        st.image("data/image/image0425/250425_page-0002.jpg")
        st.image("data/image/image0425/250425_page-0003.jpg")
        st.image("data/image/image0425/250425_page-0004.jpg")
        st.image("data/image/image0425/250425_page-0005.jpg")
        st.image("data/image/image0425/250425_page-0006.jpg")
        
        
    #----------------------------------------------------------

    elif section == "グラフの描画(脱線話)":
        csv_path = "assets/csv/mvcfSurugadai.csv"
        df = pd.read_csv(csv_path)

        # グラフ作成
        G = nx.Graph()
        G.add_edges_from(df.values)

        # レイアウトと描画
        pos = nx.spring_layout(G, seed=4)
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=12, ax=ax)

        # Streamlitで表示
        st.title("CSVから読み込んだグラフの描画")
        st.write("今回使用するデータセット(D1)は先程の駿河台の地図を参考にした以下の通り。")
        st.code("""
            source,target
            1,2
            1,5
            2,3
            2,7
            3,4
            3,8
            4,8
            5,6
            5,7
            6,10
            6,11
            7,11
            8,9
            9,10
            9,11
        """)
        st.write("今回使用するライブラリは以下の通り。")
        st.code("""
            import streamlit as st        # Webアプリを作るためのライブラリ
            import pandas as pd           # CSVなどの表データを扱うライブラリ
            import networkx as nx         # グラフ（ノードとエッジ）を扱うライブラリ
            import matplotlib.pyplot as plt  # グラフを描画するライブラリ
        """)
        st.markdown("---")
        st.write("""駿河台のグラフを参考に取ったデータ(D1)をグラフ表示した結果が以下の通り。
                networkxライブラリを元に描画したため、割と綺麗に描画はできているが、元の駿河台のマップとは少し違う。""")
        st.pyplot(fig)
        
    #----------------------------------------------------------
    elif section == "貪欲法":
        st.image("data/image/image0425/250425_page-0007.jpg")
        st.image("data/image/image0425/250425_page-0008.jpg")
        st.image("data/image/image0425/250425_page-0009.jpg")
        st.latex(r"""\text{最終的に } C = \{2,\,4,\,5,\,8,\,10,\,11\} \text{ となり、}
                    \text{最小のカバー数は } 6 \text{ となる。}
        """)
        st.markdown("---")
        st.markdown("""
            <h3><b>ここから貪欲法の実装に入る。</b></h3>
            <p>実装の順序は以下の通り。</p>
            <ol>
                <li>グラフのノードを全て選択肢に入れる。</li>
                <li>選択肢の中から、最も多くのエッジをカバーするノードを選ぶ。</li>
                <li>選んだノードをカバーセットに追加し、選択肢から削除する。</li>
                <li>選択肢の中から、選んだノードに隣接するノードを全て削除する。</li>
                <li>選択肢が空になるまで、2~4を繰り返す。</li>
                <li>カバーセットを返す。</li>
            </ol>
        """, unsafe_allow_html=True)
        #-----------------------------------------------------------
        st.markdown("""<br>
            <h3><b>実装のコードは以下の通り。</b></h3>
        """, unsafe_allow_html=True)
        st.code("""
            df = pd.read_csv("csvデータのパス")

            G = nx.Graph()
            G.add_edges_from(df.values)

            def greedy_vertex_cover(graph):
                cover = set()
                edges = set(graph.edges())

                while edges:
                    degree_node = max(graph.degree, key=lambda x: len([e for e in edges if x[0] in e]))
                    cover.add(degree_node[0])
                    edges = {e for e in edges if degree_node[0] not in e}
                return sorted(cover)
            vertex_cover = greedy_vertex_cover(G)
            st.write("最小頂点被覆（貪欲法による近似解）:", vertex_cover)
        """)
        
        # assets フォルダ内の CSV ファイルを読み込む
        df = pd.read_csv("assets/csv/mvcfSurugadai.csv")

        # グラフ構築
        G = nx.Graph() #空のグラフを作成
        G.add_edges_from(df.values) #CSVからエッジを追加

        # 貪欲法で最小頂点被覆を求める関数（前述と同じ）
        def greedy_vertex_cover(graph):
            cover = set()
            edges = set(graph.edges())

            while edges:
                degree_node = max(graph.degree, key=lambda x: len([e for e in edges if x[0] in e]))
                cover.add(degree_node[0])
                edges = {e for e in edges if degree_node[0] not in e}

            return sorted(cover)

        # 結果の表示
        vertex_cover = greedy_vertex_cover(G)
        st.markdown("<h3><b>出力結果</b></h3>", unsafe_allow_html=True)
        st.write("最小頂点被覆（貪欲法による近似解）: C =", "{" + ", ".join(map(str, vertex_cover)) + "}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <h3><b>実装の結果、最小頂点被覆は以下のようになった。</b></h3>
            <p>最小頂点被覆（貪欲法による近似解）: C = {2, 4, 5, 8, 10, 11}</p>
            <p>最小のカバー数は 6 となる。</p>
            <p>貪欲法は近似解を求める手法であり、最適解を保証するものではないが、この小ささのグラフであれば多分これが最適解なはず。</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("""
            <h2>実装コードについて</h2>
            <p>実装コードに関しては実装の順序に準じている。</p>
            <p>空のグラフを作成し、CSVからエッジを追加している。また、グラフのノードを全て選択肢に入れ、選択肢の中から最も多くのエッジをカバーするノードを選び、選んだノードをカバーセットに追加し、選択肢から削除している。</p>
            <p>ライブラリ：networkxを使用することで、グラフの描画やノードの選択肢の管理が容易になっている。</p>
            <p>最後にライブラリ：networkxの説明だけしておきます。</p>
        """, unsafe_allow_html=True)
    elif section == "networkxの説明":
        st.title("networkxの説明")
        st.markdown("""
            <h3><b>networkxとは</b></h3>
            <p>networkxはPythonで書かれたグラフ理論のライブラリで、グラフの作成、操作、描画を簡単に行うことができる。ネットワーク解析やグラフアルゴリズムの実装に便利なツール。</p>
        """, unsafe_allow_html=True)
        st.markdown("""
            <h3><b>主な機能</b></h3>
            <ul>
                <li>グラフの作成：有向グラフ、無向グラフ、重み付きグラフなどを簡単に作成できる。</li>
                <li>ノードとエッジの操作：ノードやエッジの追加、削除、属性の設定が可能。</li>
                <li>アルゴリズム：最短経路探索、連結成分の検出、中心性の計算など、多くのアルゴリズムが実装されている。</li>
                <li>描画：matplotlibを使用してグラフを描画することができる。</li>
            </ul>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h3>貪欲法で使用した例</h3>", unsafe_allow_html=True)
        st.write("今回は以下のデータを使用した。どことどこが繋がっているか(エッジ)を示した単純なデータ。")
        st.code("""
            source,target
            1,2
            1,5
            2,3
            2,7
            3,4
            3,8
            4,8
            5,6
            5,7
            6,10
            6,11
            7,11
            8,9
            9,10
            9,11
        """)
        st.markdown("<h3>まずはこのデータをコードでグラフとして保存する</h3>", unsafe_allow_html=True) 
        st.code("""
            df = pd.read_csv("なんとか.csv")
            G = nx.Graph() 
            G.add_edges_from(df.values)
        """)
        st.write("このようにしてグラフを作成することができる。空のグラフを作成し、CSVからエッジを追加している。") 
        
        #-----------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3>グラフの詳細を見る</h3>", unsafe_allow_html=True)
        st.code("""
            st.write(G)
        """)
        df = pd.read_csv("assets/csv/mvcfSurugadai.csv")
        edges = [(int(a), int(b)) for a, b in df.values]
        G = nx.Graph() 
        G.add_edges_from(edges) 
        st.write(G)
        #-----------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3>グラフのエッジ(辺)</h3>", unsafe_allow_html=True)
        st.code("""
            st.write(G.edges())
        """)
        df = pd.read_csv("assets/csv/mvcfSurugadai.csv")
        edges = [(int(a), int(b)) for a, b in df.values]
        G = nx.Graph() 
        G.add_edges_from(edges) 
        st.write(G.edges())
        #-----------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3>グラフのノード(頂点)</h3>", unsafe_allow_html=True)
        st.code("""
            st.write(G.nodes())
        """)
        df = pd.read_csv("assets/csv/mvcfSurugadai.csv")
        edges = [(int(a), int(b)) for a, b in df.values]
        G = nx.Graph() 
        G.add_edges_from(edges) 
        st.write(G.nodes())
        st.write("エッジの情報しか入れていないのに、ノードの情報も自動的に取得できる。")
        #-----------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3>各ノードの次数を取得するもの</h3>", unsafe_allow_html=True)
        st.code("""
            st.write(G.degree())
        """)
        df = pd.read_csv("assets/csv/mvcfSurugadai.csv")
        edges = [(int(a), int(b)) for a, b in df.values]
        G = nx.Graph() 
        G.add_edges_from(edges) 
        st.write(G.degree())
        st.write("(1,2)であれば、頂点1には次数が2、(2,3)であれば、頂点2には次数が3つあることになる。")
    elif section == "次回は？":
        st.title("次回は？")
        st.markdown("""
            <ul>
                <li>もっと大きなデータセットを用意する</li>
                <li>遺伝的アルゴリズム、粒子群最適化で実装してみる</li>
                <li>？？</li>
                <li>？？</li>
            </ul>
        """, unsafe_allow_html=True)
