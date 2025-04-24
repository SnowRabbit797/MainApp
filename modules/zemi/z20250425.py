import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def main():
    st.sidebar.title("25/04/25 ゼミ発表資料")
    section = st.sidebar.radio("目次", ["導入", "グラフの描画(脱線話)", "貪欲法", "結論"])

    if section == "導入":
        st.title("2025年4月25日 M2ゼミ発表(2回目)")
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
        st.write("テスト")
    elif section == "結論":
        st.write("テスト")
    
