import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import io
import random
from itertools import combinations

# ★ networkx 3.x 対応：random_tree はここから
from networkx.generators.trees import random_tree

st.set_page_config(page_title="ランダムグラフ生成（Plotly・孤立点なし）", layout="wide")
with st.container(border=False):
    st.title("ランダムグラフ生成")

# ----------------------------------
# 孤立点なしグラフ生成関数
# ----------------------------------
def generate_graph_without_isolates(n: int, p: float, seed: int = 42) -> nx.Graph:
    """
    1. ランダムな木（random_tree）で必ず連結なグラフを作る
    2. まだ存在しない辺を確率 p で追加
    → 必ず孤立点なし（連結）
    """
    rng = random.Random(seed)

    # ランダムな木（ノードは 0..n-1）
    G = random_tree(n, seed=seed)

    # 木に存在しないノードペアに対して確率 p で辺を追加
    existing_edges = set(map(frozenset, G.edges()))
    for u, v in combinations(range(n), 2):
        if frozenset((u, v)) in existing_edges:
            continue
        if rng.random() < p:
            G.add_edge(u, v)

    return G

# ----------------------------------
# 3カラムレイアウト
# ----------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

# ===== 左：設定カラム =====
with col1:
    st.subheader("パラメータ")
    n = st.number_input("ノード数", min_value=2, max_value=50, value=10)
    p = st.slider("追加の辺を張る確率 p", 0.0, 1.0, 0.3, 0.01)
    seed = st.number_input("乱数シード", min_value=0, max_value=9999, value=42)
    generate_button = st.button("グラフ生成")

# ===== ボタンが押されたらグラフ生成＆表示 =====
if generate_button:
    G = generate_graph_without_isolates(n=int(n), p=float(p), seed=int(seed))

    # 座標
    pos = nx.spring_layout(G, seed=int(seed))

    # -----------------------------
    # 中央：グラフ可視化（Plotly）
    # -----------------------------
    with col2:
        st.subheader("グラフ可視化")

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none"
        )

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(size=18),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=30),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"ノード数: {G.number_of_nodes()} / 辺数: {G.number_of_edges()}")

    # -----------------------------
    # 右：CSV プレビュー & ダウンロード
    # -----------------------------
    with col3:
        st.subheader("CSV")

        edges = [{"source": u, "target": v, "weight": 1} for u, v in G.edges()]
        df_edges = pd.DataFrame(edges)

        st.dataframe(df_edges, use_container_width=True)

        csv_buffer = io.StringIO()
        df_edges.to_csv(csv_buffer, index=False)

        st.download_button(
            label="CSV をダウンロード",
            data=csv_buffer.getvalue(),
            file_name="random_graph_no_isolates.csv",
            mime="text/csv"
        )
else:
    with col2:
        st.info("左でパラメータを設定して「グラフ生成」ボタンを押すと、ここにグラフが表示されます。")
    with col3:
        st.info("グラフ生成後、このカラムに CSV の内容プレビューとダウンロードボタンが表示されます。")
