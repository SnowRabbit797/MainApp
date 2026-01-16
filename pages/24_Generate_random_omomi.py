import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import io
import random
from itertools import combinations


st.set_page_config(page_title="ランダム重み付きグラフ生成（整数・孤立点なし）", layout="wide")
st.title("ランダム重み付きグラフ生成（整数）")

# ----------------------------------
# 孤立点なし + 整数重みグラフ生成
# ----------------------------------
def generate_weighted_graph_without_isolates(
    n: int,
    p: float,
    seed: int,
    w_min: int,
    w_max: int,
) -> nx.Graph:
    """
    - ランダムチェーンで必ず連結
    - 追加辺は確率 p
    - weight は必ず int
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # --- Step 1: 連結骨格 ---
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(n - 1):
        w = rng.randint(w_min, w_max)
        G.add_edge(nodes[i], nodes[i + 1], weight=w)

    # --- Step 2: 追加のランダムエッジ ---
    existing_edges = set(map(frozenset, G.edges()))
    for u, v in combinations(range(n), 2):
        if frozenset((u, v)) in existing_edges:
            continue
        if rng.random() < p:
            w = rng.randint(w_min, w_max)
            G.add_edge(u, v, weight=w)

    return G


# ----------------------------------
# レイアウト
# ----------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

# ===== 左：設定 =====
with col1:
    st.subheader("パラメータ")
    n = st.number_input("ノード数", 2, 50, 10)
    p = st.slider("追加辺確率 p", 0.0, 1.0, 0.3, 0.01)
    seed = st.number_input("乱数シード", 0, 9999, 42)

    st.divider()
    st.subheader("重み（整数）")
    w_min = st.number_input("weight 最小", value=1, step=1)
    w_max = st.number_input("weight 最大", value=10, step=1)

    show_weight = st.checkbox("辺に weight を表示", False)

    generate_button = st.button("グラフ生成")

# ===== 生成 =====
if generate_button:
    G = generate_weighted_graph_without_isolates(
        n=int(n),
        p=float(p),
        seed=int(seed),
        w_min=int(w_min),
        w_max=int(w_max),
    )

    pos = nx.spring_layout(G, seed=int(seed))

    # ---------- 中央：可視化 ----------
    with col2:
        st.subheader("グラフ可視化")

        edge_x, edge_y = [], []
        edge_mx, edge_my, edge_text = [], [], []

        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

            if show_weight:
                edge_mx.append((x0 + x1) / 2)
                edge_my.append((y0 + y1) / 2)
                edge_text.append(str(d["weight"]))  # ← int

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none"
        )

        node_x, node_y, node_text = [], [], []
        for v in G.nodes():
            x, y = pos[v]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(v))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=18),
        )

        data = [edge_trace, node_trace]

        if show_weight:
            weight_trace = go.Scatter(
                x=edge_mx, y=edge_my,
                mode="text",
                text=edge_text,
                hoverinfo="none"
            )
            data.insert(1, weight_trace)

        fig = go.Figure(
            data=data,
            layout=go.Layout(
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=5, r=5, b=20, t=30),
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        st.caption(
            f"ノード数: {G.number_of_nodes()} / 辺数: {G.number_of_edges()} "
            f"/ weight(min,max)=({min(weights)}, {max(weights)})"
        )

    # ---------- 右：CSV ----------
    with col3:
        st.subheader("CSV")

        df = pd.DataFrame(
            [{"source": u, "target": v, "weight": int(d["weight"])}
             for u, v, d in G.edges(data=True)]
        )

        st.dataframe(df, use_container_width=True)

        buf = io.StringIO()
        df.to_csv(buf, index=False)

        st.download_button(
            "CSV をダウンロード",
            data=buf.getvalue(),
            file_name="random_weighted_graph_int.csv",
            mime="text/csv"
        )

else:
    with col2:
        st.info("左で設定して「グラフ生成」を押してください。")
    with col3:
        st.info("生成後、CSV を表示・ダウンロードできます。")
