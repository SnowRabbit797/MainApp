# app.py
# ============================
# Streamlit + PuLP で
# 線形計画法による最小頂点被覆問題 (MVC)
# ============================

import time
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pulp

# ====================================
# 設定
# ====================================
st.set_page_config(
    page_title="最小頂点被覆（線形計画法 / PuLP）",
    layout="wide",
)

st.title("線形計画法（PuLP）による最小頂点被覆問題")

st.markdown(
    """
**概要**

- 相対パスでハードコーディングされた CSV ファイルからグラフを読み込みます  
  - 列名：`source`, `target` を想定
- PuLP で最小頂点被覆問題を線形計画法として解きます
- 計算時間・選ばれた頂点の一覧・Plotly による可視化（選ばれた頂点を塗りつぶし）を表示します
"""
)

# ★★ グラフ CSV の相対パス（必要に応じて書き換えてください） ★★
DATA_PATH = "assets/csv/G37.csv"


# ====================================
# ユーティリティ
# ====================================
@st.cache_data
def load_graph(path: str):
    """CSV から networkx グラフを構築"""
    df = pd.read_csv(path)
    # source, target の2列から無向グラフを構築
    G = nx.from_pandas_edgelist(df, source="source", target="target")
    return G, df


def solve_mvc_lp(G: nx.Graph):
    """
    PuLP を用いて最小頂点被覆問題を線形計画法で解く
    min sum x_v
    s.t. x_u + x_v >= 1  (for all edges (u,v))
         x_v in {0,1}
    """
    nodes = list(G.nodes())
    edges = list(G.edges())

    # 問題の定義（最小化問題）
    prob = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)

    # 0-1 変数 x_v
    x = pulp.LpVariable.dicts("x", nodes, cat="Binary")

    # 目的関数：選ばれた頂点数の最小化
    prob += pulp.lpSum([x[v] for v in nodes])

    # 制約：全ての辺 (u,v) について、少なくとも一方の端点を選ぶ
    for u, v in edges:
        prob += x[u] + x[v] >= 1, f"cover_edge_{u}_{v}"

    # 求解
    start_time = time.time()
    solver = pulp.PULP_CBC_CMD(msg=False)  # ログは出さない
    prob.solve(solver)
    elapsed = time.time() - start_time

    status = pulp.LpStatus[prob.status]
    obj_value = pulp.value(prob.objective)

    # x_v = 1 になっている頂点を抽出
    cover_nodes = [v for v in nodes if pulp.value(x[v]) > 0.5]

    return {
        "status": status,
        "objective": obj_value,
        "cover_nodes": cover_nodes,
        "elapsed": elapsed,
    }


def make_plotly_graph(G: nx.Graph, pos: dict, cover_nodes=None, title="グラフ可視化"):
    """
    Plotly でグラフを描画
    cover_nodes に含まれる頂点は色を変えて強調表示
    """
    if cover_nodes is None:
        cover_nodes = []

    cover_nodes = set(cover_nodes)

    # --- エッジ描画用データ ---
    edge_x = []
    edge_y = []
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
        hoverinfo="none",
        name="Edges",
    )

    # --- ノード描画用データ ---
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(n))
        # 頂点被覆に含まれるなら色を変える
        if n in cover_nodes:
            node_color.append("tomato")  # 選ばれた頂点
        else:
            node_color.append("lightgray")  # 非選択頂点

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=18,
            color=node_color,
            line=dict(width=1),
        ),
        name="Nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# ====================================
# メイン処理
# ====================================
st.subheader("1. グラフの読み込み")

try:
    G, df_edges = load_graph(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"グラフの CSV ファイル `{DATA_PATH}` が見つかりません。\n"
        "適当な `source,target` 形式の CSV を用意して、このパスに配置してください。"
    )
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**エッジリスト (source, target)**")
    st.dataframe(df_edges)

with col2:
    st.markdown("**グラフ情報**")
    st.write(f"- ノード数: {G.number_of_nodes()}")
    st.write(f"- エッジ数: {G.number_of_edges()}")

# 座標は固定しておきたいので一度だけ計算
pos = nx.spring_layout(G, seed=0)


st.subheader("2. 線形計画法で最小頂点被覆を求める")

if st.button("線形計画法で MVC を解く"):
    result = solve_mvc_lp(G)

    status = result["status"]
    objective = result["objective"]
    cover_nodes = result["cover_nodes"]
    elapsed = result["elapsed"]

    st.markdown("### 結果概要")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("最小頂点被覆のサイズ", len(cover_nodes))
    with col_b:
        st.metric("目的関数値", int(objective) if objective is not None else "-")
    with col_c:
        st.metric("計算時間 [秒]", f"{elapsed:.4f}")

    st.write(f"ソルバの状態: **{status}**")

    st.markdown("### 選ばれた頂点の一覧")
    if cover_nodes:
        st.dataframe(pd.DataFrame({"vertex": cover_nodes}))
    else:
        st.info("選ばれた頂点がありません（解が見つかっていない可能性があります）。")

    st.markdown("### 3. グラフ可視化（選ばれた頂点を塗りつぶし）")

    fig = make_plotly_graph(
        G,
        pos,
        cover_nodes=cover_nodes,
        title="最小頂点被覆の可視化（赤=選ばれた頂点）",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("「線形計画法で MVC を解く」ボタンを押すと、計算と可視化が実行されます。")
    # 初期表示としてカバー無しのグラフだけ描画
    fig0 = make_plotly_graph(G, pos, cover_nodes=None, title="元のグラフ")
    st.plotly_chart(fig0, use_container_width=True)
