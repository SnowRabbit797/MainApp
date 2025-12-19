# # demo_ga_plot.py
# # streamlit run demo_ga_plot.py
# # ---------------------------------------------
# # GA比較グラフのダミー生成ツール
# # - 実際のGA計算は一切しない
# # - ユーザーが指定した「キー点(世代:評価値)」を線形補間して
# #   Normal_GA / Strong_Perturbation_GA / New_Strong_Perturbation_GA
# #   の3本の線を描画する
# # ---------------------------------------------

# import streamlit as st
# import plotly.graph_objects as go

# st.set_page_config(page_title="GA Dummy Comparison", layout="wide")
# st.title("GA比較グラフ（ダミーデータ生成）")

# # ========== ヘルパー関数 ==========

# def parse_keypoints(s: str, max_gen: int, default_start_val: float = 600.0):
#     """
#     "1:600, 50:580, 100:560" みたいな文字列を
#     [(1,600), (50,580), (100,560)] に変換してソートして返す。
#     - 世代は 1〜max_gen の範囲にクリップ
#     - 先頭(1, ...) がなければ default_start_val で補う
#     - 末尾(max_gen, ...) がなければ最後の値を引き伸ばす
#     """
#     pts = []
#     for chunk in s.split(","):
#         chunk = chunk.strip()
#         if not chunk:
#             continue
#         if ":" not in chunk:
#             continue
#         gen_str, val_str = chunk.split(":", 1)
#         try:
#             g = int(gen_str.strip())
#             v = float(val_str.strip())
#         except ValueError:
#             continue
#         # 範囲クリップ
#         g = max(1, min(max_gen, g))
#         pts.append((g, v))

#     if not pts:
#         # 何もパースできなかった場合は「1:default_start_val, max_gen:default_start_val」
#         return [(1, default_start_val), (max_gen, default_start_val)]

#     # 世代順にソート＆重複世代は後ろ優先でマージ
#     pts.sort(key=lambda x: x[0])
#     merged = []
#     for g, v in pts:
#         if merged and merged[-1][0] == g:
#             merged[-1] = (g, v)
#         else:
#             merged.append((g, v))
#     pts = merged

#     # 先頭が1でなければ補う
#     if pts[0][0] != 1:
#         pts.insert(0, (1, default_start_val))

#     # 末尾がmax_genでなければ補う（最後の値をそのまま延長）
#     if pts[-1][0] != max_gen:
#         last_v = pts[-1][1]
#         pts.append((max_gen, last_v))

#     return pts


# def make_curve_from_keypoints(pts, max_gen: int):
#     """
#     pts: [(g1,v1), (g2,v2), ...]  (g1<g2<...)
#     各世代 t=1..max_gen について、
#     「その時点までで最後に指定されたキー点の値」をそのまま使う。
#     → 完全に段差だけのグラフ（なめらかにしない）
#     """
#     xs = list(range(1, max_gen + 1))
#     ys = []

#     idx = 0
#     for t in xs:
#         # t 以上の最初のキー点に行かないように、"最後の g_i <= t" を探す
#         while idx < len(pts) - 1 and t >= pts[idx + 1][0]:
#             idx += 1
#         g, v = pts[idx]
#         ys.append(v)

#     return xs, ys


# # ========== UI ==========

# with st.container(border=True):
#     st.subheader("基本設定")
#     max_gen = st.slider("最大世代数 (Generation)", min_value=10, max_value=1000, value=200, step=10)

# st.markdown("### 各GAのキー点を指定（形式: `世代:評価値` をカンマ区切り）")

# default_normal = "1:600, 50:580, 100:560, 200:550"
# default_kick   = "1:600, 30:590, 60:560, 120:540, 200:530"
# default_new    = "1:600, 20:580, 40:550, 80:530, 200:520"

# c1, c2, c3 = st.columns(3)
# with c1:
#     s_normal = st.text_area(
#         "Normal_GA のキー点",
#         value=default_normal,
#         height=120,
#         help="例: 1:600, 50:580, 100:560, 200:550"
#     )
# with c2:
#     s_kick = st.text_area(
#         "Strong_Perturbation_GA のキー点",
#         value=default_kick,
#         height=120,
#         help="例: 1:600, 30:590, 60:560, 120:540, 200:530"
#     )
# with c3:
#     s_new = st.text_area(
#         "New_Strong_Perturbation_GA のキー点",
#         value=default_new,
#         height=120,
#         help="例: 1:600, 20:580, 40:550, 80:530, 200:520"
#     )

# run_btn = st.button("グラフ生成", type="primary")

# # ========== グラフ生成 ==========

# if run_btn:
#     # 1. キー点をパース
#     pts_normal = parse_keypoints(s_normal, max_gen, default_start_val=600.0)
#     pts_kick   = parse_keypoints(s_kick,   max_gen, default_start_val=600.0)
#     pts_new    = parse_keypoints(s_new,    max_gen, default_start_val=600.0)

#     # 2. 線形補間で曲線を生成
#     x, y_normal = make_curve_from_keypoints(pts_normal, max_gen)
#     _, y_kick   = make_curve_from_keypoints(pts_kick,   max_gen)
#     _, y_new    = make_curve_from_keypoints(pts_new,    max_gen)

#     # 3. プロット
#     st.subheader("ダミー比較グラフ（Generation vs Evaluation_value）")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y_normal,
#         mode="lines",
#         name="Normal_GA",
#     ))
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y_kick,
#         mode="lines",
#         name="Strong_Perturbation_GA",
#     ))
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y_new,
#         mode="lines",
#         name="New_Strong_Perturbation_GA",
#     ))

#     fig.update_layout(
#         xaxis_title="Generation",
#         yaxis_title="Evaluation_value",
#         template="plotly_white",
#     )
#     st.plotly_chart(fig, use_container_width=True, key="dummy_cmp")

#     st.markdown("#### 実際に使われたキー点（整形後）")
#     st.write("Normal_GA:", pts_normal)
#     st.write("Strong_Perturbation_GA:", pts_kick)
#     st.write("New_Strong_Perturbation_GA:", pts_new)

# else:
#     st.info("キー点を調整して「グラフ生成」を押すと、ダミーの比較グラフが表示されます。")
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import io
import random
from itertools import combinations

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

    → 必ず孤立点なし（連結）になる
    """
    rng = random.Random(seed)

    # ランダムな木（ノードは 0,1,...,n-1）
    G = nx.random_tree(n=n, seed=seed)

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
col1, col2, col3 = st.columns([1, 2, 1])  # 中央を広めに

# ===== 左：設定カラム =====
with col1:
    st.subheader("パラメータ")

    n = st.number_input("ノード数", min_value=2, max_value=50, value=10)
    p = st.slider("追加の辺を張る確率 p", 0.0, 1.0, 0.3, 0.01)
    seed = st.number_input("乱数シード", min_value=0, max_value=9999, value=42)

    generate_button = st.button("グラフ生成")

# ===== ボタンが押されたらグラフ生成＆表示 =====
if generate_button:
    # グラフ生成
    G = generate_graph_without_isolates(n=int(n), p=float(p), seed=int(seed))

    # spring_layout で座標計算
    pos = nx.spring_layout(G, seed=seed)

    # -----------------------------
    # 中央：グラフ可視化（Plotly）
    # -----------------------------
    with col2:
        st.subheader("グラフ可視化")

        # Edge trace
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

        # Node trace
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
    # まだボタン押してないときは、中央と右に案内だけ出す
    with col2:
        st.info("左でパラメータを設定して「グラフ生成」ボタンを押すと、ここにグラフが表示されます。")
    with col3:
        st.info("グラフ生成後、このカラムに CSV の内容プレビューとダウンロードボタンが表示されます。")
