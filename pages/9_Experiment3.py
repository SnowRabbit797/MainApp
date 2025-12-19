# import time
# import pandas as pd
# import streamlit as st
# import networkx as nx
# import plotly.graph_objects as go
# import pulp
# import threading # 追加: 並列処理用
# import os        # 追加: ファイル操作用

# # ====================================
# # 設定
# # ====================================
# st.set_page_config(
#     page_title="最小頂点被覆（線形計画法 / PuLP）",
#     layout="wide",
# )

# st.title("線形計画法（PuLP）による最小頂点被覆問題")

# st.markdown(
#     """
# **概要**
# - ログをリアルタイムで表示しながら計算します。
# - 計算実行中にソルバーの挙動（Branch & Boundの様子など）を確認できます。
# """
# )

# DATA_PATH = "assets/csv/G2.csv"
# LOG_FILE = "solver_log.txt" # ログファイルの一時保存先

# # ====================================
# # ユーティリティ
# # ====================================
# @st.cache_data
# def load_graph(path: str):
#     df = pd.read_csv(path)
#     G = nx.from_pandas_edgelist(df, source="source", target="target")
#     return G, df

# def build_mvc_problem(G: nx.Graph):
#     """
#     PuLPの問題オブジェクトと変数を構築して返す（まだ解かない）
#     """
#     nodes = list(G.nodes())
#     edges = list(G.edges())

#     # 問題の定義（最小化問題）
#     prob = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)

#     # 0-1 変数 x_v
#     x = pulp.LpVariable.dicts("x", nodes, cat="Binary")

#     # 目的関数
#     prob += pulp.lpSum([x[v] for v in nodes])

#     # 制約
#     for u, v in edges:
#         prob += x[u] + x[v] >= 1, f"cover_edge_{u}_{v}"
    
#     return prob, x, nodes

# def run_solver_in_thread(prob, log_path):
#     """
#     別スレッドで実行するためのソルバー関数
#     msg=True にし、logPathを指定してファイルに書き込ませる
#     """
#     if os.path.exists(log_path):
#         os.remove(log_path) # 古いログを削除
    
#     # logPathを指定することで、標準出力ではなくファイルに書き込まれる
#     solver = pulp.PULP_CBC_CMD(msg=True, logPath=log_path)
#     prob.solve(solver)

# def make_plotly_graph(G: nx.Graph, pos: dict, cover_nodes=None, title="グラフ可視化"):
#     # (元のコードと同じため省略しませんが、長くなるのでそのまま使ってください)
#     if cover_nodes is None:
#         cover_nodes = []
#     cover_nodes = set(cover_nodes)

#     edge_x = []
#     edge_y = []
#     for u, v in G.edges():
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         edge_x += [x0, x1, None]
#         edge_y += [y0, y1, None]

#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y, mode="lines",
#         line=dict(width=1), hoverinfo="none", name="Edges",
#     )

#     node_x = []
#     node_y = []
#     node_color = []
#     node_text = []

#     for n in G.nodes():
#         x, y = pos[n]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(str(n))
#         if n in cover_nodes:
#             node_color.append("tomato")
#         else:
#             node_color.append("lightgray")

#     node_trace = go.Scatter(
#         x=node_x, y=node_y, mode="markers+text",
#         text=node_text, textposition="top center", hoverinfo="text",
#         marker=dict(size=18, color=node_color, line=dict(width=1)),
#         name="Nodes",
#     )

#     fig = go.Figure(data=[edge_trace, node_trace])
#     fig.update_layout(
#         title=title, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#     )
#     return fig

# # ====================================
# # メイン処理
# # ====================================
# st.subheader("1. グラフの読み込み")

# try:
#     G, df_edges = load_graph(DATA_PATH)
# except FileNotFoundError:
#     st.error(f"ファイルが見つかりません: {DATA_PATH}")
#     st.stop()

# col1, col2 = st.columns([2, 1])
# with col1:
#     st.dataframe(df_edges, height=150)
# with col2:
#     st.write(f"- ノード数: {G.number_of_nodes()}")
#     st.write(f"- エッジ数: {G.number_of_edges()}")

# pos = nx.spring_layout(G, seed=0)

# st.subheader("2. 線形計画法で最小頂点被覆を求める")

# # ログ表示用のプレースホルダーを作成（ここにリアルタイムで書き込まれる）
# log_area = st.empty()

# if st.button("線形計画法で MVC を解く"):
    
#     # 1. 問題の構築
#     prob, x, nodes = build_mvc_problem(G)
    
#     # 2. ソルバーを別スレッドで開始
#     #    メインスレッドがブロックされると画面更新ができないため
#     t = threading.Thread(target=run_solver_in_thread, args=(prob, LOG_FILE))
#     start_time = time.time()
#     t.start()
    
#     # 3. ソルバーが動いている間、ログファイルを読み込んで画面を更新
#     st.info("計算中... 以下のログが更新されます")
    
#     # コンテナを作って少しリッチに表示
#     with st.container():
#         while t.is_alive():
#             if os.path.exists(LOG_FILE):
#                 with open(LOG_FILE, "r") as f:
#                     log_content = f.read()
#                     # 最新の行が見えるように表示（行数が多い場合はスクロールバーが出る）
#                     log_area.code(log_content, language="text")
#             time.sleep(0.2) # 0.2秒ごとに更新
            
#     # スレッド終了を待機
#     t.join()
#     elapsed = time.time() - start_time

#     # 4. 最終ログの表示と結果取得
#     if os.path.exists(LOG_FILE):
#         with open(LOG_FILE, "r") as f:
#             log_area.code(f.read(), language="text")

#     status = pulp.LpStatus[prob.status]
#     obj_value = pulp.value(prob.objective)
#     cover_nodes = [v for v in nodes if pulp.value(x[v]) > 0.5]

#     st.success("計算完了！")

#     # --- 結果表示 ---
#     st.markdown("### 結果概要")
#     col_a, col_b, col_c = st.columns(3)
#     with col_a:
#         st.metric("最小頂点被覆のサイズ", len(cover_nodes))
#     with col_b:
#         st.metric("目的関数値", int(obj_value) if obj_value is not None else "-")
#     with col_c:
#         st.metric("計算時間 [秒]", f"{elapsed:.4f}")

#     st.write(f"ソルバの状態: **{status}**")

#     st.markdown("### 3. グラフ可視化")
#     fig = make_plotly_graph(G, pos, cover_nodes=cover_nodes, title="結果")
#     st.plotly_chart(fig, use_container_width=True)

# else:
#     st.info("ボタンを押すと計算を開始します。")
#     fig0 = make_plotly_graph(G, pos, cover_nodes=None, title="元のグラフ")
#     st.plotly_chart(fig0, use_container_width=True)
