# # app.py
# # ============================
# # Streamlit + PuLP で
# # 線形計画法による最大カット問題 (Max-Cut)
# # ============================

# import time
# import pandas as pd
# import streamlit as st
# import networkx as nx
# import plotly.graph_objects as go
# import pulp

# # ====================================
# # 設定
# # ====================================
# st.set_page_config(
#     page_title="最大カット問題（線形計画法 / PuLP）",
#     layout="wide",
# )

# st.title("線形計画法（PuLP）による最大カット問題")

# st.markdown(
#     """
# **概要**
# - PuLP (CBC Solver) を用いて最大カット問題を整数計画法として解きます。
# - 頂点を2つのグループ（0と1）に分割し、グループ間を結ぶエッジ数を最大化します。
# """
# )

# # ★★ グラフ CSV の相対パス ★★
# DATA_PATH = "assets/csv/G15.csv"

# # ====================================
# # ユーティリティ
# # ====================================
# @st.cache_data
# def load_graph(path: str):
#     """CSV から networkx グラフを構築"""
#     df = pd.read_csv(path)
#     # 重み列があれば読み込む、なければ重み1とする
#     if "weight" in df.columns:
#         G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr="weight")
#     else:
#         G = nx.from_pandas_edgelist(df, source="source", target="target")
#         nx.set_edge_attributes(G, 1, "weight")
#     return G, df

# def solve_max_cut_lp(G: nx.Graph):
#     """
#     PuLP を用いて最大カット問題を解く
#     Maximize sum(w_uv * z_uv)
#     s.t.
#        z_uv <= x_u + x_v
#        z_uv <= 2 - (x_u + x_v)
#        x_v in {0, 1} (所属グループ)
#        z_uv in {0, 1} (エッジがカットされたか)
#     """
#     nodes = list(G.nodes())
#     edges = list(G.edges(data=True)) # data=Trueで重み取得

#     # 問題の定義（最大化問題）
#     prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)

#     # 変数定義
#     # x[i]: 頂点 i がグループ1に属するか (0 or 1)
#     x = pulp.LpVariable.dicts("x", nodes, cat="Binary")
    
#     # z[(u,v)]: エッジ (u,v) がカットされたか (0 or 1)
#     # エッジキーをタプルで管理
#     edge_keys = [(u, v) for u, v, d in edges]
#     z = pulp.LpVariable.dicts("z", edge_keys, cat="Binary")

#     # 目的関数：カットされたエッジの重み合計の最大化
#     prob += pulp.lpSum([d.get("weight", 1) * z[(u, v)] for u, v, d in edges])

#     # 制約条件
#     # z_uv は x_u と x_v が異なるときのみ 1 になれる
#     # 論理式 (x_u XOR x_v) の線形化
#     for u, v, d in edges:
#         prob += z[(u, v)] <= x[u] + x[v]
#         prob += z[(u, v)] <= 2 - (x[u] + x[v])

#     # 求解 & 時間計測
#     start_wall = time.time()
#     start_cpu = time.process_time()
    
#     # logPathを指定してログを残すことも可能
#     solver = pulp.PULP_CBC_CMD(msg=True) 
#     prob.solve(solver)
    
#     end_cpu = time.process_time()
#     end_wall = time.time()

#     cpu_time = end_cpu - start_cpu
#     wall_time = end_wall - start_wall

#     status = pulp.LpStatus[prob.status]
#     obj_value = pulp.value(prob.objective)

#     # 結果の取得
#     # x_v = 1 の頂点リスト（グループ1）、それ以外はグループ0
#     group1_nodes = [v for v in nodes if pulp.value(x[v]) > 0.5]
    
#     return {
#         "status": status,
#         "objective": obj_value,
#         "group1_nodes": group1_nodes,
#         "cpu_time": cpu_time,
#         "wall_time": wall_time
#     }

# def make_plotly_graph_maxcut(G: nx.Graph, pos: dict, group1_nodes=None, title="グラフ可視化"):
#     """
#     Max-Cut 用の可視化
#     - ノード: グループによって色分け (青 vs 赤)
#     - エッジ: カットされたエッジ（グループをまたぐエッジ）を強調表示
#     """
#     if group1_nodes is None:
#         group1_set = set()
#     else:
#         group1_set = set(group1_nodes)

#     # --- エッジ描画 ---
#     # 通常のエッジ（カットされていない）と、カットエッジを分けて描画
#     uncut_x, uncut_y = [], []
#     cut_x, cut_y = [], []

#     for u, v in G.edges():
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
        
#         # 両方のノードが異なるグループなら「カットエッジ」
#         # (片方が group1_set にあり、もう片方がない)
#         u_in = u in group1_set
#         v_in = v in group1_set
        
#         if u_in != v_in:
#             cut_x += [x0, x1, None]
#             cut_y += [y0, y1, None]
#         else:
#             uncut_x += [x0, x1, None]
#             uncut_y += [y0, y1, None]

#     # カットされていないエッジ（薄いグレー）
#     trace_uncut = go.Scatter(
#         x=uncut_x, y=uncut_y, mode="lines",
#         line=dict(width=1, color="#dddddd"), hoverinfo="none", name="Uncut Edges"
#     )
    
#     # カットされたエッジ（黄色・太線）
#     trace_cut = go.Scatter(
#         x=cut_x, y=cut_y, mode="lines",
#         line=dict(width=2, color="#facc15"), hoverinfo="none", name="Cut Edges"
#     )

#     # --- ノード描画 ---
#     node_x, node_y = [], []
#     node_color = []
#     node_text = []

#     for n in G.nodes():
#         x, y = pos[n]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(str(n))
        
#         if n in group1_set:
#             node_color.append("#EF553B") # 赤 (Group 1)
#         else:
#             node_color.append("#636EFA") # 青 (Group 0)

#     node_trace = go.Scatter(
#         x=node_x, y=node_y, mode="markers+text",
#         text=node_text, textposition="top center", hoverinfo="text",
#         marker=dict(size=18, color=node_color, line=dict(width=2, color="white")),
#         name="Nodes"
#     )

#     fig = go.Figure(data=[trace_uncut, trace_cut, node_trace])
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
#     st.error(f"ファイル `{DATA_PATH}` が見つかりません。")
#     st.stop()

# col1, col2 = st.columns([2, 1])
# with col1:
#     st.dataframe(df_edges, height=150)
# with col2:
#     st.write(f"- ノード数: {G.number_of_nodes()}")
#     st.write(f"- エッジ数: {G.number_of_edges()}")

# pos = nx.spring_layout(G, seed=42)

# st.subheader("2. 線形計画法で Max-Cut を解く")

# if st.button("計算開始 (PuLP)"):
#     with st.spinner("計算中..."):
#         result = solve_max_cut_lp(G)

#     status = result["status"]
#     objective = result["objective"]
#     cpu_time = result["cpu_time"]
#     wall_time = result["wall_time"]
    
#     # 指定されたフォーマットでの出力
#     st.markdown("### 計算結果")
    
#     # テキスト形式（コピペ用）
#     output_text = f"""
#     最適値: {int(objective)}
#     CPU時間 [sec]: {cpu_time:.6f}
#     Wall時間 [sec]: {wall_time:.6f}
#     Optimal solution found: {status == 'Optimal'}
#     """
#     st.code(output_text, language="text")

#     # 視覚的なメトリクス
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Max Cut Score", int(objective))
#     c2.metric("Wall Time", f"{wall_time:.4f}s")
#     c3.metric("Status", status)

#     st.markdown("### 3. 可視化（黄色線がカットされたエッジ）")
#     fig = make_plotly_graph_maxcut(G, pos, result["group1_nodes"], title="Max-Cut 解の可視化")
#     st.plotly_chart(fig, use_container_width=True)

# else:
#     st.info("ボタンを押すと計算を開始します。")
#     fig0 = make_plotly_graph_maxcut(G, pos, None, title="元のグラフ")
#     st.plotly_chart(fig0, use_container_width=True)
