# # app.py  （streamlit run app.py）
# # ---------------------------------------------
# # 最小頂点被覆 GA 比較ツール
# # - Normal_GA（通常GA）
# # - Strong_Perturbation_GA（強い摂動付きGA）
# # - New_Strong_Pertubation_GA（前処理付き・強い摂動GA）
# #
# # 共通：
# # - 一様交叉＋ルーレット選択
# # - mは自動（m = max{2, round(|V|^0.6 / 3)})
# # - エリート1体 + superchild
# # - 軽量Greedy補正＋超軽量削減（膨張抑制）
# # - best-so-far 改善履歴を記録して一覧表示
# # - サブグラフ分割の全体図＋各サブグラフ個別図を可視化（各モードごと）
# # ---------------------------------------------

# import random
# import time
# import math
# import streamlit as st
# import pandas as pd
# import networkx as nx
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt

# # =========================
# # ★ ここだけ差し替えればOK（source,target想定）
# # =========================
# DATA_PATH = "assets/csv/G_set1_small.csv"

# # ---- ページ設定 ----
# st.set_page_config(
#     page_title="MVC-GA: Normal vs Strong Perturbation",
#     layout="wide"
# )
# st.title("最小頂点被覆 GA 比較（Normal / Strong / New-Strong）")

# # ========= ユーティリティ =========
# def load_graph_from_csv(path: str):
#     df = pd.read_csv(path)
#     if not set(["source", "target"]).issubset(df.columns):
#         raise ValueError("CSVに 'source','target' 列が必要です。")
#     G = nx.from_pandas_edgelist(df, source="source", target="target")
#     # ノードラベルを 0,1,2,... に詰める
#     mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
#     return nx.relabel_nodes(G, mapping)


# def greedy_correction(ind, G: nx.Graph, leaf_forbidden=None):
#     """
#     未被覆エッジを最小限の追加で埋める。
#     leaf_forbidden が与えられた場合：
#       - そのノードは極力 1 にしないようにする
#     """
#     n = len(ind)
#     chosen = ind[:]

#     # 葉ノードを禁止したい場合は、最初に強制的に0に戻す
#     if leaf_forbidden:
#         for u in leaf_forbidden:
#             if 0 <= u < n:
#                 chosen[u] = 0

#     # uncovered は frozenset({u,v}) で管理（順序の問題を回避）
#     uncovered = {
#         frozenset({u, v}) for u, v in G.edges()
#         if not (chosen[u] or chosen[v])
#     }
#     if not uncovered:
#         return chosen

#     neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}
#     degrees = dict(G.degree())

#     while uncovered:
#         best_u = None
#         best_gain = -1
#         best_key = None

#         for u in range(n):
#             if chosen[u] == 1:
#                 continue
#             if leaf_forbidden and u in leaf_forbidden:
#                 # 葉ノードは原則スキップ（本当に必要になったら後で選ぶ）
#                 continue

#             gain = 0
#             for v in neighbors[u]:
#                 if frozenset({u, v}) in uncovered:
#                     gain += 1

#             if gain > 0:
#                 # tie-breaker: gain優先, 同じgainなら「次数が低いノード」を優先
#                 key = (gain, -degrees.get(u, 0))
#                 if best_key is None or key > best_key:
#                     best_key = key
#                     best_gain = gain
#                     best_u = u

#         if best_gain <= 0:
#             # もう葉禁止では埋まらない → 葉ノードも含めてランダムに選ぶ
#             e = next(iter(uncovered))
#             u, v = tuple(e)
#             best_u = u if random.random() < 0.5 else v

#         chosen[best_u] = 1

#         # best_u がカバーしたエッジを uncovered から削除
#         to_remove = []
#         for e in uncovered:
#             a, b = tuple(e)
#             if a == best_u or b == best_u:
#                 to_remove.append(e)
#         for e in to_remove:
#             uncovered.discard(e)

#     return chosen


# def light_prune_all_neighbors_one(ind, G):
#     """
#     超軽量削減：
#       bit=1 の頂点 u について、
#       - すべての隣接頂点が1なら u を 0 に下げる。
#       - 孤立点(次数0)なら 0 に下げる。
#     収束まで反復。被覆は壊さない。
#     """
#     chosen = ind[:]
#     changed = True
#     while changed:
#         changed = False
#         ones = [u for u, b in enumerate(chosen) if b == 1]
#         for u in ones:
#             nbrs = list(G.neighbors(u))
#             if not nbrs:  # 孤立点は不要
#                 if chosen[u] == 1:
#                     chosen[u] = 0
#                     changed = True
#                 continue
#             # 隣接が全部1なら、自分は0に下げてOK
#             if all(chosen[v] == 1 for v in nbrs):
#                 chosen[u] = 0
#                 changed = True
#     return chosen


# def init_population_random(n, size=50, ps=(0.15, 0.25, 0.40), seed=None):
#     rng = random.Random(seed) if seed is not None else random
#     pop = []
#     for i in range(size):
#         p = ps[i % len(ps)]
#         ind = [1 if rng.random() < p else 0 for _ in range(n)]
#         pop.append(ind)
#     return pop


# def fitness_size(ind):
#     """使用ノード数（小さいほど良い）"""
#     return sum(ind)


# def auto_num_parts(n_nodes: int) -> int:
#     """サブグラフ数 m を自動決定"""
#     return max(2, round((n_nodes ** 0.6) / 3))


# def bfs_block_division(G: nx.Graph, m: int, seed: int = 1):
#     """
#     高次数ノードをシードにした並行BFSでサブグラフ分割。
#     各ブロックの最大サイズ制約はあえて入れず、自然な塊ごとに分割する。
#     """
#     from collections import deque

#     rng = random.Random(seed)
#     nodes = list(G.nodes())
#     n = len(nodes)
#     m = max(2, min(int(m), n))

#     # --- シード選定：次数の高い順に上位 m 個 ---
#     nodes_sorted = sorted(nodes, key=lambda x: G.degree[x], reverse=True)
#     seeds = nodes_sorted[:m]

#     parts = {pid: [] for pid in range(1, m + 1)}
#     queues = {}
#     node_to_pid = {}

#     # シードを初期配置
#     for i, s in enumerate(seeds):
#         pid = i + 1
#         parts[pid].append(s)
#         node_to_pid[s] = pid
#         queues[pid] = deque([s])

#     # --- 並行BFS（上限なし） ---
#     active_pids = list(range(1, m + 1))
#     while active_pids:
#         rng.shuffle(active_pids)
#         next_active = []
#         for pid in active_pids:
#             q = queues[pid]
#             if not q:
#                 continue
#             u = q.popleft()
#             for v in G.neighbors(u):
#                 if v not in node_to_pid:
#                     node_to_pid[v] = pid
#                     parts[pid].append(v)
#                     q.append(v)
#             if q:
#                 next_active.append(pid)
#         active_pids = next_active

#     # --- 残余ノード処理（非連結成分など） ---
#     remaining = [v for v in nodes if v not in node_to_pid]
#     for v in remaining:
#         adj_pids = set()
#         for nbr in G.neighbors(v):
#             if nbr in node_to_pid:
#                 adj_pids.add(node_to_pid[nbr])

#         if adj_pids:
#             # 隣接しているパートの中で、ノード数が最小のところへ
#             target = min(adj_pids, key=lambda pid: len(parts[pid]))
#         else:
#             # 完全孤立なら全体で最小サイズのパートへ
#             target = min(parts, key=lambda pid: len(parts[pid]))

#         parts[target].append(v)
#         node_to_pid[v] = target

#     return {pid: sorted(ns) for pid, ns in parts.items()}


# # ---- 選択・交叉・突然変異 ----
# def roulette_select(pop_eval, rng=None):
#     """
#     pop_eval: [(fitness, individual), ...]
#     最小化問題なので w = 1/(f+1)^2 を重みとしてルーレット選択。
#     """
#     rng = rng or random
#     weights = [1.0 / ((f + 1.0) ** 2) for f, _ in pop_eval]
#     s = sum(weights)
#     r = rng.uniform(0.0, s)
#     acc = 0.0
#     for w, (_, ind) in zip(weights, pop_eval):
#         acc += w
#         if acc >= r:
#             return ind
#     return pop_eval[-1][1]


# def uniform_crossover(p1, p2, rng=None):
#     rng = rng or random
#     c1, c2 = [], []
#     for a, b in zip(p1, p2):
#         if rng.random() < 0.5:
#             c1.append(a)
#             c2.append(b)
#         else:
#             c1.append(b)
#             c2.append(a)
#     return c1, c2


# def mutate(ind, rate=0.05, rng=None):
#     rng = rng or random
#     out = ind[:]
#     for i in range(len(out)):
#         if rng.random() < rate:
#             out[i] = 1 - out[i]
#     return out


# # ========= 強い摂動（Kick） =========
# def apply_strong_perturbation(ind, G, strength=0.2, rng=None):
#     """
#     強い摂動:
#       - ランダムに選んだノードの一部に対して
#           * 0に強制する
#           * ビット反転する
#         をランダムに行う。
#     """
#     rng = rng or random
#     out = ind[:]
#     n = len(out)

#     num_targets = max(1, int(n * strength))
#     idxs = rng.sample(range(n), num_targets)

#     for i in idxs:
#         mode = rng.choice(["zero", "flip"])
#         if mode == "zero":
#             out[i] = 0
#         else:
#             out[i] = 1 - out[i]

#     return out


# # ========= メインGA（Normal / Strong / New-Strong） =========
# def run_ga(G, pop_size, generations, mutation_rate, seed, mode="normal"):
#     """
#     mode:
#       - "normal"    : 強い摂動なし (Normal_GA)
#       - "kick"      : 強い摂動あり (Strong_Perturbation_GA)
#       - "kick_leaf" : 強い摂動 + 葉ノード回避前処理 (New_Strong_Pertubation_GA)
#     """
#     start_time = time.time()
#     rng = random.Random(seed) if seed is not None else random
#     n = G.number_of_nodes()

#     use_kick = mode in ("kick", "kick_leaf")
#     avoid_leaf = mode == "kick_leaf"

#     # 葉ノード集合（avoid_leaf のときだけ使う）
#     leaf_forbidden = set()
#     if avoid_leaf:
#         leaf_forbidden = {v for v in G.nodes() if G.degree[v] == 1}

#     # 停滞判定パラメータ
#     stagnation_limit = 30
#     kick_strength = 0.20

#     num_elite = int(pop_size * 0.30)
#     num_ga = int(pop_size * 0.50)

#     # 初期集団
#     population = init_population_random(n, size=pop_size, seed=seed)
#     population = [
#         greedy_correction(ind, G, leaf_forbidden=leaf_forbidden)
#         for ind in population
#     ]
#     population = [
#         light_prune_all_neighbors_one(ind, G)
#         for ind in population
#     ]

#     # サブグラフ情報（superchild 用）
#     m_parts = auto_num_parts(n)
#     parts = bfs_block_division(G, m=m_parts, seed=seed)
#     best_local_genes = {pid: (float('inf'), []) for pid in parts}

#     best_hist = []
#     best_so_far = None
#     improvements = []
#     last_improve_gen = 0

#     bar = st.progress(
#         0.0,
#         text=f"準備中… m={m_parts} / mode={mode}"
#     )

#     for gen in range(1, generations + 1):
#         # 評価
#         evaluated = []
#         for ind in population:
#             fit = fitness_size(ind)
#             evaluated.append((fit, ind))
#             for pid, nodes in parts.items():
#                 local_score = sum(ind[i] for i in nodes)
#                 if local_score < best_local_genes[pid][0]:
#                     best_local_genes[pid] = (local_score, [ind[i] for i in nodes])

#         evaluated.sort(key=lambda x: x[0])
#         curr_best_fit, curr_best_ind = evaluated[0]

#         # ベスト更新
#         if best_so_far is None or curr_best_fit < best_so_far:
#             best_so_far = curr_best_fit
#             improvements.append((gen, best_so_far))
#             last_improve_gen = gen

#         best_hist.append(best_so_far)

#         # 停滞判定
#         stagnation_count = gen - last_improve_gen
#         is_stagnant = use_kick and (stagnation_count >= stagnation_limit)

#         next_pop = []

#         # =========================
#         # 強い摂動モード
#         # =========================
#         if is_stagnant:
#             last_improve_gen = gen

#             # エリート1体だけ残す
#             next_pop.append(curr_best_ind)

#             # 残りは best_ind ベースに強い摂動をかけて補正
#             base = curr_best_ind
#             while len(next_pop) < pop_size:
#                 kicked = apply_strong_perturbation(
#                     base, G, strength=kick_strength, rng=rng
#                 )
#                 repaired = greedy_correction(
#                     kicked, G, leaf_forbidden=leaf_forbidden
#                 )
#                 repaired = light_prune_all_neighbors_one(repaired, G)
#                 next_pop.append(repaired)

#         # =========================
#         # 通常モード
#         # =========================
#         else:
#             # 1. Superchild（各サブグラフ局所ベストの寄せ集め）
#             superchild = [0] * n
#             for pid, nodes in parts.items():
#                 genes = best_local_genes[pid][1]
#                 if not genes:
#                     genes = [0] * len(nodes)
#                 for k, node_idx in enumerate(nodes):
#                     superchild[node_idx] = genes[k]
#             superchild = greedy_correction(
#                 superchild, G, leaf_forbidden=leaf_forbidden
#             )
#             superchild = light_prune_all_neighbors_one(superchild, G)
#             next_pop.append(superchild)

#             # 2. Elite
#             for i in range(num_elite):
#                 if i < len(evaluated):
#                     next_pop.append(evaluated[i][1])

#             # 3. GA 個体
#             target = len(next_pop) + num_ga
#             while len(next_pop) < target:
#                 p1 = roulette_select(evaluated, rng=rng)
#                 p2 = roulette_select(evaluated, rng=rng)
#                 c1, c2 = uniform_crossover(p1, p2, rng=rng)
#                 c1 = mutate(c1, rate=mutation_rate, rng=rng)
#                 next_pop.append(c1)
#                 if len(next_pop) < target:
#                     c2 = mutate(c2, rate=mutation_rate, rng=rng)
#                     next_pop.append(c2)

#             # 4. ランダム個体
#             while len(next_pop) < pop_size:
#                 p = rng.choice((0.15, 0.25, 0.40))
#                 rnd = [1 if rng.random() < p else 0 for _ in range(n)]
#                 next_pop.append(rnd)

#         # 共通：補正＋削減
#         population = [
#             greedy_correction(ind, G, leaf_forbidden=leaf_forbidden)
#             for ind in next_pop
#         ]
#         population = [
#             light_prune_all_neighbors_one(ind, G)
#             for ind in population
#         ]

#         # 進捗表示
#         if use_kick:
#             status_text = (
#                 "【KICK発動中💥】"
#                 if is_stagnant
#                 else f"通常モード (停滞: {stagnation_count})"
#             )
#         else:
#             status_text = "Normal_GA（KICKなし）"

#         bar.progress(
#             gen / generations,
#             text=f"mode={mode} | gen={gen} | Best={best_so_far} | {status_text}",
#         )

#     bar.empty()
#     evaluated = [(fitness_size(ind), ind) for ind in population]
#     evaluated.sort(key=lambda x: x[0])

#     return {
#         "mode": mode,
#         "best_fit": evaluated[0][0],
#         "best_ind": evaluated[0][1],
#         "hist": best_hist,
#         "parts": parts,
#         "m": m_parts,
#         "improvements": improvements,
#         "elapsed": time.time() - start_time,
#     }


# # ========= サブグラフ可視化 =========
# def plot_partition_overview(G, parts, seed_layout=1, label_fontsize=7):
#     pos = nx.spring_layout(G, seed=seed_layout)
#     palette = [
#         "#60a5fa", "#fbbf24", "#34d399", "#f472b6",
#         "#a78bfa", "#f87171", "#fb7185", "#22d3ee",
#         "#84cc16", "#f59e0b", "#c084fc", "#10b981"
#     ]
#     color_map = {}
#     for i, (pid, nodes) in enumerate(sorted(parts.items())):
#         col = palette[i % len(palette)]
#         for v in nodes:
#             color_map[v] = col
#     node_colors = [color_map.get(v, "#cbd5e1") for v in G.nodes()]

#     fig, ax = plt.subplots(figsize=(6.8, 5.8))
#     nx.draw_networkx_edges(
#         G, pos, edge_color="#9ca3af", width=1.2, alpha=0.85, ax=ax
#     )
#     nx.draw_networkx_nodes(
#         G, pos,
#         node_size=420, node_color=node_colors,
#         edgecolors="#1f2937", linewidths=1.0, ax=ax,
#     )
#     labels = {
#         v: f"{v}\n(P{next(pid for pid, ns in parts.items() if v in ns)})"
#         for v in G.nodes()
#     }
#     nx.draw_networkx_labels(
#         G, pos, labels=labels, font_size=label_fontsize, ax=ax
#     )
#     ax.axis("off")
#     return fig, pos


# def plot_each_partition(G, parts, pos, cols=3, label_fontsize=7):
#     """各サブグラフを個別に並べて表示（全体posを流用して見た目の位置関係を保つ）"""
#     import math as _math
#     pids = list(sorted(parts.keys()))
#     r = _math.ceil(len(pids) / cols)
#     c = cols
#     fig, axes = plt.subplots(r, c, figsize=(6 * c, 5 * r))
#     if r == 1 and c == 1:
#         axes = [[axes]]
#     elif r == 1:
#         axes = [axes]
#     palette = [
#         "#60a5fa", "#fbbf24", "#34d399", "#f472b6",
#         "#a78bfa", "#f87171", "#fb7185", "#22d3ee",
#         "#84cc16", "#f59e0b", "#c084fc", "#10b981"
#     ]
#     for idx, pid in enumerate(pids):
#         ax = axes[idx // c][idx % c]
#         nodes = parts[pid]
#         sub = G.subgraph(nodes).copy()
#         color = palette[(pid - 1) % len(palette)]
#         nx.draw_networkx_edges(
#             sub, pos, edgelist=sub.edges(),
#             edge_color="#9ca3af", width=1.2, alpha=0.85, ax=ax
#         )
#         nx.draw_networkx_nodes(
#             sub, pos, nodelist=sub.nodes(),
#             node_size=420, node_color=color,
#             edgecolors="#1f2937", linewidths=1.0, ax=ax
#         )
#         labels = {v: str(v) for v in sub.nodes()}
#         nx.draw_networkx_labels(
#             sub, pos, labels=labels, font_size=label_fontsize, ax=ax
#         )
#         ax.set_title(f"P{pid}（|V|={len(nodes)}）", fontsize=12, pad=6)
#         ax.axis("off")

#     # 余り枠を消す
#     total = r * c
#     for k in range(len(pids), total):
#         axes[k // c][k % c].axis("off")

#     fig.tight_layout()
#     return fig


# # ========= UI =========
# with st.container(border=True):
#     st.subheader(f"設定（CSV固定: {DATA_PATH}）")
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         pop_size = st.slider("個体数", min_value=10, max_value=1000, value=50, step=10)
#     with c2:
#         generations = st.slider("世代数", min_value=10, max_value=5000, value=200, step=10)
#     with c3:
#         mutation_rate = st.slider("突然変異率", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
#     with c4:
#         seed = st.slider("シード値", min_value=0, max_value=50, value=1, step=1)
#     run_btn = st.button("実行（3種類まとめて）", type="primary")

# # ========= 実行 =========
# if run_btn:
#     try:
#         G = load_graph_from_csv(DATA_PATH)
#     except Exception as e:
#         st.error(f"CSV読み込みに失敗：{e}")
#         st.stop()

#     # 3種類まとめて実行（同じパラメータ・同じseed）
#     res_normal = run_ga(
#         G,
#         pop_size=int(pop_size),
#         generations=int(generations),
#         mutation_rate=float(mutation_rate),
#         seed=int(seed),
#         mode="normal",
#     )
#     res_kick = run_ga(
#         G,
#         pop_size=int(pop_size),
#         generations=int(generations),
#         mutation_rate=float(mutation_rate),
#         seed=int(seed),
#         mode="kick",
#     )
#     res_new = run_ga(
#         G,
#         pop_size=int(pop_size),
#         generations=int(generations),
#         mutation_rate=float(mutation_rate),
#         seed=int(seed),
#         mode="kick_leaf",
#     )

#     # --- 結果サマリ ---
#     st.subheader("結果サマリ")
#     summary_df = pd.DataFrame(
#         [
#             ["Normal_GA",                  res_normal["best_fit"], res_normal["elapsed"]],
#             ["Strong_Perturbation_GA",     res_kick["best_fit"],   res_kick["elapsed"]],
#             ["New_Strong_Pertubation_GA",  res_new["best_fit"],    res_new["elapsed"]],
#         ],
#         columns=["Variant", "Best_Evaluation_value", "Elapsed(sec)"],
#     )
#     st.dataframe(summary_df, use_container_width=True)

#     # ===== 比較グラフ（Generation vs Evaluation_value）=====
#     st.subheader("GA比較グラフ（Generation vs Evaluation_value）")
#     x = list(range(1, len(res_normal["hist"]) + 1))
#     fig_cmp = go.Figure()
#     fig_cmp.add_trace(go.Scatter(
#         x=x,
#         y=res_normal["hist"],
#         mode="lines",
#         name="Normal_GA",
#     ))
#     fig_cmp.add_trace(go.Scatter(
#         x=x,
#         y=res_kick["hist"],
#         mode="lines",
#         name="Strong_Perturbation_GA",
#     ))
#     fig_cmp.add_trace(go.Scatter(
#         x=x,
#         y=res_new["hist"],
#         mode="lines",
#         name="New_Strong_Pertubation_GA",
#     ))

#     fig_cmp.update_layout(
#         xaxis_title="Generation",
#         yaxis_title="Evaluation_value",
#         template="plotly_white",
#     )
#     st.plotly_chart(fig_cmp, use_container_width=True)

#     # ===== 各モードの詳細（タブ）=====
#     st.subheader("各GAの詳細")
#     tab_norm, tab_kick, tab_new = st.tabs(
#         ["Normal_GA", "Strong_Perturbation_GA", "New_Strong_Pertubation_GA"]
#     )

#     def show_detail(tab, res, label, seed_layout):
#         with tab:
#             st.markdown(f"### {label}")
#             st.write(f"自動算出 m = **{res['m']}**")
#             st.write(f"最良被覆サイズ（使用ノード数）: **{res['best_fit']}**")
#             st.code(
#                 "best_ind = " + "".join(map(str, res["best_ind"])),
#                 language="text",
#             )
#             st.write(f"実行時間: **{res['elapsed']:.2f} 秒**")

#             # 学習履歴（累積最良）
#             fig_hist = go.Figure()
#             x_local = list(range(1, len(res["hist"]) + 1))
#             fig_hist.add_trace(go.Scatter(
#                 x=x_local,
#                 y=res["hist"],
#                 mode="lines",
#                 name="best-so-far",
#             ))
#             fig_hist.update_layout(
#                 xaxis_title="generation",
#                 yaxis_title="best fitness (lower is better)",
#                 template="plotly_white",
#             )
#             st.plotly_chart(
#                 fig_hist,
#                 use_container_width=True,
#                 key=f"{label}_hist",   # ★ここ追加
#             )

#             # 以下はそのまま
#             if res["improvements"]:
#                 df_imp = pd.DataFrame(
#                     res["improvements"],
#                     columns=["世代", "best-so-far"],
#                 )
#                 st.markdown("#### best-so-far の更新履歴")
#                 st.dataframe(df_imp, use_container_width=True)
#             else:
#                 st.info("この実行では best-so-far の更新はありませんでした。")

#             with st.expander("サブグラフ分割の可視化（全体）", expanded=False):
#                 fig_overview, pos = plot_partition_overview(
#                     G,
#                     res["parts"],
#                     seed_layout=seed_layout,
#                     label_fontsize=7
#                 )
#                 st.pyplot(fig_overview)
#                 lines = [
#                     f"P{pid}: {nodes}"
#                     for pid, nodes in sorted(res["parts"].items())
#                 ]
#                 st.code("\n".join(lines), language="text")

#             with st.expander("各サブグラフを個別に表示", expanded=False):
#                 fig_parts = plot_each_partition(
#                     G, res["parts"], pos, cols=3, label_fontsize=7
#                 )
#                 st.pyplot(fig_parts)


#     show_detail(tab_norm, res_normal, "Normal_GA", seed_layout=seed)
#     show_detail(tab_kick,  res_kick,  "Strong_Perturbation_GA", seed_layout=seed)
#     show_detail(tab_new,   res_new,   "New_Strong_Pertubation_GA", seed_layout=seed)

# else:
#     st.info("スライダーを設定して「実行（3種類まとめて）」を押してください。")
