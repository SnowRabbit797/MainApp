# app.py  （streamlit run app.py）
# ---------------------------------------------
# 最小頂点被覆 GA：一様交叉＋ルーレット選択
# - mは自動（m = max{2, round(|V|^0.6 / 3)})
# - エリート1体 + superchild
# - 軽量Greedy補正＋超軽量削減（膨張抑制）
# - best-so-far 改善履歴を記録して最後に一覧表示
# - サブグラフ分割の全体図＋各サブグラフ個別図を可視化
# ---------------------------------------------

import random
import time 
import math
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# =========================
# ★ ここだけ差し替えればOK（source,target想定）
# =========================
DATA_PATH = "assets/csv/G_set1.csv"

# ---- ページ設定 ----
st.set_page_config(page_title="MVC-GA: Uniform+Roulette / auto-m / logs+viz", layout="wide")
st.subheader("サブグラフあり")

# ========= ユーティリティ =========
def load_graph_from_csv(path: str):
    df = pd.read_csv(path)
    if not set(["source", "target"]).issubset(df.columns):
        raise ValueError("CSVに 'source','target' 列が必要です。")
    G = nx.from_pandas_edgelist(df, source="source", target="target")
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping)

def greedy_correction(ind, G: nx.Graph):
    """未被覆エッジを最小限の追加で埋める"""
    n = len(ind)
    chosen = ind[:]
    uncovered = {(u, v) for u, v in G.edges() if not (chosen[u] or chosen[v])}
    if not uncovered:
        return chosen
    neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}
    while uncovered:
        best_u, best_gain = None, -1
        for u in range(n):
            if chosen[u] == 1:
                continue
            gain = 0
            for v in neighbors[u]:
                if (u, v) in uncovered or (v, u) in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_u = u
        if best_gain <= 0:
            u, v = next(iter(uncovered))
            best_u = u if random.random() < 0.5 else v
        chosen[best_u] = 1
        to_remove = []
        for (a, b) in uncovered:
            if a == best_u or b == best_u:
                to_remove.append((a, b))
        for e in to_remove:
            uncovered.discard(e)
    return chosen

def light_prune_all_neighbors_one(ind, G):
    """
    超軽量削減：bit=1 の頂点 u について、
    すべての隣接頂点が1なら u を 0 に下げる。収束まで反復。
    （被覆は壊さない／高速）
    """
    chosen = ind[:]
    changed = True
    while changed:
        changed = False
        ones = [u for u, b in enumerate(chosen) if b == 1]
        for u in ones:
            nbrs = list(G.neighbors(u))
            if not nbrs:  # 孤立点は不要
                if chosen[u] == 1:
                    chosen[u] = 0
                    changed = True
                continue
            if all(chosen[v] == 1 for v in nbrs):
                chosen[u] = 0
                changed = True
    return chosen

def init_population_random(n, size=50, ps=(0.15, 0.25, 0.40), seed=None):
    rng = random.Random(seed) if seed is not None else random
    pop = []
    for i in range(size):
        p = ps[i % len(ps)]
        ind = [1 if rng.random() < p else 0 for _ in range(n)]
        pop.append(ind)
    return pop

def fitness_size(ind):  # 小さいほど良い
    return sum(ind)

def auto_num_parts(n_nodes: int) -> int:
    return max(2, round((n_nodes ** 0.6) / 3))

def bfs_block_division(G: nx.Graph, m: int, seed: int = 1):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    m = max(2, min(int(m), n))
    base, rem = n // m, n % m
    caps = {pid: (base + 1 if pid <= rem else base) for pid in range(1, m + 1)}
    remaining = set(nodes)
    parts = {pid: [] for pid in range(1, m + 1)}

    for pid in range(1, m + 1):
        if not remaining:
            break
        start = rng.choice(list(remaining))
        q = [start]; seen = {start}
        block = [start]; remaining.remove(start)
        while q and len(block) < caps[pid]:
            u = q.pop(0)
            for v in G.neighbors(u):
                if v in remaining and v not in seen:
                    seen.add(v); block.append(v); remaining.remove(v); q.append(v)
                    if len(block) >= caps[pid]:
                        break
        parts[pid] = block

    for v in list(remaining):
        smallest = min(parts, key=lambda k: len(parts[k]))
        parts[smallest].append(v)

    return {pid: sorted(ns) for pid, ns in parts.items()}

def make_superchild(population, parts):
    """各パートで “1が最少” の個体の部分を採用して結合"""
    n = len(population[0])
    child = [0] * n
    for _, nodes in sorted(parts.items()):
        best = min(population, key=lambda ind: sum(ind[i] for i in nodes))
        for i in nodes:
            child[i] = best[i]
    return child

# ---- 選択・交叉・突然変異 ----
def roulette_select(pop_eval, rng=None):
    rng = rng or random
    weights = [1.0 / ((f + 1.0) ** 2) for f, _ in pop_eval]  # 最小化対応
    s = sum(weights)
    r = rng.uniform(0.0, s)
    acc = 0.0
    for w, (_, ind) in zip(weights, pop_eval):
        acc += w
        if acc >= r:
            return ind
    return pop_eval[-1][1]

def uniform_crossover(p1, p2, rng=None):
    rng = rng or random
    c1, c2 = [], []
    for a, b in zip(p1, p2):
        if rng.random() < 0.5:
            c1.append(a); c2.append(b)
        else:
            c1.append(b); c2.append(a)
    return c1, c2

def mutate(ind, rate=0.05, rng=None):
    rng = rng or random
    out = ind[:]
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = 1 - out[i]
    return out

# ========= メインGA（改善ログ & 可視化用情報も返す）=========
def run_ga(G, pop_size, generations, mutation_rate, seed):
    start_time = time.time()
    rng = random.Random(seed) if seed is not None else random
    n = G.number_of_nodes()
    rand_inject = max(0, int(round(0.20 * pop_size)))  # 20% ランダム注入

    # 初期集団 → 全体Greedy補正
    population = init_population_random(n, size=pop_size, seed=seed)
    population = [greedy_correction(ind, G) for ind in population]

    # m自動 → BFS分割固定
    m_parts = auto_num_parts(n)
    parts = bfs_block_division(G, m=m_parts, seed=seed)

    best_hist = []
    best_so_far = None
    improvements = []  # (gen, best_value) を記録
    bar = st.progress(0.0, text=f"準備中… m={m_parts}")

    for gen in range(1, generations + 1):
        evaluated = [(fitness_size(ind), ind) for ind in population]
        evaluated.sort(key=lambda x: x[0])
        curr_best_fit, curr_best_ind = evaluated[0]

        # 累積最良・改善ログ
        if best_so_far is None or curr_best_fit < best_so_far:
            best_so_far = curr_best_fit
            improvements.append((gen, best_so_far))
        best_hist.append(best_so_far)

        # 次世代種：エリート + superchild
        elite = curr_best_ind
        next_pop = [elite]

        superchild = make_superchild([ind for _, ind in evaluated], parts)
        superchild = greedy_correction(superchild, G)
        superchild = light_prune_all_neighbors_one(superchild, G)  # ★膨張抑制
        next_pop.append(superchild)

        # GA子（ルーレット×一様×突然変異）
        while len(next_pop) < pop_size - rand_inject:
            p1 = roulette_select(evaluated, rng=rng)
            p2 = roulette_select(evaluated, rng=rng)
            c1, c2 = uniform_crossover(p1, p2, rng=rng)
            c1 = mutate(c1, rate=mutation_rate, rng=rng)
            next_pop.append(c1)
            if len(next_pop) < pop_size - rand_inject:
                c2 = mutate(c2, rate=mutation_rate, rng=rng)
                next_pop.append(c2)

        # ランダム注入
        while len(next_pop) < pop_size:
            p = rng.choice((0.15, 0.25, 0.40))
            rnd = [1 if rng.random() < p else 0 for _ in range(n)]
            next_pop.append(rnd)

        # 次世代整合性（Greedy）＋超軽量削減
        population = [greedy_correction(ind, G) for ind in next_pop]
        population = [light_prune_all_neighbors_one(ind, G) for ind in population]

        bar.progress(gen / generations, text=f"実行中… gen={gen}/{generations} / m={m_parts}")

    bar.empty()
    evaluated = [(fitness_size(ind), ind) for ind in population]
    evaluated.sort(key=lambda x: x[0])
    end_time = time.time()
    elapsed = end_time - start_time
    return {
        "best_fit": evaluated[0][0],
        "best_ind": evaluated[0][1],
        "hist": best_hist,
        "parts": parts,
        "m": m_parts,
        "improvements": improvements, 
        "elapsed": elapsed, # 世代ごとの改善ログ
    }

# ========= サブグラフ可視化 =========
def plot_partition_overview(G, parts, seed_layout=1, label_fontsize=7):
    pos = nx.spring_layout(G, seed=seed_layout)
    palette = ["#60a5fa","#fbbf24","#34d399","#f472b6","#a78bfa","#f87171",
               "#fb7185","#22d3ee","#84cc16","#f59e0b","#c084fc","#10b981"]
    color_map = {}
    for i, (pid, nodes) in enumerate(sorted(parts.items())):
        col = palette[i % len(palette)]
        for v in nodes:
            color_map[v] = col
    node_colors = [color_map.get(v, "#cbd5e1") for v in G.nodes()]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    nx.draw_networkx_edges(G, pos, edge_color="#9ca3af", width=1.2, alpha=0.85, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=420, node_color=node_colors,
                           edgecolors="#1f2937", linewidths=1.0, ax=ax)
    labels = {v: f"{v}\n(P{next(pid for pid, ns in parts.items() if v in ns)})" for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_fontsize, ax=ax)
    ax.axis("off")
    return fig, pos

def plot_each_partition(G, parts, pos, cols=3, label_fontsize=7):
    """各サブグラフを個別に並べて表示（全体posを流用して見た目の位置関係を保つ）"""
    import math as _math
    pids = list(sorted(parts.keys()))
    r = _math.ceil(len(pids) / cols)
    c = cols
    fig, axes = plt.subplots(r, c, figsize=(6*c, 5*r))
    if r == 1 and c == 1:
        axes = [[axes]]
    elif r == 1:
        axes = [axes]
    palette = ["#60a5fa","#fbbf24","#34d399","#f472b6","#a78bfa","#f87171",
               "#fb7185","#22d3ee","#84cc16","#f59e0b","#c084fc","#10b981"]
    for idx, pid in enumerate(pids):
        ax = axes[idx // c][idx % c]
        nodes = parts[pid]
        sub = G.subgraph(nodes).copy()
        color = palette[(pid-1) % len(palette)]
        nx.draw_networkx_edges(sub, pos, edgelist=sub.edges(),
                               edge_color="#9ca3af", width=1.2, alpha=0.85, ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=sub.nodes(),
                               node_size=420, node_color=color,
                               edgecolors="#1f2937", linewidths=1.0, ax=ax)
        labels = {v: str(v) for v in sub.nodes()}
        nx.draw_networkx_labels(sub, pos, labels=labels, font_size=label_fontsize, ax=ax)
        ax.set_title(f"P{pid}（|V|={len(nodes)}）", fontsize=12, pad=6)
        ax.axis("off")
    # 余り枠を消す
    total = r*c
    for k in range(len(pids), total):
        axes[k // c][k % c].axis("off")
    fig.tight_layout()
    return fig

# ========= UI =========
with st.container(border=True):
    st.subheader(f"設定（CSV固定: {DATA_PATH}）")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pop_size = st.slider("個体数", min_value=10, max_value=1000, value=50, step=10)
    with c2:
        generations = st.slider("世代数", min_value=10, max_value=5000, value=200, step=10)
    with c3:
        mutation_rate = st.slider("突然変異率", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
    with c4:
        seed = st.slider("シード値", min_value=0, max_value=50, value=1, step=1)
    run_btn = st.button("実行", type="primary")

# ========= 実行 =========
if run_btn:
    try:
        G = load_graph_from_csv(DATA_PATH)
    except Exception as e:
        st.error(f"CSV読み込みに失敗：{e}")
        st.stop()

    res = run_ga(
        G,
        pop_size=int(pop_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        seed=int(seed),
    )

    st.subheader("結果")
    st.write(f"自動算出 m = **{res['m']}**")
    st.write(f"最良被覆サイズ（使用ノード数）: **{res['best_fit']}**")
    st.code("best_ind = " + "".join(map(str, res["best_ind"])), language="text")
    st.write(f"実行時間: **{res['elapsed']:.2f} 秒**")  # ← ここを追加！


    # 学習履歴（累積最良）
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(y=res["hist"], mode="lines", name="best-so-far"))
    fig_hist.update_layout(xaxis_title="generation", yaxis_title="best fitness (lower is better)",
                           template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ★ 改善ログを一覧表示
    if res["improvements"]:
        df_imp = pd.DataFrame(res["improvements"], columns=["世代", "best-so-far"])
        st.markdown("#### best-so-far の更新履歴")
        st.dataframe(df_imp, use_container_width=True)
    else:
        st.info("この実行では best-so-far の更新はありませんでした。")

    # サブグラフ分割の可視化（全体＋各パート）
    with st.expander("サブグラフ分割の可視化（全体）」", expanded=True):
        fig_overview, pos = plot_partition_overview(G, res["parts"], seed_layout=seed, label_fontsize=7)
        st.pyplot(fig_overview)
        lines = [f"P{pid}: {nodes}" for pid, nodes in sorted(res["parts"].items())]
        st.code("\n".join(lines), language="text")

    with st.expander("各サブグラフを個別に表示", expanded=False):
        fig_parts = plot_each_partition(G, res["parts"], pos, cols=3, label_fontsize=7)
        st.pyplot(fig_parts)

else:
    st.info("スライダーを設定して「実行」を押してください。")
