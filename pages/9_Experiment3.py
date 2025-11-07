# app_baseline.py  （streamlit run app_baseline.py）
# ----------------------------------------------------
# 最小頂点被覆 GA：区間値なし・サブグラフなしのベースライン
# - 目的：使用ノード数の最小化（被覆はGreedy補正で必ず満たす）
# - 一様交叉＋ルーレット選択＋ビット反転突然変異
# - エリート1体保持、ランダム注入20%
# - best-so-far の履歴＆改善ログ＆実行時間を表示
# ----------------------------------------------------

import time
import random
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# ====== ここだけ差し替えればOK（source,target想定）======
DATA_PATH = "assets/csv/G_set1.csv"

# ---- ページ設定 ----
st.set_page_config(page_title="MVC-GA Baseline (no-interval, no-subgraph)", layout="wide")
st.subheader("サブグラフなし")

# ========= I/O & 基本ユーティリティ =========
def load_graph_from_csv(path: str):
    df = pd.read_csv(path)
    if not set(["source", "target"]).issubset(df.columns):
        raise ValueError("CSVに 'source','target' 列が必要です。")
    G = nx.from_pandas_edgelist(df, source="source", target="target")
    # 0..n-1 に詰め替え（ビット列対応）
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping)

def greedy_correction(ind, G: nx.Graph):
    """未被覆エッジを最小限の追加で埋める（被覆を保証）"""
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
    超軽量削減：bit=1 の頂点 u について、全隣接が1なら u を 0 に下げる。
    収束まで反復（被覆は壊さない／高速）。
    """
    chosen = ind[:]
    changed = True
    while changed:
        changed = False
        ones = [u for u, b in enumerate(chosen) if b == 1]
        for u in ones:
            nbrs = list(G.neighbors(u))
            if not nbrs:
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

# ========= GA演算子 =========
def roulette_select(pop_eval, rng=None):
    """最小化対応のルーレット：重み = 1/(f+1)^2"""
    rng = rng or random
    weights = [1.0 / ((f + 1.0) ** 2) for f, _ in pop_eval]
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

# ========= メインGA（ベースライン） =========
def run_ga_baseline(G, pop_size, generations, mutation_rate, seed):
    start_time = time.time()  # 計測開始
    rng = random.Random(seed) if seed is not None else random
    n = G.number_of_nodes()
    rand_inject = max(0, int(round(0.20 * pop_size)))  # 20% ランダム注入

    # 初期集団 → Greedy補正（被覆保証）
    population = init_population_random(n, size=pop_size, seed=seed)
    population = [greedy_correction(ind, G) for ind in population]

    best_hist = []
    best_so_far = None
    improvements = []  # (gen, best_value)
    bar = st.progress(0.0, text="実行中…")

    for gen in range(1, generations + 1):
        evaluated = [(fitness_size(ind), ind) for ind in population]
        evaluated.sort(key=lambda x: x[0])
        curr_best_fit, curr_best_ind = evaluated[0]

        # 累積最良（単調非増加）＆更新ログ
        if best_so_far is None or curr_best_fit < best_so_far:
            best_so_far = curr_best_fit
            improvements.append((gen, best_so_far))
        best_hist.append(best_so_far)

        # 次世代：エリート1体
        next_pop = [curr_best_ind]

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

        # ランダム注入（多様性）
        while len(next_pop) < pop_size:
            p = rng.choice((0.15, 0.25, 0.40))
            rnd = [1 if rng.random() < p else 0 for _ in range(n)]
            next_pop.append(rnd)

        # 次世代整合性：Greedy補正＋軽量削減（膨張抑制）
        population = [greedy_correction(ind, G) for ind in next_pop]
        population = [light_prune_all_neighbors_one(ind, G) for ind in population]

        bar.progress(gen / generations, text=f"実行中… gen={gen}/{generations}")

    bar.empty()
    evaluated = [(fitness_size(ind), ind) for ind in population]
    evaluated.sort(key=lambda x: x[0])

    elapsed = time.time() - start_time  # 計測終了
    return {
        "best_fit": evaluated[0][0],
        "best_ind": evaluated[0][1],
        "hist": best_hist,
        "improvements": improvements,
        "elapsed": elapsed,
    }

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
    run_btn = st.button("実行（ベースライン）", type="primary")

# ========= 実行 =========
if run_btn:
    try:
        G = load_graph_from_csv(DATA_PATH)
    except Exception as e:
        st.error(f"CSV読み込みに失敗：{e}")
        st.stop()

    res = run_ga_baseline(
        G,
        pop_size=int(pop_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        seed=int(seed),
    )

    st.subheader("結果（ベースライン）")
    st.write(f"最良被覆サイズ（使用ノード数）: **{res['best_fit']}**")
    st.write(f"実行時間: **{res['elapsed']:.2f} 秒**")
    st.code("best_ind = " + "".join(map(str, res["best_ind"])), language="text")

    # 学習履歴（累積最良）
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(y=res["hist"], mode="lines", name="best-so-far"))
    fig_hist.update_layout(xaxis_title="generation", yaxis_title="best fitness (lower is better)",
                           template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 改善ログテーブル
    if res["improvements"]:
        df_imp = pd.DataFrame(res["improvements"], columns=["世代", "best-so-far"])
        st.markdown("#### best-so-far の更新履歴")
        st.dataframe(df_imp, use_container_width=True)
    else:
        st.info("この実行では best-so-far の更新はありませんでした。")

else:
    st.info("スライダーを設定して「実行（ベースライン）」を押してください。")
