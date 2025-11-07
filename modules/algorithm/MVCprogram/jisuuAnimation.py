# streamlit module: degree round-robin partition (cap付き) のステップ可視化
# あなたが送ってくれた構成を踏襲し、ランダム→次数ロビンに置き換え

import streamlit as st
import math, random
import networkx as nx
import matplotlib.pyplot as plt

KEY = "deg_anim"  # ← ウィジェット衝突回避のキー（ページ内で一意に）

# ========= グラフ生成（nだけ指定） =========
def make_grid_n(n: int):
    n = max(4, int(n))
    r = int(math.floor(math.sqrt(n)))
    c = int(math.ceil(n / r))
    G = nx.grid_2d_graph(r, c)
    mapping = {xy: i + 1 for i, xy in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    extra = len(G) - n
    if extra > 0:
        to_remove = list(sorted(G.nodes(), reverse=True))[:extra]
        G.remove_nodes_from(to_remove)
    if not nx.is_connected(G) and len(G) > 0:
        comps = list(nx.connected_components(G))
        for i in range(len(comps)-1):
            u = next(iter(comps[i])); v = next(iter(comps[i+1]))
            G.add_edge(u, v)
    return G

def make_watts_strogatz_n(n: int, seed: int = 1):
    n = max(6, int(n))
    k = max(2, int(round(n * 0.1)))
    if k % 2 == 1: k += 1
    k = min(k, max(2, n-1-(n-1)%2))
    p = 0.10
    # 連結保証のWS
    return nx.connected_watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

def make_barabasi_albert_n(n: int, seed: int = 1):
    n = max(6, int(n))
    m = max(1, min(10, int(round(n * 0.05))))
    if m >= n: m = max(1, n // 5)
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)

# ========= レイアウト自動選択 =========
def auto_layout(G):
    n = len(G)
    if n <= 20:
        return nx.kamada_kawai_layout(G)
    elif n <= 100:
        return nx.spring_layout(G, seed=1)
    else:
        return nx.spectral_layout(G)

# ========= 次数ロビン（cap付き）ステップ生成 =========
# 1) ノードを次数降順で並べる
# 2) 各ラウンドで「サイズの小さい区画から順に」まだ未割当のノードを1つずつ獲得
# 3) cap（均等上限）に達した区画はスキップ
def degree_round_robin_states_balanced(G, m=3):
    nodes = list(G.nodes())
    n = len(nodes)
    m = max(2, min(int(m), n))

    # cap 設定（均等割り）
    base, rem = n // m, n % m
    caps = {pid: (base + 1 if pid <= rem else base) for pid in range(1, m+1)}

    # 次数降順（同率はノードID昇順っぽく）
    def _tie_id(v):
        try:
            return int(v)
        except Exception:
            return 0
    nodes_sorted = sorted(nodes, key=lambda v: (G.degree(v), _tie_id(v)), reverse=True)

    # 見た目用の“シード”（上位mノード）。分割ロジックには影響しない
    seeds = nodes_sorted[:m] if m <= len(nodes_sorted) else nodes_sorted

    # 初期化
    parts = {pid: set() for pid in range(1, m+1)}
    claimed = set()
    states = [{pid: set(parts[pid]) for pid in parts}]  # round 0（空）

    idx = 0
    while True:
        pids = sorted(parts.keys(), key=lambda pid: (len(parts[pid]), pid))
        added_any = False
        for pid in pids:
            if len(parts[pid]) >= caps[pid]:
                continue
            # 未割当の“次数の高い順リスト”から次を1つ
            while idx < len(nodes_sorted) and nodes_sorted[idx] in claimed:
                idx += 1
            if idx >= len(nodes_sorted):
                continue
            v = nodes_sorted[idx]
            parts[pid].add(v)
            claimed.add(v)
            idx += 1
            added_any = True

        states.append({pid: set(parts[pid]) for pid in parts})

        # 終了条件
        if (not added_any) or all(len(parts[pid]) >= caps[pid] for pid in parts) or len(claimed) == n:
            break

    final_round = len(states) - 1
    return states, seeds, final_round, caps

# ========= 描画ユーティリティ =========
def find_part(snapshot, v):
    for pid, nodes in snapshot.items():
        if v in nodes: return pid
    return 0

def draw_step(ax, G, pos, snapshot, seeds, title=""):
    palette = ["#60a5fa","#fbbf24","#34d399","#f472b6","#a78bfa","#f87171",
               "#fb7185","#22d3ee","#84cc16","#f59e0b","#c084fc","#10b981"]
    node_colors = []
    for v in G.nodes():
        pid = find_part(snapshot, v)
        node_colors.append("#cbd5e1" if pid == 0 else palette[(pid-1) % len(palette)])

    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8", width=1.2, alpha=0.85, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=420, node_color=node_colors,
                           edgecolors="#1f2937", linewidths=1.2, ax=ax)

    labels = {v: f"{v}\n(P{find_part(snapshot, v) or '?'} )" for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

    # シードは太枠で強調（見た目用）
    nx.draw_networkx_nodes(G, pos, nodelist=list(seeds), node_size=420,
                           node_color="none", edgecolors="#111827", linewidths=2.0, ax=ax)

    ax.set_title(title, fontsize=12, pad=6)
    ax.axis("off")

def compute_metrics(G, snapshot):
    g = w = 0
    for u, v in G.edges():
        pu, pv = find_part(snapshot, u), find_part(snapshot, v)
        if pu == 0 or pv == 0:
            continue
        if pu == pv: g += 1
        else:        w += 1
    a = g / (g + w) if (g + w) > 0 else 0.0
    sizes = [len(nodes) for nodes in snapshot.values() if nodes]
    b = (min(sizes)/max(sizes)) if sizes and max(sizes) > 0 else 0.0
    return g, w, a, b

# ========= UI（サイドバー無し・コンテナ内完結） =========
def main():
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            gtype = st.selectbox("グラフ種類", ["グリッド", "Watts–Strogatz", "Barabási–Albert"], key=f"{KEY}_gtype")
            n = st.number_input("ノード数 n", min_value=6, max_value=400, value=40, step=2, key=f"{KEY}_n")
            m = st.number_input("分割数 m", min_value=2, max_value=12, value=3, step=1, key=f"{KEY}_m")

            # グラフ生成
            if gtype == "グリッド":
                G = make_grid_n(int(n))
            elif gtype == "Watts–Strogatz":
                G = make_watts_strogatz_n(int(n), seed=1)
            else:
                G = make_barabasi_albert_n(int(n), seed=1)

            pos = auto_layout(G)

            # 次数ロビン分割のステップ列
            states, seeds, final_round, caps = degree_round_robin_states_balanced(G, m=int(m))

            # ステップ選択
            step = st.slider("ステップ（ラウンド）", 0, int(final_round), 0, 1, key=f"{KEY}_step")
            snapshot = states[min(step, len(states)-1)]

        with c2:
            # 描画
            fig, ax = plt.subplots(figsize=(7.2, 6.0))
            draw_step(ax, G, pos, snapshot, seeds, title=f"step {step}")
            st.pyplot(fig)

            # 指標と割当一覧
            g, w, a, b = compute_metrics(G, snapshot)
            parts_text = [f"P{pid}: {sorted(nodes)}" for pid, nodes in sorted(snapshot.items())]
            st.code(" | ".join(parts_text), language="text")

