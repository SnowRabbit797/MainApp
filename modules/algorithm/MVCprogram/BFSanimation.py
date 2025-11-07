# streamlit run app.py
import streamlit as st
import math, random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

KEY = "bfs_anim"


# ========= グラフ生成（nだけ指定） =========
def make_grid_n(n: int):
    """nに近い矩形グリッドを作り、余ったノードは削る（連番ノード）"""
    n = max(4, int(n))
    r = int(math.floor(math.sqrt(n)))
    c = int(math.ceil(n / r))
    G = nx.grid_2d_graph(r, c)
    mapping = {xy: i + 1 for i, xy in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    # 余分があれば削除
    extra = len(G) - n
    if extra > 0:
        to_remove = list(sorted(G.nodes(), reverse=True))[:extra]
        G.remove_nodes_from(to_remove)
    # 連結確保（念のため）
    if not nx.is_connected(G) and len(G) > 0:
        comps = list(nx.connected_components(G))
        for i in range(len(comps)-1):
            u = next(iter(comps[i])); v = next(iter(comps[i+1]))
            G.add_edge(u, v)
    return G

def make_watts_strogatz_n(n: int, seed: int = 1):
    """k, p は自動設定（軽量＆見やすさ重視）"""
    n = max(6, int(n))
    k = max(2, int(round(n * 0.1)))
    if k % 2 == 1: k += 1  # WSは偶数kが望ましい
    k = min(k, max(2, n-1-(n-1)%2))  # 上限調整
    p = 0.10
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

def make_barabasi_albert_n(n: int, seed: int = 1):
    """m は自動設定（薄めのハブ構造）"""
    n = max(6, int(n))
    m = max(1, min(10, int(round(n * 0.05))))
    if m >= n: m = max(1, n // 5)
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return G


# ========= レイアウト（自動選択） =========
def auto_layout(G):
    n = len(G)
    if n <= 20:
        return nx.kamada_kawai_layout(G)
    elif n <= 100:
        return nx.spring_layout(G, seed=1)
    else:
        return nx.spectral_layout(G)


# ========= cap付き・ラウンド同時拡張BFS（ステップ可視化用） =========
def bfs_conquer_states_balanced(G, m=2, seed=1):
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    m = max(2, min(int(m), n))  # 安全側

    # 1) シード（重複なし）
    seeds = random.sample(nodes, m)

    # 2) cap（各パート最大サイズ）：n をほぼ等分
    base, rem = n // m, n % m
    caps = {pid: (base + 1 if pid <= rem else base) for pid in range(1, m+1)}

    # 3) 初期化
    parts  = {pid: set() for pid in range(1, m+1)}
    queues = {pid: deque() for pid in range(1, m+1)}
    for pid, s in enumerate(seeds, start=1):
        parts[pid].add(s)
        queues[pid].append(s)
    claimed = set(seeds)
    states = [{pid: set(parts[pid]) for pid in parts}]  # round 0

    # 4) ラウンド（層）ごとに同時拡張
    while True:
        proposals = {}           # v -> [pid,...]
        added_any = False
        next_queues = {pid: deque() for pid in range(1, m+1)}

        for pid in range(1, m+1):
            if len(parts[pid]) >= caps[pid]:
                continue
            width = len(queues[pid])  # 今層のみ処理
            for _ in range(width):
                u = queues[pid].popleft()
                for v in G.neighbors(u):
                    if v in claimed:
                        continue
                    if len(parts[pid]) < caps[pid]:
                        proposals.setdefault(v, []).append(pid)

        # 競合解決：サイズが小さい区画優先 → 同率ならpid昇順
        for v, cand in proposals.items():
            cand = [pid for pid in cand if len(parts[pid]) < caps[pid]]
            if not cand:
                continue
            cand.sort(key=lambda pid: (len(parts[pid]), pid))
            winner = cand[0]
            parts[winner].add(v)
            next_queues[winner].append(v)
            claimed.add(v)
            added_any = True

        queues = next_queues
        states.append({pid: set(parts[pid]) for pid in parts})

        # 収束：誰も追加できない or 全cap到達
        if (not added_any) or all(len(parts[pid]) >= caps[pid] for pid in parts):
            break

    # 5) 未割当があれば最近接シードへ（cap内）
    rest = [v for v in nodes if v not in claimed]
    if rest:
        for v in rest:
            dists = []
            for pid, s in enumerate(seeds, start=1):
                try:
                    d = nx.shortest_path_length(G, source=s, target=v)
                except nx.NetworkXNoPath:
                    d = 10**9
                dists.append((d, pid))
            dists.sort()
            for _, pid in dists:
                if len(parts[pid]) < caps[pid]:
                    parts[pid].add(v); claimed.add(v)
                    break
        states.append({pid: set(parts[pid]) for pid in parts})

    final_round = len(states) - 1
    return states, seeds, final_round, caps


# ========= 可視化ユーティリティ =========
def find_part(snapshot, v):
    for pid, nodes in snapshot.items():
        if v in nodes:
            return pid
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
    nx.draw_networkx_nodes(G, pos, nodelist=list(seeds), node_size=420,
                           node_color="none", edgecolors="#111827", linewidths=2.2, ax=ax)
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
        b1, b2 = st.columns([1,3])
        
        with b1:
            gtype = st.selectbox("グラフ種類",["グリッド", "Watts–Strogatz", "Barabási–Albert"],key=f"{KEY}_gtype",)
            n = st.number_input("ノード数 n", min_value=6, max_value=400, value=40, step=2, key=f"{KEY}_n")
            m = st.number_input("分割数 m", min_value=2, max_value=12, value=3, step=1, key=f"{KEY}_m")
            
            # グラフ生成（k, p, mなど内部自動設定・seed固定で再現性あり）
            if gtype == "グリッド":
                G = make_grid_n(int(n))
            elif gtype == "Watts–Strogatz":
                G = make_watts_strogatz_n(int(n), seed=1)
            else:  # Barabási–Albert
                G = make_barabasi_albert_n(int(n), seed=1)

            pos = auto_layout(G)

            # BFS 陣取り（ステップ列）
            states, seeds, final_round, caps = bfs_conquer_states_balanced(G, m=int(m), seed=1)

            # ステップ選択
            step = st.slider("ステップ（ラウンド）", 0, int(final_round), 0, 1, key=f"{KEY}_step")
            snapshot = states[min(step, len(states)-1)]
            
        with b2:
            # 描画
            fig, ax = plt.subplots(figsize=(7.2, 6.0))
            draw_step(ax, G, pos, snapshot, seeds, title=f" step {step}")
            st.pyplot(fig)

            # 指標とcap
            g, w, a, b = compute_metrics(G, snapshot)
            parts_text = [f"P{pid}: {sorted(nodes)}" for pid, nodes in sorted(snapshot.items())]
            st.code(" | ".join(parts_text), language="text")
