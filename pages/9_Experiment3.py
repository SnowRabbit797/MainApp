# app.py
# ---------------------------------------------
# Max-Cut GA 比較シミュレータ (完全版)
# ・標準GA vs 強い摂動付きGA (Kick GA)
# ・シード完全同期による公平な比較
# ・Kickロジック最適化 (エリート保存 + 破壊後の局所探索強化)
# ・詳細な推移グラフ (Best-so-far / Average / Kick Markers)
# ---------------------------------------------

import random
import time 
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# =========================
# ★ データパス設定
# =========================
# ※ ここを分析したいCSVファイルに書き換えてください
DATA_PATH = "assets/csv/G81.csv" 

# ---- ページ設定 ----
st.set_page_config(page_title="Max-Cut GA Comparison", layout="wide")
st.title("Max-Cut GA: 標準 vs 強い摂動 (Final Version)")

# ========= 1. ユーティリティ関数 =========

@st.cache_data
def load_graph_from_csv(path: str):
    """CSVからグラフを読み込む"""
    df = pd.read_csv(path)
    if not set(["source", "target"]).issubset(df.columns):
        raise ValueError("CSVに 'source','target' 列が必要です。")
    
    # 重み付きグラフ対応
    if "weight" in df.columns:
        G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr="weight")
    else:
        G = nx.from_pandas_edgelist(df, source="source", target="target")
        nx.set_edge_attributes(G, 1, "weight")
        
    # ノード番号を 0, 1, 2... にリナンバー
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping)

def calculate_cut_size(ind, G):
    """目的関数: カットサイズ (最大化)"""
    cut_val = 0
    # ind[u] は 0 or 1 (グループID)
    for u, v, data in G.edges(data=True):
        if ind[u] != ind[v]:
            cut_val += data.get("weight", 1)
    return cut_val

def one_opt_local_search(ind, G, max_iter=5, rng=None):
    """
    1-opt 局所探索 (Greedy)
    rngを受け取ることで、探索順序もシード固定する
    """
    current_ind = ind[:]
    nodes = list(G.nodes())
    
    # 探索順序をランダムにシャッフル (バイアス除去)
    if rng: rng.shuffle(nodes) 
    else: random.shuffle(nodes)
    
    for _ in range(max_iter):
        improved = False
        for u in nodes:
            my_group = current_ind[u]
            # 反転した場合の利得計算
            gain = 0
            for v in G.neighbors(u):
                w = G[u][v].get("weight", 1)
                if current_ind[v] == my_group:
                    gain += w  # 今は切れてない -> 反転で切れる (+Gain)
                else:
                    gain -= w  # 今は切れてる -> 反転で切れなくなる (-Gain)
            
            if gain > 0:
                current_ind[u] = 1 - current_ind[u]
                improved = True
        
        if not improved:
            break # 改善がなくなれば終了
            
    return current_ind

# ========= 2. GA オペレータ =========

def init_population_random(n, size=50, rng=None):
    pop = []
    for _ in range(size):
        ind = [rng.randint(0, 1) for _ in range(n)]
        pop.append(ind)
    return pop

def tournament_select(pop_eval, tournament_size=3, rng=None):
    candidates = rng.sample(pop_eval, tournament_size)
    candidates.sort(key=lambda x: x[0], reverse=True) # Max化
    return candidates[0][1]

def uniform_crossover(p1, p2, rng=None):
    c1, c2 = [], []
    for a, b in zip(p1, p2):
        if rng.random() < 0.5:
            c1.append(a); c2.append(b)
        else:
            c1.append(b); c2.append(a)
    return c1, c2

def mutate(ind, rate=0.05, rng=None):
    out = ind[:]
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = 1 - out[i]
    return out

def apply_kick_maxcut(ind, strength=0.10, rng=None):
    """
    Kick: 遺伝子の一定割合を強制反転させる
    strength: 反転率 (0.05 ~ 0.20 推奨)
    """
    kicked = ind[:]
    n = len(kicked)
    num_flips = int(n * strength)
    if num_flips == 0: num_flips = 1
    
    indices = rng.sample(range(n), num_flips)
    for idx in indices:
        kicked[idx] = 1 - kicked[idx]
    return kicked

# ========= 3. メインGAエンジン =========

def run_maxcut_ga(G, pop_size, generations, mutation_rate, seed, 
                  stagnation_limit=30, kick_strength=0.10, 
                  use_kick=True, progress_callback=None):
    
    start_time = time.time()
    # シード固定: これによりKick発動までは標準GAと全く同じ乱数系列になる
    rng = random.Random(seed) 
    n = G.number_of_nodes()

    # パラメータ設定
    num_elite = int(pop_size * 0.10) # エリート保存率
    if num_elite < 2: num_elite = 2  # 最低2体は守る
    
    # 初期化
    population = init_population_random(n, size=pop_size, rng=rng)
    # 初期個体にも軽い局所探索をかけてスタートダッシュ
    population = [one_opt_local_search(ind, G, max_iter=2, rng=rng) for ind in population]

    # 履歴データコンテナ
    history = {
        "best": [],      # Best-so-far
        "average": [],   # 平均適応度
        "kick_gen": [],  # Kick発生世代
        "kick_val": []   # Kick発生時の値
    }
    
    best_so_far = -1
    improvements = [] 
    last_improve_gen = 0
    curr_best_ind = population[0]

    for gen in range(1, generations + 1):
        if progress_callback: progress_callback(gen)

        # --- 評価 ---
        evaluated = []
        sum_fit = 0
        for ind in population:
            fit = calculate_cut_size(ind, G)
            evaluated.append((fit, ind))
            sum_fit += fit
        
        # 統計記録
        avg_fit = sum_fit / pop_size
        history["average"].append(avg_fit)

        # ソート (降順)
        evaluated.sort(key=lambda x: x[0], reverse=True)
        curr_gen_best_fit, curr_gen_best_ind = evaluated[0]

        # Best更新
        if curr_gen_best_fit > best_so_far:
            best_so_far = curr_gen_best_fit
            curr_best_ind = curr_gen_best_ind[:]
            improvements.append((gen, best_so_far))
            last_improve_gen = gen
        
        history["best"].append(best_so_far)
        
        # --- 次世代生成 ---
        next_pop = []
        is_stagnant = False
        
        if use_kick:
            is_stagnant = (gen - last_improve_gen) >= stagnation_limit

        # ============================
        # ★ Kick (強い摂動) ブランチ
        # ============================
        if use_kick and is_stagnant:
            last_improve_gen = gen # カウンタリセット
            
            # 記録
            history["kick_gen"].append(gen)
            history["kick_val"].append(best_so_far)
            
            # 1. エリート保存 (全滅を防ぐため上位はそのまま残す)
            for i in range(num_elite):
                next_pop.append(evaluated[i][1])
            
            # 2. 残りの枠を「破壊＆再構築」で埋める
            kick_base = curr_best_ind
            while len(next_pop) < pop_size:
                # 破壊 (Kick)
                kicked = apply_kick_maxcut(kick_base, strength=kick_strength, rng=rng)
                # 再構築 (強い局所探索で谷底から這い上がらせる)
                repaired = one_opt_local_search(kicked, G, max_iter=5, rng=rng)
                next_pop.append(repaired)

        # ============================
        # ★ 標準GA ブランチ
        # ============================
        else:
            # エリート保存
            for i in range(num_elite):
                if i < len(evaluated):
                    next_pop.append(evaluated[i][1])

            # 選択・交叉・変異
            while len(next_pop) < pop_size:
                p1 = tournament_select(evaluated, rng=rng)
                p2 = tournament_select(evaluated, rng=rng)
                c1, c2 = uniform_crossover(p1, p2, rng=rng)
                c1 = mutate(c1, rate=mutation_rate, rng=rng)
                next_pop.append(c1)
                if len(next_pop) < pop_size:
                    c2 = mutate(c2, rate=mutation_rate, rng=rng)
                    next_pop.append(c2)
            
            # 軽い局所探索 (Memetic Algorithm)
            next_pop = [one_opt_local_search(ind, G, max_iter=1, rng=rng) for ind in next_pop]

        population = next_pop
        
    elapsed = time.time() - start_time
    
    return {
        "best_fit": best_so_far,
        "best_ind": curr_best_ind,
        "history": history,
        "improvements": improvements,
        "elapsed": elapsed
    }

# ========= 4. 可視化関数 =========
def plot_cut_visualization(G, ind):
    """結果のグラフ構造を可視化"""
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["#636EFA" if ind[n] == 0 else "#EF553B" for n in G.nodes()]
    
    edge_x, edge_y = [], []     # Uncut
    cut_x, cut_y = [], []       # Cut (Yellow)

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if ind[u] != ind[v]:
            cut_x.extend([x0, x1, None])
            cut_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()
    # Uncut edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#ddd', width=1), name='Uncut'))
    # Cut edges
    fig.add_trace(go.Scatter(x=cut_x, y=cut_y, mode='lines', line=dict(color='#facc15', width=2), name='Cut Edge'))
    # Nodes
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()],
                             mode='markers', marker=dict(color=node_colors, size=10), name='Node'))
    
    fig.update_layout(showlegend=True, height=400, margin=dict(l=0,r=0,t=0,b=0))
    return fig

# ========= 5. メインUI =========
with st.container(border=True):
    st.subheader(f"🛠️ 実験設定 (File: {DATA_PATH})")
    
    # パラメータ設定 (6カラム)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: pop_size = st.slider("個体数", 10, 500, 50, 10)
    with c2: generations = st.slider("世代数", 10, 2000, 200, 50)
    with c3: mutation_rate = st.slider("変異率", 0.0, 1.0, 0.05, 0.01)
    with c4: stagnation_limit = st.slider("停滞判定(世代)", 5, 100, 30, 5)
    with c5: kick_strength = st.slider("Kick強度", 0.01, 0.50, 0.10, 0.01, help="破壊率。0.05-0.15推奨")
    with c6: seed = st.slider("シード値", 0, 100, 42, 1)
    
    run_btn = st.button("🚀 比較実行", type="primary")

if run_btn:
    try:
        G = load_graph_from_csv(DATA_PATH)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        st.stop()

    st.markdown("---")
    
    # 進捗バー
    total_bar = st.progress(0.0, text="待機中...")
    
    # --- 1. 標準GA ---
    def update_std(gen):
        p = (gen / generations) * 0.5
        total_bar.progress(p, text=f"標準GA 実行中... {int(p * 200)}%")
    
    res_std = run_maxcut_ga(G, pop_size, generations, mutation_rate, seed, 
                            stagnation_limit=stagnation_limit, kick_strength=kick_strength, 
                            use_kick=False, progress_callback=update_std)

    # --- 2. Kick GA ---
    def update_kick(gen):
        p = 0.5 + (gen / generations) * 0.5
        total_bar.progress(p, text=f"強い摂動付きGA 実行中... {int(p * 100)}%")
    
    res_kick = run_maxcut_ga(G, pop_size, generations, mutation_rate, seed, 
                             stagnation_limit=stagnation_limit, kick_strength=kick_strength, 
                             use_kick=True, progress_callback=update_kick)
    
    total_bar.progress(1.0, text="完了！")

    # --- 結果表示 ---
    st.subheader("📊 結果比較")

    col1, col2 = st.columns(2)
    gen_std = res_std["improvements"][-1][0] if res_std["improvements"] else 0
    gen_kick = res_kick["improvements"][-1][0] if res_kick["improvements"] else 0
    delta = res_kick["best_fit"] - res_std["best_fit"]
    
    # 色分けロジック
    delta_color = "normal"
    if delta > 0: delta_color = "inverse" # 緑
    elif delta < 0: delta_color = "off"   # 赤

    with col1:
        st.info("🔹 標準GA")
        st.metric("最良解 (Cut Size)", int(res_std["best_fit"]))
        st.metric("到達世代", f"{gen_std} gen")
        st.metric("計算時間", f"{res_std['elapsed']:.3f} s")
        # コピペ用
        with st.expander("履歴コードを表示"):
            st.code(f"ga_history_std = {res_std['improvements']}", language="python")

    with col2:
        st.success(f"💥 強い摂動付きGA (Kick強度: {kick_strength})")
        st.metric("最良解 (Cut Size)", int(res_kick["best_fit"]), delta=delta, delta_color=delta_color)
        st.metric("到達世代", f"{gen_kick} gen")
        st.metric("計算時間", f"{res_kick['elapsed']:.3f} s")
        # コピペ用
        with st.expander("履歴コードを表示"):
            st.code(f"ga_history_kick = {res_kick['improvements']}", language="python")

    # --- グラフ描画 ---
    st.subheader("📈 推移グラフ詳細")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08,
                        subplot_titles=("① 最良解 (Best-so-far)", "② 集団平均 (Average Fitness)"))

    x_axis = list(range(1, generations + 1))

    # 1. Best-so-far
    fig.add_trace(go.Scatter(x=x_axis, y=res_std["history"]["best"], 
                             mode='lines', name='標準GA (Best)', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=res_kick["history"]["best"], 
                             mode='lines', name='Kick GA (Best)', line=dict(color='red')), row=1, col=1)
    
    # 理論上限の補助線 (総重み)
    total_weight = sum([d.get("weight", 1) for u,v,d in G.edges(data=True)])
    fig.add_hline(y=total_weight, line_dash="dot", line_color="green", annotation_text="Total Weight (Upper Bound)", row=1, col=1)

    # Kick Marks on Best
    if res_kick["history"]["kick_gen"]:
        fig.add_trace(go.Scatter(
            x=res_kick["history"]["kick_gen"], 
            y=res_kick["history"]["kick_val"],
            mode='markers', name='Kick発動',
            marker=dict(symbol='x', size=12, color='black', line=dict(width=2))
        ), row=1, col=1)

    # 2. Average
    fig.add_trace(go.Scatter(x=x_axis, y=res_std["history"]["average"], 
                             mode='lines', name='標準GA (Avg)', line=dict(color='silver')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=res_kick["history"]["average"], 
                             mode='lines', name='Kick GA (Avg)', line=dict(color='orange')), row=2, col=1)
    
    # Kick Marks on Average
    kick_gen_indices = [g-1 for g in res_kick["history"]["kick_gen"]] 
    kick_avg_vals = [res_kick["history"]["average"][i] for i in kick_gen_indices]
    
    if kick_gen_indices:
        fig.add_trace(go.Scatter(
            x=res_kick["history"]["kick_gen"], 
            y=kick_avg_vals,
            mode='markers', name='Avg Drop',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=2, col=1)

    fig.update_layout(height=700, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- 最終解の可視化 ---
    with st.expander("参考: 最終解のグラフ構造可視化 (Kick GA)", expanded=False):
        if G.number_of_nodes() <= 300: # ノード数が多いと重いので制限
            st.plotly_chart(plot_cut_visualization(G, res_kick["best_ind"]), use_container_width=True)
        else:
            st.warning("ノード数が多いため可視化をスキップしました。")
