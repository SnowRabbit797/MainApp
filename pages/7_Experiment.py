# streamlit run app.py
import streamlit as st
import pandas as pd
import random
import networkx as nx
import plotly.graph_objects as go

# =========================
# ページ設定
# =========================
st.set_page_config(page_title="GA for Minimum Vertex Cover", layout="wide")
st.title("遺伝的アルゴリズムによる最小頂点被覆（MVC）")

# =========================
# ユーティリティ関数（GA本体）
# =========================
def build_graph(file_path):
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr=True)
    node = list(G.nodes())
    node_index = {u: i for i, u in enumerate(node)}
    all_edges_set = {frozenset(e) for e in G.edges()}
    neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}
    return G, node, node_index, all_edges_set, neighbors

def greedyCorrection(individual, node, node_index, all_edges_set, neighbors):
    individual = individual.copy()
    uncovered = all_edges_set.copy()
    for i, bit in enumerate(individual):
        if bit == 1:
            u = node[i]
            for v in neighbors[u]:
                uncovered.discard(frozenset((u, v)))
    while uncovered:
        scores = {}
        for e in uncovered:
            for u in e:
                if individual[node_index[u]] == 0:
                    scores[u] = scores.get(u, 0) + 1
        if not scores:
            break
        best_u = max(scores, key=scores.get)
        individual[node_index[best_u]] = 1
        for v in neighbors[best_u]:
            uncovered.discard(frozenset((best_u, v)))
    return individual

def greedyReduction(individual, node, node_index, neighbors):
    individual = individual.copy()
    ones = [i for i, bit in enumerate(individual) if bit == 1]
    random.shuffle(ones)
    for i in ones:
        individual[i] = 0
        u = node[i]
        valid = True
        for v in neighbors[u]:
            if individual[node_index[v]] == 0:
                valid = False
                break
        if not valid:
            individual[i] = 1
    return individual

def fitness(genotype, node, node_index, all_edges_set, neighbors):
    p = greedyCorrection(genotype, node, node_index, all_edges_set, neighbors)
    p = greedyReduction(p, node, node_index, neighbors)
    return sum(p)

def create_initial_population(n, size, node, node_index, all_edges_set, neighbors):
    pop = []
    super_ind = greedyCorrection([0]*n, node, node_index, all_edges_set, neighbors)
    pop.append(super_ind)
    for _ in range(size - 1):
        p = random.choice([0.2, 0.3, 0.5])
        ind = [1 if random.random() < p else 0 for _ in range(n)]
        pop.append(ind)
    return pop

def roulette_select(evaluated):
    scores = [1.0/((f+1.0)**2) for f, _ in evaluated]
    ssum = sum(scores)
    r = random.uniform(0, ssum)
    acc = 0.0
    for i, (_, g) in enumerate(evaluated):
        acc += scores[i]
        if acc >= r:
            return g
    return evaluated[-1][1]

def tournament_select(evaluated, k=3):
    cand = random.sample(evaluated, k)
    cand.sort(key=lambda x: x[0])
    return cand[0][1]

def select_parent(evaluated, p_roulette=0.5, k=3):
    return roulette_select(evaluated) if random.random() < p_roulette else tournament_select(evaluated, k)

def mutate(ind, rate):
    for i in range(len(ind)):
        if random.random() < rate:
            ind[i] = 1 - ind[i]
    return ind

def one_point_crossover(p1, p2, mut_rate):
    L = len(p1)
    c = random.randint(1, L-1)
    c1 = p1[:c] + p2[c:]
    c2 = p2[:c] + p1[c:]
    return mutate(c1, mut_rate), mutate(c2, mut_rate)

def uniform_crossover(p1, p2, mut_rate):
    c1, c2 = [], []
    for a, b in zip(p1, p2):
        if random.random() < 0.5:
            c1.append(a); c2.append(b)
        else:
            c1.append(b); c2.append(a)
    return mutate(c1, mut_rate), mutate(c2, mut_rate)

def strong_perturbation(ind, rate, node, node_index, all_edges_set, neighbors):
    new_ind = ind.copy()
    flip = max(1, int(len(ind)*rate))
    pos = random.sample(range(len(ind)), flip)
    for p in pos:
        new_ind[p] = 1 - new_ind[p]
    new_ind = greedyCorrection(new_ind, node, node_index, all_edges_set, neighbors)
    new_ind = greedyReduction(new_ind, node, node_index, neighbors)
    return new_ind

def run_ga(params, graph_ctx, use_strong, progress_label="最適化中..."):
    (G, node, node_index, all_edges_set, neighbors) = graph_ctx
    node_num = len(node)
    pop_size = params["pop_size"]
    generation = params["generation"]
    mutation_rate = params["mutation_rate"]
    elite_ratio = params["elite_ratio"]
    tournament_k = params["tournament_k"]
    p_roulette = params["p_roulette"]
    cx_type = params["cx_type"]
    stagnation_threshold = params["stagnation_threshold"]
    sp_rate = params["sp_rate"]
    sp_bottom_ratio = params["sp_bottom_ratio"]

    elite_size = max(1, int(pop_size * elite_ratio))
    cx = one_point_crossover if cx_type == "一点交叉" else uniform_crossover

    genotypes = create_initial_population(node_num, pop_size, node, node_index, all_edges_set, neighbors)
    best_hist, best_ever = [], (float('inf'), None)
    no_improve = 0
    bar = st.progress(0, text=progress_label)

    for gen in range(generation):
        evaluated = [(fitness(g, node, node_index, all_edges_set, neighbors), g) for g in genotypes]
        evaluated.sort(key=lambda x: x[0])
        cur_best_fit, cur_best_geno = evaluated[0]

        best_hist.append(min(best_hist[-1], cur_best_fit) if best_hist else cur_best_fit)

        if cur_best_fit < best_ever[0]:
            best_ever = (cur_best_fit, cur_best_geno)
            no_improve = 0
        else:
            no_improve += 1

        # ---- 強い摂動（停滞時） ----
        if use_strong and no_improve >= stagnation_threshold:
            split = max(1, int(pop_size * (1.0 - sp_bottom_ratio)))
            elites_keep = [g for _, g in evaluated[:split]]
            bottoms = [g for _, g in evaluated[split:]]
            perturbed = [strong_perturbation(g, sp_rate, node, node_index, all_edges_set, neighbors) for g in bottoms]
            genotypes = elites_keep + perturbed
            no_improve = 0
            bar.progress((gen+1)/generation, text=f"{progress_label}｜強い摂動を適用（gen={gen+1}）")
            continue

        # ---- 次世代生成 ----
        next_gen = [g for _, g in evaluated[:elite_size]]  # エリート保存
        while len(next_gen) < pop_size:
            p1 = select_parent(evaluated, p_roulette=p_roulette, k=tournament_k)
            p2 = select_parent(evaluated, p_roulette=p_roulette, k=tournament_k)
            c1, c2 = cx(p1, p2, mutation_rate)
            next_gen.append(c1)
            if len(next_gen) < pop_size:
                next_gen.append(c2)
        genotypes = next_gen

        bar.progress((gen+1)/generation, text=f"{progress_label}｜現在の最良={cur_best_fit}")

    bar.empty()
    return best_hist, best_ever

# =========================
# サイドバー（フォーム + 実行ボタン）
# =========================
with st.sidebar:
    st.header("パラメータ設定")
    with st.form("params_form", clear_on_submit=False):
        file_path = st.text_input("エッジリストCSV", "assets/csv/G_set1_small.csv")
        pop_size = st.number_input("集団サイズ", 10, 5000, 50, 10)
        generation = st.number_input("世代数", 10, 20000, 500, 10)
        mutation_rate = st.slider("突然変異率", 0.0, 1.0, 0.10, 0.01)
        elite_ratio = st.slider("エリート率", 0.0, 0.5, 0.05, 0.01)
        seed = st.number_input("乱数シード（任意）", min_value=0, max_value=10**9, value=0, step=1)

        st.markdown("---")
        use_sp = st.checkbox("強い摂動を使う", value=True)
        stagnation_threshold = st.number_input("停滞閾値（連続非改善世代）", 1, 5000, 20, 1)
        sp_rate = st.slider("摂動の反転割合", 0.01, 0.9, 0.2, 0.01)
        sp_bottom_ratio = st.slider("摂動対象（下位割合）", 0.1, 0.9, 0.5, 0.05)

        st.markdown("---")
        cx_type = st.radio("交叉法", ["一点交叉", "一様交叉"], index=0, horizontal=True)
        tournament_k = st.number_input("トーナメントサイズ k", 2, 50, 3, 1)
        p_roulette = st.slider("親選択：ルーレットの比率", 0.0, 1.0, 0.5, 0.05)

        submitted = st.form_submit_button("実行")

# =========================
# 実行・結果の保持と表示
# =========================
if submitted:
    # 乱数シード（0は固定しない。0でない時のみ固定）
    if seed != 0:
        random.seed(seed)

    try:
        graph_ctx = build_graph(file_path)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.stop()

    params = dict(
        pop_size=pop_size,
        generation=int(generation),
        mutation_rate=float(mutation_rate),
        elite_ratio=float(elite_ratio),
        tournament_k=int(tournament_k),
        p_roulette=float(p_roulette),
        cx_type=cx_type,
        stagnation_threshold=int(stagnation_threshold),
        sp_rate=float(sp_rate),
        sp_bottom_ratio=float(sp_bottom_ratio),
    )

    # 実行（GAのみ / GA+強い摂動）
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GAのみ")
        hist_ga, best_ga = run_ga(params, graph_ctx, use_strong=False, progress_label="GAのみ")
        st.write(f"最終最良: **{best_ga[0]}**")

    with col2:
        st.subheader("GA + 強い摂動")
        hist_sp, best_sp = run_ga(params, graph_ctx, use_strong=use_sp, progress_label="GA+強い摂動")
        st.write(f"最終最良: **{best_sp[0]}**")

    # 結果をセッションに保存（パラメータも一緒に）
    st.session_state["last_params"] = params
    st.session_state["hist_ga"] = hist_ga
    st.session_state["best_ga"] = best_ga
    st.session_state["hist_sp"] = hist_sp
    st.session_state["best_sp"] = best_sp

# 直近の結果表示（実行後にスライダーを動かしても残る）
if "hist_ga" in st.session_state and "hist_sp" in st.session_state:
    hist_ga = st.session_state["hist_ga"]
    hist_sp = st.session_state["hist_sp"]
    best_ga = st.session_state["best_ga"]
    best_sp = st.session_state["best_sp"]

    st.markdown("### 学習履歴（累積最良）")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist_ga, mode='lines', name='GAのみ'))
    fig.add_trace(go.Scatter(y=hist_sp, mode='lines', name='GA+強い摂動'))
    fig.update_layout(xaxis_title='generation', yaxis_title='best fitness', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # 世代スナップショット
    gen_list = [100, 200, 300, 400, 500]
    for g in gen_list:
        if g <= len(hist_ga):
            st.write(f"{g}世代: GAのみ={hist_ga[g-1]}, GA+強い摂動={hist_sp[g-1]}")
else:
    st.info("左のサイドバーでパラメータを設定し、**実行**ボタンを押してください。")
