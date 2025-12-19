import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import time
import plotly.graph_objects as go
import random


def main():
    st.sidebar.title("12/19の資料")
    section = st.sidebar.radio("目次", ["発表内容", "サブグラフ分割の変更について",  "グラフ縮約による前処理", "強い摂動の再設計", "線形計画法との比較①", "線形計画法との比較②", "次回"])

    if section == "発表内容":
        st.header("12月19日の発表")
        st.markdown("""<br>""", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.subheader("今回の発表内容", divider="orange")
            st.markdown("""
            - グラフ縮約による前処理
            - 強い摂動の再設計
            - 他のメタヒューリスティクスへの応用の検討
            """, unsafe_allow_html=True)
            
            st.image("data/image/image1217/A1648E9C-8D61-44C9-ADAF-FEFFAA8F398E.jpeg")
    

    elif section == "サブグラフ分割の変更について":
        st.title("サブグラフ分割手法の改善")

        st.markdown("""
        #### 概要
        探索性能を向上させるため、サブグラフ（ブロック）の分割方法を変更しました。
        """)

        st.markdown("""
            | | 変更前 | 変更後|
            |---|---|---|
            |制約|上限あり|上限なし|
            |起点| ランダム |ハブ (高次数ノード)|
            |メリット| 各サブグラフのサイズが均等になる | 自然なつながりが保たれる |
            |デメリット| 上限オーバーであふれた頂点が遠くのグラフへ飛ばされる| サブグラフのサイズに偏りが生じる |
            """)

        st.divider()

        # ==========================================
        #  ⚙️ グラフ設定 (ここを書き換えて形を変える)
        # ==========================================
        # プレゼンで違うパターンを見せたい場合は、ここの数値を変更して保存してください
        GRAPH_SEED = 224  

        # 設定定数
        N_NODES = 50
        M_PARTS = 5
        LIMIT_CAP = 10  # 平均ぴったり(余裕なし設定)
        # 1. グラフ生成 (ランダム幾何グラフ固定)
        G_demo = nx.random_geometric_graph(N_NODES, 0.25, seed=GRAPH_SEED)
        pos_demo = nx.get_node_attributes(G_demo, 'pos')

        # ヘルパー関数: サブグラフグリッド表示
        def display_subgraphs_grid(container_title, parts):
            st.markdown(f"#### {container_title}")
            palette = ["#60a5fa", "#fbbf24", "#34d399", "#f472b6", "#a78bfa", "#94a3b8"]
            
            sorted_pids = sorted(parts.keys())
            
            cols = st.columns(3)
            for idx, pid in enumerate(sorted_pids):
                nodes = parts[pid]
                col_idx = idx % 3
                
                with cols[col_idx]:
                    with st.container(border=True):
                        color_hex = palette[pid % len(palette)]
                        
                        st.markdown(f"**Subgraph {pid+1}**")
                        st.caption(f"頂点数: **{len(nodes)}**")
                        
                        fig, ax = plt.subplots(figsize=(3.5, 3))
                        
                        # 背景（薄く全ノード）
                        nx.draw_networkx_nodes(G_demo, pos_demo, node_color="#f3f4f6", node_size=100, ax=ax)
                        nx.draw_networkx_edges(G_demo, pos_demo, edge_color="#e5e7eb", width=0.5, ax=ax)

                        # サブグラフ強調
                        if len(nodes) > 0:
                            subG = G_demo.subgraph(nodes)
                            nx.draw_networkx_edges(subG, pos_demo, edge_color="#64748b", width=1.0, ax=ax)
                            nx.draw_networkx_nodes(subG, pos_demo, nodelist=nodes, node_color=color_hex, 
                                                    node_size=300, edgecolors="#333", ax=ax)
                            nx.draw_networkx_labels(subG, pos_demo, font_size=8, ax=ax)
                        
                        # 軸固定
                        xs, ys = zip(*pos_demo.values())
                        ax.set_xlim(min(xs)-0.05, max(xs)+0.05)
                        ax.set_ylim(min(ys)-0.05, max(ys)+0.05)
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        with st.expander("頂点リスト", expanded=False):
                            st.code(str(sorted(nodes)), language="text")

        # --- ロジック1: 変更前 (上限あり・ランダム・あふれ処理) ---
        rng_before = random.Random(GRAPH_SEED)
        seeds_before = rng_before.sample(list(G_demo.nodes()), M_PARTS)
        parts_before = {i: [s] for i, s in enumerate(seeds_before)}
        visited_before = set(seeds_before)

        queue = [(i, s) for i, s in enumerate(seeds_before)]
        idx = 0
        while idx < len(queue):
            pid, u = queue[idx]
            idx += 1
            
            if len(parts_before[pid]) >= LIMIT_CAP:
                continue

            neighbors = sorted(list(G_demo.neighbors(u)))
            for v in neighbors:
                if v not in visited_before:
                    if len(parts_before[pid]) < LIMIT_CAP:
                        visited_before.add(v)
                        parts_before[pid].append(v)
                        queue.append((pid, v))
                    else:
                        # 上限到達（ここでは入れない）
                        pass

        # あふれたノードの強制割当 (空いているところへ)
        leftovers = [n for n in G_demo.nodes() if n not in visited_before]
        for n in leftovers:
            for pid in parts_before:
                if len(parts_before[pid]) < LIMIT_CAP:
                    parts_before[pid].append(n)
                    break
            else:
                # 全て満杯なら最後のグラフへ（エラー回避）
                parts_before[M_PARTS-1].append(n)

        # --- ロジック2: 変更後 (上限なし・ハブ) ---
        sorted_nodes = sorted(G_demo.nodes(), key=lambda x: G_demo.degree[x], reverse=True)
        seeds_after = sorted_nodes[:M_PARTS]

        parts_after = {i: [s] for i, s in enumerate(seeds_after)}
        node_to_pid = {s: i for i, s in enumerate(seeds_after)}
        queues_after = {i: [s] for i, s in enumerate(seeds_after)}

        active_pids = list(range(M_PARTS))
        while active_pids:
            next_active = []
            for pid in active_pids:
                if not queues_after[pid]: continue
                u = queues_after[pid].pop(0)
                for v in sorted(list(G_demo.neighbors(u))):
                    if v not in node_to_pid:
                        node_to_pid[v] = pid
                        parts_after[pid].append(v)
                        queues_after[pid].append(v)
                if queues_after[pid]:
                    next_active.append(pid)
            active_pids = next_active

        remaining = [n for n in G_demo.nodes() if n not in node_to_pid]
        for n in remaining:
            parts_after[0].append(n)

        # ==========================
        #  可視化表示
        # ==========================


        tab1, tab2 = st.tabs(["変更前-上限あり-", "変更後-上限なし-)"])

        with tab1:
            st.markdown(f"**特徴**: 上限 ({LIMIT_CAP}個) があるため、入りきらなかった頂点が別のサブグラフへ飛ばされている。")
            display_subgraphs_grid(f"変更前の分割結果", parts_before)

        with tab2:
            st.markdown("**特徴**: 上限がないため、頂点が自然な繋がりを保ったまま分割されている。")
            display_subgraphs_grid(f"変更後の分割結果", parts_after)

    
    elif section == "グラフ縮約による前処理":
        st.header("葉(次数1の頂点)に対する強制選択ルール")
        st.markdown("""
                    #### 概要
                    次数1の頂点に隣接する頂点は最適解に必ず含まれるという性質を利用し事前にそれらを固定・削除する手法。
                    """, unsafe_allow_html=True)
        with st.container(border=True):

            st.markdown(
                """
                  - 葉(v):次数1の頂点の唯一の隣の頂点(u)をforced(確定選択)に入れる  
                  - uを選ぶとuに接続する辺は全て被覆されるので、uをグラフから削除する
                  - これを、葉が無くなるまで繰り返す
                """
            )
                
            # ==========================================
            # ★ここにCSVのパスを指定してください
            # ==========================================
            DATA_PATH2 = "assets/csv/1217random.csv" 
            # ※ CSVにはヘッダー "source", "target" が必要です
            # ==========================================
            # -------------------------
            # 1. データ読み込み関数
            # -------------------------
            def load_graph_for_demo(path):
                try:
                    df = pd.read_csv(path)
                    if "source" not in df.columns or "target" not in df.columns:
                        st.error("CSVエラー: 'source' と 'target' カラムが必要です。")
                        st.stop()
                    G = nx.from_pandas_edgelist(df, source="source", target="target")
                    return G
                except FileNotFoundError:
                    st.error(f"ファイルが見つかりません: {path}")
                    st.stop()
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")
                    st.stop()

            # -------------------------
            # 2. ステップごとの状態を事前計算
            # -------------------------
            def get_reduction_steps(G_origin):
                steps = []
                H = G_origin.copy()
                
                # Step 0: 初期状態
                steps.append({
                    "G": H.copy(),
                    "leaves": [],
                    "forced": [],
                    "title": "初期状態",
                    "message": f"ノード数: {H.number_of_nodes()}, エッジ数: {H.number_of_edges()}\nここから「次数1の頂点」を探します。"
                })

                round_num = 1
                while True:
                    leaves = [v for v, d in H.degree() if d == 1]
                    if not leaves:
                        break
                    
                    # Step A: ターゲット特定
                    neighbors = set()
                    for v in leaves:
                        if H.has_node(v):
                            # vの隣接点を取得
                            for u in H.neighbors(v):
                                neighbors.add(u)
                    
                    steps.append({
                        "G": H.copy(),
                        "leaves": list(leaves),
                        "forced": list(neighbors),
                        "title": f"ラウンド {round_num}: ターゲット特定",
                        "message": f"葉 {len(leaves)}個 ({leaves[:5]}...) を検出。\n隣接する {len(neighbors)}個 ({list(neighbors)[:5]}...) を強制選択します。\n赤色の辺は全て被覆（削除）されます。"
                    })

                    # Step B: 削除実行
                    H.remove_nodes_from(neighbors)
                    # 孤立点除去（デモを見やすくするため）
                    isolated = [v for v, d in H.degree() if d == 0]
                    if isolated:
                        H.remove_nodes_from(isolated)
                    
                    steps.append({
                        "G": H.copy(),
                        "leaves": [],
                        "forced": [],
                        "title": f"ラウンド {round_num}: 削除後",
                        "message": f"選択された頂点と接続辺が削除されました。\n残存ノード数: {H.number_of_nodes()}"
                    })
                    
                    round_num += 1

                # Final Step
                steps.append({
                    "G": H.copy(),
                    "leaves": [],
                    "forced": [],
                    "title": "縮約完了",
                    "message": f"これ以上、次数1の頂点がないため終了です。\n最終的な残存ノード数: {H.number_of_nodes()}"
                })
                
                return steps

            # -------------------------
            # 3. 可視化関数
            # -------------------------
            def plot_demo_step(G_current, pos_fixed, leaves, forced, title):
                # --- ノード設定 ---
                node_colors = []
                node_text = []
                node_sizes = []
                
                for v in G_current.nodes():
                    if v in leaves:
                        col = "#00CC96" # 緑
                        txt = "葉 (Leaf)"
                        sz = 20
                    elif v in forced:
                        col = "#EF553B" # 赤
                        txt = "強制選択 (Forced)"
                        sz = 20
                    else:
                        col = "#636EFA" # 青
                        txt = "未確定"
                        sz = 12
                    node_colors.append(col)
                    node_text.append(f"{v}<br>{txt}")
                    node_sizes.append(sz)

                # --- エッジ設定（色分け） ---
                edge_x_normal, edge_y_normal = [], []
                edge_x_alert, edge_y_alert = [], []
                
                for u, v in G_current.edges():
                    if u in pos_fixed and v in pos_fixed:
                        x0, y0 = pos_fixed[u]
                        x1, y1 = pos_fixed[v]
                        
                        is_alert = (u in forced) or (v in forced) or (u in leaves) or (v in leaves)
                        
                        if is_alert:
                            edge_x_alert.extend([x0, x1, None])
                            edge_y_alert.extend([y0, y1, None])
                        else:
                            edge_x_normal.extend([x0, x1, None])
                            edge_y_normal.extend([y0, y1, None])

                fig = go.Figure()

                # 普通の辺
                fig.add_trace(go.Scatter(
                    x=edge_x_normal, y=edge_y_normal,
                    mode="lines",
                    line=dict(width=1, color="lightgray"),
                    hoverinfo="none"
                ))

                # 注目の辺
                fig.add_trace(go.Scatter(
                    x=edge_x_alert, y=edge_y_alert,
                    mode="lines",
                    line=dict(width=3, color="#EF553B"),
                    hoverinfo="none"
                ))

                # ノード
                fig.add_trace(go.Scatter(
                    x=[pos_fixed[v][0] for v in G_current.nodes()],
                    y=[pos_fixed[v][1] for v in G_current.nodes()],
                    mode="markers+text",
                    text=[str(v) for v in G_current.nodes()],
                    textposition="top center",
                    textfont=dict(color="black", size=10),
                    hovertext=node_text,
                    hoverinfo="text",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color='white')
                    )
                ))

                fig.update_layout(
                    title=dict(text=title, font=dict(size=20)),
                    showlegend=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=500,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="white"
                )
                return fig

            # -------------------------
            # 4. Streamlit UI実装
            # -------------------------

            # CSVから読み込んだグラフのキャッシュ管理
            # キーにファイルパスを含めることで、パスを変えたら再読み込みされるようにする
            session_key_pos = f"demo_pos_{DATA_PATH2}"
            session_key_steps = f"demo_steps_{DATA_PATH2}"
            session_key_idx = f"demo_idx_{DATA_PATH2}"

            if session_key_steps not in st.session_state:
                with st.spinner("CSV読み込み中..."):
                    G_demo = load_graph_for_demo(DATA_PATH2)
                    # レイアウト固定 (kの値で広がりを調整可能)
                    st.session_state[session_key_pos] = nx.spring_layout(G_demo, seed=42, k=0.15, iterations=50)
                    st.session_state[session_key_steps] = get_reduction_steps(G_demo)
                    st.session_state[session_key_idx] = 0

            # --- UI ---
            col_ctrl, col_msg = st.columns([1, 2])

            # コントローラー
            with col_ctrl:
                
                if st.button("⏮ 最初に戻す"):
                    st.session_state[session_key_idx] = 0
                
                steps_data = st.session_state[session_key_steps]
                total_steps = len(steps_data)
                current_idx = st.session_state[session_key_idx]
                
                is_last = current_idx >= total_steps - 1
                

                st.write(f"Step: {current_idx} / {total_steps-1}")

            # メッセージ表示
            current_data = steps_data[st.session_state[session_key_idx]]
            with col_msg:
                if st.button("次へ進む ➡", type="primary", disabled=is_last):
                    st.session_state[session_key_idx] += 1
                    st.rerun()

                st.write(f"Step: {current_idx} / {total_steps-1}")

            # グラフ描画
            fig = plot_demo_step(
                current_data["G"],
                st.session_state[session_key_pos],
                current_data["leaves"],
                current_data["forced"],
                current_data["title"]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div style="display: flex; gap: 20px; font-size: 0.9em;">
                <div><span style="color:#00CC96">●</span> <b>葉</b></div>
                <div><span style="color:#EF553B">●</span> <b>強制選択</b></div>
                <div><span style="color:#EF553B">━</span> <b>削除される辺</b></div>
            </div>
            """, unsafe_allow_html=True)

        
        
        
        DATA_PATH = "assets/csv/G_set1_small.csv"
        # -------------------------
        # ユーティリティ
        # -------------------------
        def load_graph_from_csv(path: str) -> nx.Graph:
            df = pd.read_csv(path)
            # source,target が無い場合に備えて最低限のチェック
            if "source" not in df.columns or "target" not in df.columns:
                raise ValueError("CSVに 'source' と 'target' 列が必要です。")
            G = nx.from_pandas_edgelist(df, source="source", target="target")
            return G

        def leaf_forcing_reduction(G: nx.Graph):
            """
            - 葉（次数1）v があるとき、その隣 u は forced に入れてよい。
            - forced に入れた頂点 u はグラフから削除（uに接続する辺は全被覆済み扱い）。
            - これを葉が無くなるまで繰り返す。

            Returns:
                forced_set: set
                reduced_G: nx.Graph
                rounds: int（何回ループしたか）
                history: list of dict（各ラウンドの統計）
            """
            H = G.copy()
            forced = set()
            rounds = 0
            history = []

            while True:
                # 現在の葉
                leaves = [v for v, d in H.degree() if d == 1]
                if not leaves:
                    break

                rounds += 1
                # 葉の隣接頂点（重複ありうる）
                neighbors_of_leaves = set()
                for v in leaves:
                    # degree==1 なので隣は1個
                    u = next(iter(H.neighbors(v)))
                    neighbors_of_leaves.add(u)

                # forcedに追加
                forced |= neighbors_of_leaves

                # 削除前の統計
                history.append({
                    "round": rounds,
                    "num_leaves": len(leaves),
                    "num_forced_added": len(neighbors_of_leaves),
                    "nodes_before": H.number_of_nodes(),
                    "edges_before": H.number_of_edges(),
                })

                # forcedにした頂点を削除
                H.remove_nodes_from(neighbors_of_leaves)

                # ついでに孤立点（次数0）は縮約上、残しても意味が薄いので消す（必要ならOFFにしてOK）
                isolated = [v for v, d in H.degree() if d == 0]
                if isolated:
                    H.remove_nodes_from(isolated)

                history[-1].update({
                    "nodes_after": H.number_of_nodes(),
                    "edges_after": H.number_of_edges(),
                    "isolated_removed": len(isolated),
                })

            return forced, H, rounds, history

        def plot_graph(G: nx.Graph, forced=set(), leaves=set(), title=""):
            if G.number_of_nodes() == 0:
                fig = go.Figure()
                fig.update_layout(title=title + "（空グラフ）")
                return fig

            pos = nx.spring_layout(G, seed=1)  # 発表用に固定
            # edges
            edge_x, edge_y = [], []
            for a, b in G.edges():
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                name="辺",
                hoverinfo="none"
            )

            # nodes
            node_x, node_y, node_text, node_group = [], [], [], []
            for v in G.nodes():
                x, y = pos[v]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{v} (deg={G.degree(v)})")

                if v in forced:
                    node_group.append("確定選択")
                elif v in leaves:
                    node_group.append("葉っぱ")
                else:
                    node_group.append("その他")

            # グループごとに分けて描画（凡例を出すため）
            fig = go.Figure()
            fig.add_trace(edge_trace)

            for group_name in ["確定選択", "葉っぱ", "その他"]:
                idx = [i for i, g in enumerate(node_group) if g == group_name]
                fig.add_trace(go.Scatter(
                    x=[node_x[i] for i in idx],
                    y=[node_y[i] for i in idx],
                    mode="markers",
                    marker=dict(size=10),
                    name=group_name,
                    text=[node_text[i] for i in idx],
                    hoverinfo="text"
                ))

            fig.update_layout(
                title=title,
                showlegend=True,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            return fig

        # -------------------------
        # 章1：入力
        # -------------------------
        with st.container(border=True):
            st.subheader("1. 入力グラフ")
            colL, colR = st.columns([1, 1])
            with colL:
                st.markdown("例。")
                path = DATA_PATH

            try:
                t0 = time.time()
                G = load_graph_from_csv(path)
                load_time = time.time() - t0

                with colR:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric("ノード数 |V|", G.number_of_nodes())
                    with col2:
                        st.metric("エッジ数 |E|", G.number_of_edges())
                    with col3:
                        st.metric("読み込み時間（秒）", f"{load_time:.3f}")

                # 現在の葉
                leaves0 = {v for v, d in G.degree() if d == 1}

                colA, colB = st.columns([2, 1])
                with colA:
                    fig0 = plot_graph(G, forced=set(), leaves=leaves0, title="入力グラフ（leaf表示）")
                    st.plotly_chart(fig0, use_container_width=True)
                with colB:
                    st.write("**この時点の葉（次数1）**")
                    st.write(f"{len(leaves0)} 個")
                    st.write(list(leaves0)[:30])

            except Exception as e:
                st.error(f"読み込みに失敗しました: {e}")
                st.stop()

        # -------------------------
        # 章2：縮約（葉ルール）
        # -------------------------
        with st.container(border=True):
            st.subheader("2. 縮約後グラフ")


            t1 = time.time()
            forced, G_red, rounds, history = leaf_forcing_reduction(G)
            red_time = time.time() - t1

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("反復回数", rounds)
            col2.metric("forced（確定頂点数）", len(forced))
            col3.metric("縮約後ノード数", G_red.number_of_nodes())
            col4.metric("縮約後エッジ数", G_red.number_of_edges())
            st.caption(f"縮約処理時間: {red_time:.3f} 秒")

            # forced と leaf の可視化（入力側：forcedになった頂点も色付けして見せる）
            leaves_in_input = {v for v, d in G.degree() if d == 1}
            fig_in = plot_graph(G, forced=forced, leaves=leaves_in_input, title="縮約前グラフ")
            st.plotly_chart(fig_in, use_container_width=True)

            # 縮約後グラフ
            leaves_red = {v for v, d in G_red.degree() if d == 1}
            fig_red = plot_graph(G_red, forced=set(), leaves=leaves_red, title="縮約後グラフ")
            st.plotly_chart(fig_red, use_container_width=True)







    elif section == "線形計画法との比較①":
        st.header("線形計画法（LP）と現状GAの比較")
        st.caption("G_set に対する最小頂点被覆問題（MVC）の解法比較")

        # =========================
        # 対象データ（まずは G13）
        # =========================
        DATA_NAME = "G_set G13"
        DATA_PATH = "assets/csv/G13.csv"   # ← 相対パスは環境に合わせて

        # =========================
        # LP / GA の結果（手入力）
        # =========================
        lp_result = {
            "objective": 416,
            "cpu_time": 1152.60,
            "wall_time": 1155.82,
            "nodes": 37576,
            "iterations": 8397360,
        }

        ga_history = [
            (1, 467),
            (2, 452),
            (5, 448),
            (6, 440),
            (8, 437),
            (9, 434),
            (10, 429),
            (12, 422),
            (15, 419),
            (17, 417),
            (38, 416),
        ]

        ga_best = 416
        ga_best_gen = 38
        ga_time = 47.87


        # =========================
        # グラフ読み込み
        # =========================
        with st.container(border=True):
            st.subheader("① 入力グラフ（G13）")

            try:
                t0 = time.time()
                df = pd.read_csv(DATA_PATH)
                G = nx.from_pandas_edgelist(df, source="source", target="target")
                load_time = time.time() - t0

                col1, col2, col3 = st.columns(3)
                col1.metric("ノード数 |V|", G.number_of_nodes())
                col2.metric("エッジ数 |E|", G.number_of_edges())
                col3.metric("読込時間 [sec]", f"{load_time:.3f}")

                pos = nx.spring_layout(G, seed=1)
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    mode="lines",
                    name="Edges",
                    hoverinfo="none"
                )

                node_x, node_y = zip(*[pos[v] for v in G.nodes()])
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers",
                    name="Vertices",
                    marker=dict(size=6),
                    hoverinfo="none"
                )

                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    title="G13 グラフ構造",
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"グラフの読み込みに失敗しました: {e}")
                st.stop()

        # =========================
        # LP 結果
        # =========================
        with st.container(border=True):
            st.subheader("② 線形計画法（LP）の結果（厳密解）")

            col1, col2, col3 = st.columns(3)
            col1.metric("最適値 |VC|", lp_result["objective"])
            col2.metric("CPU時間 [sec]", f"{lp_result['cpu_time']:.1f}")
            col3.metric("Wall時間 [sec]", f"{lp_result['wall_time']:.1f}")

            st.markdown(
                f"""
                - Optimal solution found
                - Enumerated nodes: {lp_result['nodes']}
                - Total iterations: {lp_result['iterations']:,}
                - 解は 厳密最適解
                
                CPU時間はソルバが実際に計算していた時間の合計で、
                Wall時間はユーザが待つ実時間のこと。
                """
            )

        # =========================
        # GA 結果
        # =========================
        with st.container(border=True):
            st.subheader("③ 遺伝的アルゴリズム（GA）の結果")

            col1, col2, col3 = st.columns(3)
            col1.metric("最良解 |VC|", ga_best)
            col2.metric("到達世代", ga_best_gen)
            col3.metric("計算時間 [sec]", f"{ga_time:.2f}")  # ← 追加

            # GA 履歴の可視化
            gens = [g for g, _ in ga_history]
            vals = [v for _, v in ga_history]

            ga_trace = go.Scatter(
                x=gens,
                y=vals,
                mode="lines+markers",
                name="GA best-so-far"
            )

            fig_ga = go.Figure(data=[ga_trace])
            fig_ga.update_layout(
                title="GA による最良解の推移（G13）",
                xaxis_title="Generation",
                yaxis_title="Vertex Cover Size",
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig_ga, use_container_width=True)

            st.markdown(
                f"""
        - {ga_best_gen} 世代・{ga_time:.2f} 秒で LP の最適値 {ga_best} に到達  
        - LP（約 {lp_result['wall_time']:.0f} 秒）と比べ、計算時間は約 {lp_result['wall_time']/ga_time:.1f} 倍高速 
        - 厳密性は保証されないが、大規模問題に対して実用的
                """
            )


        # =========================
        # 考察（発表用）
        # =========================
        with st.container(border=True):
            st.subheader("④ 結果G13ver（LP vs GA）")

            st.markdown(
                """
                    強い摂動を発動する間もなく、GAがLPの厳密最適解に到達してしまった。
                    
                    なので、もう一つ違うデータセットで比較してみる必要がある。
                """
            )
            
#-------------------------

    elif section == "線形計画法との比較②":
        st.header("線形計画法（LP）と現状GAの比較")
        st.caption("G_set に対する最小頂点被覆問題（MVC）の解法比較")

        # =========================
        # 対象データ（まずは G13）
        # =========================
        DATA_NAME = "G_set G2"
        DATA_PATH = "assets/csv/G13.csv"   # ← 相対パスは環境に合わせて

        # =========================
        # LP / GA の結果（手入力）
        # =========================
        lp_result = {
            "best_solution": 727,          # 現時点で見つかっている最良の整数解（上界）
            "best_possible": 580.84742,    # LP緩和＋カットによる下界（bound）
            "elapsed_time": 3822.04,       # 計算経過時間 [sec]       # 未処理で探索木に残っているノード数
        }


        ga_history = [
            (1, 722),
            (2, 719),
            (16, 717),
            (47, 715),
            (76, 714),
            (106, 713),
            (109, 711),
            (144, 710)
        ]

        ga_best = 710
        ga_best_gen = 144
        ga_time = 1524.48


        # =========================
        # グラフ読み込み
        # =========================
        with st.container(border=True):
            st.subheader("① 入力グラフ（G2）")

            col1, col2 = st.columns(2)
            col1.metric("ノード数 |V|", 800)
            col2.metric("エッジ数 |E|", 19176)
            
            st.write("読み込みに時間がかかるため、描画はしないことにする")
        # =========================
        # LP 結果
        # =========================
        with st.container(border=True):
            st.subheader("② 線形計画法(LP)の結果")

            # 相対Gapの計算
            gap_rel = (
                lp_result["best_solution"] - lp_result["best_possible"]
            ) / lp_result["best_solution"]

            col1, col2, col3 = st.columns(3)
            col1.metric("最良解 |VC|（上界）", lp_result["best_solution"])
            col2.metric("下界 best possible", f'{lp_result["best_possible"]:.3f}')
            col3.metric("計算時間 [sec]", f'{lp_result["elapsed_time"]:.1f}')

            st.markdown(
                f"""
        - CBC ソルバにより分枝限定法（Branch & Bound）を用いて探索を実施
        - 現時点で得られている最良の整数解（上界）は {lp_result["best_solution"]}
        - best possible = {lp_result["best_possible"]:.5f} は、
          LP 緩和およびカットによって得られた下界（bound）
        - 0/1 制約を一度ゆるめて連続問題として解き、明らかに不自然な解を排除する制約を加えることで、最適値の理論的な下限を求めている。
        - よって真の最適値(opt)は
          {lp_result["best_possible"]:.5f} ≤ opt ≤ {lp_result["best_solution"]}
        - 相対 Gap は {gap_rel*100:.1f} %(最適解がどこにあるかを示す“区間の幅”が、まだ20%残っている)

        - 最良値727から全く改善できていないまま止まっていたので、3822秒で打ち切り終了した。
                """
            )


        # =========================
        # GA 結果
        # =========================
        with st.container(border=True):
            st.subheader("③ 遺伝的アルゴリズム（GA）の結果")

            col1, col2, col3 = st.columns(3)
            col1.metric("最良解 |VC|", ga_best)
            col2.metric("到達世代", ga_best_gen)
            col3.metric("計算時間 [sec]", f"{ga_time:.2f}")  # ← 追加

            # GA 履歴の可視化
            gens = [g for g, _ in ga_history]
            vals = [v for _, v in ga_history]

            ga_trace = go.Scatter(
                x=gens,
                y=vals,
                mode="lines+markers",
                name="GA best-so-far"
            )

            fig_ga = go.Figure(data=[ga_trace])
            fig_ga.update_layout(
                title="GA による最良解の推移（G13）",
                xaxis_title="Generation",
                yaxis_title="Vertex Cover Size",
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig_ga, use_container_width=True)

            st.markdown(
                f"""
            - GAでは20世代以上の停滞を検知したタイミングで強い摂動を発動し、局所最適化から抜け出して解の改善に向かう挙動が確認できた。
            - グラフ前処理が探索空間の無駄を減らし、サブグラフ分割によって局所探索が効率化され、強い摂動も効果的に働いた結果、かなり良好な解が得られた。
            """
            )

            
            
    elif section == "次回":
        st.header("次回の予定")

        st.write("次回は、GA（遺伝的アルゴリズム）に「強い摂動」を組み合わせて、TSPで試してみようと思う。")
        st.write("局所解から抜け出すための処理を入れることで、どれくらい探索性能が良くなるか有効性を検証したい。")
        st.write("普通のGAとの比較実験をする")
