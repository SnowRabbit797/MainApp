# streamlit run app.py
import streamlit as st
import random, math
import networkx as nx
import matplotlib.pyplot as plt

# ---- 文字化け回避（使えるフォントがあれば優先的に使う）----
for f in ["IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic"]:
    try:
        plt.rcParams["font.family"] = f
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ===== デモ用グラフ（8頂点・2クラスター＋橋） =====
def make_demo_graph(seed=0):
    random.seed(seed)
    G = nx.Graph()
    left = [1,2,3,4]; right = [5,6,7,8]
    G.add_nodes_from(left + right)
    # 左の密なつながり
    G.add_edges_from([(1,2),(2,3),(3,4),(4,1),(1,3)])
    # 右の密なつながり
    G.add_edges_from([(5,6),(6,7),(7,8),(8,5),(6,8)])
    # 左右の橋
    G.add_edge(4,5)
    return G

def fixed_layout(G):
    pos = {}
    n = len(G.nodes())
    for i, v in enumerate(sorted(G.nodes())):
        ang = 2*math.pi * (i/n)
        pos[v] = (math.cos(ang), math.sin(ang))
    # 左右へ寄せる
    for v in [1,2,3,4]:
        x,y = pos[v]; pos[v] = (x-0.45, y)
    for v in [5,6,7,8]:
        x,y = pos[v]; pos[v] = (x+0.45, y)
    return pos

# ===== 分割手法 =====
def partition_random(G, m=2, seed=0):
    """完全ランダム + 非空保証のみ（偏る可能性あり）"""
    random.seed(seed)
    parts = {i: [] for i in range(1, m+1)}
    for v in G.nodes():
        k = random.randint(1, m)
        parts[k].append(v)
    # 非空に調整
    empties = [k for k, vs in parts.items() if not vs]
    for k in empties:
        src = max(parts, key=lambda kk: len(parts[kk]))
        if parts[src]:
            parts[k].append(parts[src].pop())
    return parts

def partition_degree_round_robin(G, m=2):
    """次数降順→ラウンドロビン（サイズ偏りを抑える）"""
    parts = {i: [] for i in range(1, m+1)}
    nodes_sorted = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    i = 1
    for v in nodes_sorted:
        parts[i].append(v)
        i = 1 if i == m else i+1
    return parts

def partition_bfs_blocks(G, m=2, seed=0):
    """BFSで近接ノードをまとまりに（構造を反映）"""
    random.seed(seed)
    remaining = set(G.nodes())
    parts = {i: [] for i in range(1, m+1)}
    target = math.ceil(len(G)/m)
    i = 1
    while remaining and i <= m:
        start = random.choice(list(remaining))
        block = []
        for v in nx.bfs_tree(G, start):
            if v in remaining:
                block.append(v); remaining.remove(v)
            if len(block) >= target:
                break
        parts[i] = block; i += 1
    # 余りを少ないパートへ
    for v in list(remaining):
        smallest = min(parts, key=lambda k: len(parts[k]))
        parts[smallest].append(v)
    return parts

# ===== 描画 =====
def find_part(parts, v):
    for pid, nodes in parts.items():
        if v in nodes: return pid
    return "?"

def draw_partitioned(ax, G, parts, pos, title=""):
    palette = ["#60a5fa","#fbbf24","#34d399","#f472b6","#a78bfa","#f87171"]
    color_map = {}
    for i, (pid, nodes) in enumerate(parts.items()):
        col = palette[i % len(palette)]
        for v in nodes:
            color_map[v] = col
    node_colors = [color_map[v] for v in G.nodes()]
    nx.draw_networkx_edges(G, pos, width=1.6, edge_color="#9ca3af", alpha=0.85, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="#334155", linewidths=1.6, ax=ax)
    labels = {v: f"{v}\n(P{find_part(parts, v)})" for v in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11, ax=ax)
    ax.set_title(title, fontsize=12, pad=6)
    ax.axis("off")

# ===== ここから「1つのコンテナ内で完結」 =====
def main():
  with st.container(border=True):
      st.markdown("### グラフ分割の比較（ランダム / 次数ロビン / BFS）")
      # ← ここに日本語を書けば文字化けしません（Streamlit側の描画）
      #    図のタイトルは英語にしてフォント依存を回避

      # 小さなUIもコンテナ内に（サイドバーは使わない）
      colA, colB= st.columns([1,1])
      with colA:
          m = st.number_input("分割数 m", min_value=2, max_value=4, value=2, step=1)
      with colB:
          seed = st.number_input("シード", min_value=0, max_value=10000, value=1, step=1)

      # データ生成＆分割
      G = make_demo_graph(seed=seed)
      pos = fixed_layout(G)
      parts_rand = partition_random(G, m=m, seed=seed)
      parts_deg  = partition_degree_round_robin(G, m=m)
      parts_bfs  = partition_bfs_blocks(G, m=m, seed=seed)

      # 図を1つのFigureにまとめて表示（コンテナ内で完結）
      fig, axes = plt.subplots(1, 3, figsize=(12, 4))
      draw_partitioned(axes[0], G, parts_rand, pos, "Random")
      draw_partitioned(axes[1], G, parts_deg,  pos, "Degree RR")
      draw_partitioned(axes[2], G, parts_bfs,  pos, "BFS Blocks")
      plt.tight_layout()
      st.pyplot(fig)
