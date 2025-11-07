import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(page_title="Edge List Visualizer", layout="wide")
st.title("Edge List Visualizer — source,target (CSV from file)")
st.caption("同じディレクトリ内のCSVファイルを相対パスで読み込み、Plotlyでネットワークを可視化します。")

# --- CSVファイルを相対パスで指定 ---
file_path = "assets/csv/edges.csv"  # ← 後でここを変更してください

# --- CSV読み込み ---
try:
    df = pd.read_csv(file_path)
    st.success(f"CSVを読み込みました: {file_path}")
except Exception as e:
    st.error(f"CSVの読み込みに失敗しました: {e}")
    st.stop()

# --- 検証 ---
if not {'source','target'}.issubset(df.columns):
    st.error("CSVに'source'と'target'列が必要です。")
    st.stop()

# --- グラフ構築 ---
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row['source'], row['target'])

# --- レイアウト計算 ---
pos = nx.spring_layout(G, seed=42)

# --- Plotlyで描画 ---
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1), mode='lines', hoverinfo='none')

node_x, node_y = [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[str(n) for n in G.nodes()],
    textposition='top center',
    marker=dict(size=12, line=dict(width=1)),
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    showlegend=False,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write(f"ノード数: {G.number_of_nodes()}  /  エッジ数: {G.number_of_edges()}")
