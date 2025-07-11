import streamlit as st
import pandas as pd
import numpy as np
import random
import networkx as nx
import plotly.graph_objects as go

file_path = "assets/csv/G_set1.csv"
df = pd.read_csv(file_path, skiprows=3)

G = nx.from_pandas_edgelist(df, source="from", target="to", edge_attr="weight")
pos = nx.spring_layout(G, k=0.1, seed=7)


#各種設定
nodeNum = len(G.nodes()) #ノード数=遺伝子数
popSize = 100 #個体数
gengeration = 100 #世代数
mutationRate = 0.1 #突然変異率

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

node = list(G.nodes())
node_index = {u: i for i, u in enumerate(node)}
all_edges = list(G.edges())
neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}

def greedyCorrection(individual, node, node_index, all_edges, neighbors):
    individual = individual.copy()
    node = list(G.nodes())
    
    covered_edges = set()
    #現在カバーされている辺を covered_edges に入れる
    for i, bit in enumerate(individual):
        if bit == 1:
            u = node[i]
            for v in neighbors[u]:
                covered_edges.add(tuple(sorted((u, v))))

    #カバーされていない辺を見つける
    uncovered_edges = []
    for e in all_edges:
        edge = tuple(sorted(e))
        if edge not in covered_edges:
            uncovered_edges.append(e)
            
    while uncovered_edges:
        #スコアテーブル初期化(各ノードが何本の未被覆辺に接続しているか)
        scores = [0] * len(individual)

        #各未被覆辺について、両端のノードにスコアを加算
        for u, v in uncovered_edges:
            if individual[node_index[u]] == 0:
                scores[node_index[u]] += 1
            else: #デバッグ用
                st.write(f"error")
            if individual[node_index[v]] == 0:
                scores[node_index[v]] += 1
            else: #デバッグ用
                st.write(f"error")

        #最もスコアの高いノード(最も多くの未被覆辺に接している)を追加
        max_idx = scores.index(max(scores))
        individual[max_idx] = 1  #individual にノードを追加（= 1 にする）

        #新たに追加したノードがカバーする辺を covered_edges に追加
        u = node[max_idx]
        for v in neighbors[u]:
            covered_edges.add(tuple(sorted((u, v))))

        #uncovered_edges を再更新
        uncovered_edges = []
        for e in all_edges:
            if tuple(sorted(e)) not in covered_edges:
                uncovered_edges.append(e)
    return individual #修正された個体を返す

zeroindividual = [0] * nodeNum
corrected = greedyCorrection(zeroindividual, node, node_index, all_edges, neighbors)
st.write("修正後の個体:", corrected)
st.write("修正後の頂点数:", sum(corrected))
