import streamlit as st
import pandas as pd
import numpy as np
import random
import networkx as nx
import time
import plotly.graph_objects as go

file_path = "assets/csv/G_set1.csv"
df = pd.read_csv(file_path, skiprows=3)

G = nx.from_pandas_edgelist(df, source="from", target="to", edge_attr="weight")
pos = nx.spring_layout(G, k=0.1, seed=7)


#各種設定
nodeNum = len(G.nodes()) #ノード数=遺伝子数
popSize = 100 #個体数
gengeration = 3000 #世代数
mutationRate = 0.1 #突然変異率

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

node = list(G.nodes())
node_index = {u: i for i, u in enumerate(node)}
all_edges = list(G.edges())
neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}


#初期個体群の生成(完全ランダムx100体)
def per50randomPopulation(nodeNum, popSize):
  Population = []
  for i in range(popSize):
    individual = [random.randint(0,1) for i in range(nodeNum)]
    corrected = greedyCorrection(individual, node, node_index, all_edges, neighbors)
    Population.append(corrected)
  return Population

def per33randomPopulation(nodeNum, popSize):
    Population = []
    for _ in range(popSize):
        base = [0] * nodeNum
        individual = []
        for bit in base:
            if random.random() > 0.33:
                individual.append(bit)
            else:
                individual.append(random.randint(0, 1))
        corrected = greedyCorrection(individual, node, node_index, all_edges, neighbors)
        Population.append(corrected)
    return Population




#適応度関数：個体内の1の数（例：頂点被覆サイズ）を最小化
def mainFitness(population):
  return sum(population)

#ルーレット選択
def rouletteSelect(evaPopList):
    # 逆数型適応度の計算
    converted = [1 / ((fit[0] + 1)**2) for fit in evaPopList]
    totalConvertedFitness = sum(converted)

    cumulative = []
    cumSum = 0
    for cf in converted:
        cumSum += cf
        cumulative.append(cumSum / totalConvertedFitness)

    r = random.random()
    for i, cum in enumerate(cumulative):
        if r <= cum:
            return i, evaPopList[i] #iは選択された個体のインデックス、evaPopList[i]は選択された個体の値[1,0,1,...]
          

#一点交叉
def pointCrossover(evaPopList):
    idx1, p1 = rouletteSelect(evaPopList)
    
    while True:
        idx2, p2 = rouletteSelect(evaPopList)
        if idx1 != idx2:
            break
    
    parent1 = p1[1] #選択された個体の値
    parent2 = p2[1] #選択された個体の値
    
    point = random.randint(1, nodeNum - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    
    #突然変異
    child1 = mutate_single_bit(child1, mutationRate)
    child2 = mutate_single_bit(child2, mutationRate)
    
    child1 = greedyCorrection(child1, node, node_index, all_edges, neighbors)
    child1 = greedyReduction(child1, G, node, node_index)
    child2 = greedyCorrection(child2, node, node_index, all_edges, neighbors)
    child2 = greedyReduction(child2, G, node, node_index)


    
    return child1, child2

def greedyCorrection(individual, node, node_index, all_edges, neighbors):
    individual = individual.copy()
    
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
            if individual[node_index[v]] == 0:
                scores[node_index[v]] += 1

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

def greedyReduction(individual, G, node, node_index):
    individual = individual.copy()

    #被覆に含まれる頂点のインデックスリストを作成
    current_cover_indices = [i for i, bit in enumerate(individual) if bit == 1]

    #次数が小さい順にソート
    current_cover_indices.sort(key=lambda i: G.degree(node[i]))

    for i in current_cover_indices:
        v = node[i]
        individual[i] = 0  #一旦削除

        #この削除によってカバーされなくなる辺があるか
        uncovered = [
            (u, w) for u, w in G.edges(v)
            if not (individual[node_index[u]] == 1 or individual[node_index[w]] == 1)
        ]

        if uncovered:
            individual[i] = 1  #必須だったので戻す

    return individual


def mutate_single_bit(individual, mutationRate):
    if random.random() < mutationRate:
        idx_to_mutate = random.randint(0, len(individual) - 1)
        individual[idx_to_mutate] = 1 - individual[idx_to_mutate]
    return individual
  
def is_vertex_cover(individual, all_edges, node, node_index):
    covered_edges = set()
    for i, bit in enumerate(individual):
        if bit == 1:
            u = node[i]
            for v in G.neighbors(u):
                covered_edges.add(tuple(sorted((u, v))))

    for u, v in all_edges:
        if tuple(sorted((u, v))) not in covered_edges:
            print(f"未カバーの辺: ({u}, {v})")
            return False
    return True


#-----------------------------------------------------------------------------#
start_time = time.time()

populations = per50randomPopulation(nodeNum, popSize) #初期個体群の生成
# corrected_zero = greedyCorrection([0] * nodeNum, node, node_index, all_edges, neighbors)
# populations.append(corrected_zero) #全て0の個体を追加

bestFitnessHistory = []

for gen in range(gengeration):
  
  
    fitnessList=[]
    for ind in populations:
      fitnessList.append(mainFitness(ind)) #適応度評価

    evaluatedPopulationList = [[fit, ind] for fit, ind in zip(fitnessList, populations)] 
    evaluatedPopulationList.sort(reverse=False, key=lambda fitness:fitness[0]) #評価値と各個体値リストをソート
    bestFitnessHistory.append(evaluatedPopulationList[0]) #最良適応度を記録

    elitePop = []
    elitePop = [ind for fit, ind in evaluatedPopulationList[:popSize//10]] #エリート個体を10%選択)

    nextGeneration = [] #次世代の個体群
    nextGeneration = elitePop.copy() #エリート個体を10%残す

    for i in range(popSize//4):
        child1, child2 = pointCrossover(evaluatedPopulationList)
        nextGeneration.append(child1)
        nextGeneration.append(child2) #交差による個体生成。50%の個体を生成

    nextGeneration.extend(per50randomPopulation(nodeNum, popSize-len(nextGeneration))) #ランダムに残りの個体を生成(大体70%くらい)
    
    populations = nextGeneration.copy()

    elapsed_time = time.time() - start_time
    avg_time_per_gen = elapsed_time / (gen + 1)
    remaining_time = avg_time_per_gen * (gengeration - gen - 1)

    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    time_text = f"残り推定時間: {minutes}分 {seconds}秒"

    my_bar.progress((gen + 1) / gengeration, text=time_text)




my_bar.empty()



# テキストでも最良適応度を確認
st.write(f"最良適応度（全世代）: {[entry[0] for entry in bestFitnessHistory]}")

# グラフ表示
fitness_values = [entry[0] for entry in bestFitnessHistory]

fig = go.Figure(data=go.Scatter(
    y=fitness_values,
    mode='lines+markers',
    name='Best Fitness'
))
fig.update_layout(
    title='世代ごとの最良適応度の推移',
    xaxis_title='世代',
    yaxis_title='適応度（1の数）',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)


#個体数100,世代数100
#[642, 640, 636, 629, 628, 628, 628, 628, 628, 628, 628, 628, 628, 628, 627, 627, 627, 627, 627, 627, 627, 627, 627, 627, 627, 627, 627, 623, 623, 623, 623, 623, 623, 623, 615, 615, 615, 615, 615, 615, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614, 614]
#全て0のビット列を個体群の中に入れた
#[599, 579, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 572, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 571, 569, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567, 567]



