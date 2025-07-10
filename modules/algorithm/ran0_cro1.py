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

st.write(G)

#各種設定
nodeNum = len(G.nodes()) #ノード数=遺伝子数
popSize = 1000 #個体数
gengeration = 1000 #世代数
mutationRate = 0 #突然変異率

#初期個体群の生成(ランダムx100体)
def randomPopulation(nodeNum, popSize):
  Population = []
  for i in range(popSize):
    individual = [random.randint(0,1) for i in range(nodeNum)]
    Population.append(individual)
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
    if random.random() < mutationRate:
        child1 = [1 - bit for bit in child1]

    if random.random() < mutationRate:
        child2 = [1 - bit for bit in child2]
        
    
    return child1, child2


#-----------------------------------------------------------------------------#
populations = randomPopulation(nodeNum, popSize) #初期個体群の生成
bestFitnessHistory = []

for gen in range(gengeration):
    fitnessList=[]
    for list in populations:
      fitnessList.append(mainFitness(list)) #適応度評価

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
        nextGeneration.append(child2)#交差による個体生成。20%の個体を生成

    nextGeneration.extend(randomPopulation(nodeNum, popSize-len(nextGeneration))) #ランダムに残りの個体を生成(大体70%くらい)
    
    populations = nextGeneration.copy()


st.write(bestFitnessHistory) #最良適応度


fitness_values = [entry[0] for entry in bestFitnessHistory]

fig = go.Figure(data=go.Scatter(
    y=fitness_values,
    mode='lines',
    name='Best Fitness'
))
fig.update_layout(
    xaxis_title='Generation',
    yaxis_title='Fitness'
)

st.plotly_chart(fig)
