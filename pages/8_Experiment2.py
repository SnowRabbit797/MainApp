import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.font_manager as fm

POP_SIZE = 100
N_GENERATIONS = 1000
ELITE_SIZE = 10
NUM_RANDOM = 15
# ALPHA = 10
# BETA = 1
MUTATION_RATE = 0.01

@st.cache_data
def load_graph(filepath):
    df = pd.read_csv(filepath, skiprows=2)
    edge_list = [(int(row["from"]) - 1, int(row["to"]) - 1) for _, row in df.iterrows()]
    return edge_list

edge_list = load_graph("assets/csv/G_set1.csv")
num_nodes = 800

def init_population(pop_size, num_nodes):
    return [[random.randint(0, 1) for _ in range(num_nodes)] for _ in range(pop_size)]

def evaluate_fitness(individual, edge_list):
    covered_edges = sum(1 for u, v in edge_list if individual[u] == 1 or individual[v] == 1)
    used_nodes = sum(individual)
    total_edges = len(edge_list)

    if covered_edges == total_edges:
        return 10000 - used_nodes  # 少ないノード数ほど高評価
    else:
        return covered_edges - total_edges * 2  # カバー不足には重いペナルティ


def roulette_selection(population, fitness_list, num_selected):
    total_fitness = sum(fitness_list)
    cumulative_probs = []
    cumulative_sum = 0
    for f in fitness_list:
        cumulative_sum += f
        cumulative_probs.append(cumulative_sum / total_fitness)
    selected = []
    for _ in range(num_selected):
        r = random.random()
        for i, prob in enumerate(cumulative_probs):
            if r <= prob:
                selected.append(population[i][:])
                break
    return selected

def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2


def mutate(individual, mutation_rate=MUTATION_RATE):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual
  
def run_ga():
    population = init_population(POP_SIZE, num_nodes)
    best_individual_history = []
    min_valid_node_counts = []
    node_history = []

    for _ in range(N_GENERATIONS):
        fitness_list = [evaluate_fitness(ind, edge_list) for ind in population]

        best_index = fitness_list.index(max(fitness_list))
        best_node_count = sum(population[best_index])
        node_history.append(best_node_count)
        best_individual_history.append(population[best_index][:]) 

        elite_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:ELITE_SIZE]
        elites = [population[i][:] for i in elite_indices]

        selected = roulette_selection(population, fitness_list, POP_SIZE)
        next_generation = elites[:]
        num_offspring = POP_SIZE - ELITE_SIZE - NUM_RANDOM
        while len(next_generation) < ELITE_SIZE + num_offspring:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            next_generation.append(mutate(c1))
            if len(next_generation) < ELITE_SIZE + num_offspring:
                next_generation.append(mutate(c2))
        while len(next_generation) < POP_SIZE:
            new_random = [random.randint(0, 1) for _ in range(num_nodes)]
            next_generation.append(new_random)

        population = next_generation[:POP_SIZE]

    return population, node_history, best_individual_history

st.title("GAによる最小頂点被覆問題 (MVC)")
if st.button("GAを実行"):
    with st.spinner("ただいま進化中..."):
        population, node_history, best_individual_history = run_ga()

    fitness_list = [evaluate_fitness(ind, edge_list) for ind in population]
    best_index = fitness_list.index(max(fitness_list))
    best_individual = population[best_index]
    used_nodes = sum(best_individual)
    covered_edges = sum(1 for u, v in edge_list if best_individual[u] == 1 or best_individual[v] == 1)

    st.success("完了~~~")
    st.markdown(f"### 使用ノード数: **{used_nodes}**")
    st.markdown(f"### カバーされたエッジ数: **{covered_edges} / {len(edge_list)}**")

    fig, ax = plt.subplots()
    ax.plot(range(N_GENERATIONS), node_history)
    st.pyplot(fig)
