import random

population = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
fitness = [80, 70, 60, 50, 40, 30, 20, 10]

def roulette_selection(population, fitness):
    total_fitness = sum(fitness)
    probs = []
    r_sum = 0
    for f in fitness:
        r_sum += f
        probs.append(r_sum / total_fitness)
    r = random.random()
    for i, prob in enumerate(probs):
        if r <= prob:
            return population[i]

for _ in range(10):
    print(roulette_selection(population, fitness))
