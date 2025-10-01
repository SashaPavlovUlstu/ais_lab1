# Код выполняется и будет видим пользователю.
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import time
import os

# Установим семя для воспроизводимости
random.seed(42)
np.random.seed(42)

# --- ПАРАМЕТРЫ ЗАДАЧИ (реалистичные диапазоны) ---
N = 5  # Количество пунктов производства (фабрик/складов)
K = 8  # Количество городов
Y_max = 200  # Максимальное производство на пункт (шт/день) — реалистично для небольшого склада
X_need = 150  # Базовая потребность города (шт/день)

price_per_unit_distance = 0.5  # Стоимость доставки за ед. продукта на ед. расстояния (рубли/шт*km)
penalty_per_excess_unit = 2.0  # Штраф за единицу превышения (руб/шт)
penalty_per_shortage_unit = 5.0  # Штраф за единицу недостачи (руб/шт)

# --- ПАРАМЕТРЫ ГА ---
POPULATION_SIZE = 80
GENERATIONS = 120
MUTATION_RATE = 0.12

# --- Генерация "максимально правдоподобных" данных ---
# Координаты моделей в пределах 0..1000 км (условная сетка); расстояния — Евклидовы.
production_points = []
for i in range(N):
    supply = random.randint(int(Y_max*0.7), Y_max)  # склады загружены 70-100%
    production_points.append({'id': i,
                              'x': random.randint(0, 1000),
                              'y': random.randint(0, 1000),
                              'supply': supply})

cities = []
for j in range(K):
    demand = random.randint(int(X_need*0.7), int(X_need*1.3))  # города потребляют ±30%
    cities.append({'id': j,
                   'x': random.randint(0, 1000),
                   'y': random.randint(0, 1000),
                   'demand': demand})

# Матрица расстояний (в км)
distance_matrix = np.zeros((N, K))
for i in range(N):
    for j in range(K):
        dx = production_points[i]['x'] - cities[j]['x']
        dy = production_points[i]['y'] - cities[j]['y']
        distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)

# --- Функция стоимости/пригодности ---
def total_cost_and_penalty(individual):
    """Возвращает (total_cost + penalties, total_cost, excess_penalty, shortage_penalty)."""
    # Проверка ограничений по запасу на пункте
    for i in range(N):
        if individual[i, :].sum() > production_points[i]['supply']:
            # Запрещённая конфигурация — очень большой штраф
            return 1e12, 1e12, 1e12, 1e12

    total_trans_cost = 0.0
    excess_pen = 0.0
    short_pen = 0.0

    for i in range(N):
        for j in range(K):
            total_trans_cost += individual[i, j] * distance_matrix[i, j] * price_per_unit_distance

    for j in range(K):
        received = individual[:, j].sum()
        if received > cities[j]['demand']:
            excess_pen += (received - cities[j]['demand']) * penalty_per_excess_unit
        elif received < cities[j]['demand']:
            short_pen += (cities[j]['demand'] - received) * penalty_per_shortage_unit

    total = total_trans_cost + excess_pen + short_pen
    return total, total_trans_cost, excess_pen, short_pen

def fitness(individual):
    total, _, _, _ = total_cost_and_penalty(individual)
    # Преобразуем в fitness: чем меньше cost — тем лучше
    return 1.0 / (1.0 + total)

# --- ОПЕРАТОРЫ ---

def create_initial_population(size):
    pop = []
    for _ in range(size):
        indiv = np.zeros((N, K), dtype=int)
        for i in range(N):
            supply = production_points[i]['supply']
            # случайное разбиение supply по K городам (включая нули)
            if supply == 0:
                continue
            cuts = sorted(random.sample(range(1, supply + K), K - 1))
            parts = []
            prev = 0
            for c in cuts:
                parts.append(c - prev - (len(parts)))  # корректируем смещение
                prev = c
            parts.append(supply - sum(parts))
            # Если что-то не так, используем равномерное распределение
            if any(p < 0 for p in parts):
                parts = [supply // K] * K
                rem = supply - sum(parts)
                for r in range(rem):
                    parts[r % K] += 1
            indiv[i, :] = np.array(parts[:K], dtype=int)
        pop.append(indiv)
    return pop

def tournament_selection(population, k=3):
    selected = []
    for _ in range(len(population)//2):
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=fitness)
        selected.append(winner)
    return selected


def crossover_single(parent1, parent2):
    cp = random.randint(1, N-1)
    child1 = np.vstack((parent1[:cp, :], parent2[cp:, :]))
    child2 = np.vstack((parent2[:cp, :], parent1[cp:, :]))
    return child1, child2

def crossover_two_point(parent1, parent2):
    p1, p2 = sorted(random.sample(range(1, N-1), 2))
    child1 = np.vstack((parent1[:p1, :], parent2[p1:p2, :], parent1[p2:, :]))
    child2 = np.vstack((parent2[:p1, :], parent1[p1:p2, :], parent2[p2:, :]))
    return child1, child2

def crossover_uniform(parent1, parent2):
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    for i in range(N):
        if random.random() < 0.5:
            child1[i, :] = parent1[i, :]
            child2[i, :] = parent2[i, :]
        else:
            child1[i, :] = parent2[i, :]
            child2[i, :] = parent1[i, :]
    return child1, child2


def mutation_single_point(indiv):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, N-1)
        j = random.randint(0, K-1)
        delta = random.randint(-20, 20)
        new_val = max(0, indiv[i, j] + delta)
        row_sum = indiv[i, :].sum() - indiv[i, j] + new_val
        supply = production_points[i]['supply']
        if row_sum > supply:
            scale = (supply - new_val) / max(1, indiv[i, :].sum() - indiv[i, j])
            for c in range(K):
                if c == j: continue
                indiv[i, c] = int(max(0, indiv[i, c] * scale))
        indiv[i, j] = new_val
    return indiv

def mutation_reset(indiv):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, N-1)
        j = random.randint(0, K-1)
        indiv[i, j] = random.randint(0, production_points[i]['supply'])
        row_sum = indiv[i, :].sum()
        supply = production_points[i]['supply']
        if row_sum > supply:
            indiv[i, :] = (indiv[i, :] * (supply / row_sum)).astype(int)
    return indiv

def mutation_redistribute_row(indiv):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, N-1)
        supply = production_points[i]['supply']
        parts = np.random.multinomial(supply, [1.0/K]*K)
        indiv[i, :] = parts
    return indiv

CROSSOVERS = {
    'single_point': crossover_single,
    'two_point': crossover_two_point,
    'uniform': crossover_uniform
}

MUTATIONS = {
    'single_point': mutation_single_point,
    'reset': mutation_reset,
    'redistribute': mutation_redistribute_row
}

def genetic_algorithm(crossover_name='single_point', mutation_name='single_point',
                      population_size=POPULATION_SIZE, generations=GENERATIONS):
    pop = create_initial_population(population_size)
    best_history = []
    best_indiv = None
    best_fit = -1.0

    crossover_fn = CROSSOVERS[crossover_name]
    mutation_fn = MUTATIONS[mutation_name]

    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in pop]
        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > best_fit:
            best_fit = fitnesses[best_idx]
            best_indiv = pop[best_idx].copy()
        best_history.append(best_fit)

        parents = tournament_selection(pop, k=3)

        next_gen = []
        while len(next_gen) < population_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child1, child2 = crossover_fn(p1, p2)
            # мутация
            child1 = mutation_fn(child1.copy())
            child2 = mutation_fn(child2.copy())
            next_gen.append(child1)
            if len(next_gen) < population_size:
                next_gen.append(child2)
        pop = next_gen

    final_fitnesses = [fitness(ind) for ind in pop]
    best_idx = int(np.argmax(final_fitnesses))
    if final_fitnesses[best_idx] > best_fit:
        best_fit = final_fitnesses[best_idx]
        best_indiv = pop[best_idx].copy()

    total, trans_cost, excess_pen, short_pen = total_cost_and_penalty(best_indiv)
    return {
        'best_individual': best_indiv,
        'fitness': best_fit,
        'total_cost': total,
        'transport_cost': trans_cost,
        'excess_penalty': excess_pen,
        'shortage_penalty': short_pen,
        'history': best_history
    }

def brute_force_small(Nb=3, Kb=3, Yb=6):
    brute_prod = [{'supply': Yb} for _ in range(Nb)]
    brute_cities = [{'demand': random.randint(max(1, Yb-2), Yb+2)} for _ in range(Kb)]
    brute_dist = np.random.randint(5, 30, size=(Nb, Kb))
    brute_price = 1.0
    brute_pen_ex = 5.0
    brute_pen_sh = 10.0

    def penalty_brute(ind):
        for i in range(Nb):
            if ind[i, :].sum() > brute_prod[i]['supply']:
                return 1e9
        tcost = 0.0
        ex = 0.0
        sh = 0.0
        for i in range(Nb):
            for j in range(Kb):
                tcost += ind[i, j] * brute_dist[i, j] * brute_price
        for j in range(Kb):
            rec = ind[:, j].sum()
            if rec > brute_cities[j]['demand']:
                ex += (rec - brute_cities[j]['demand']) * brute_pen_ex
            elif rec < brute_cities[j]['demand']:
                sh += (brute_cities[j]['demand'] - rec) * brute_pen_sh
        return tcost + ex + sh

    best = None
    best_val = 1e18
    start = time.time()
    all_vals = range(Yb+1)
    total_combos = (Yb+1) ** (Nb * Kb)
    for combo in itertools.product(all_vals, repeat=Nb*Kb):
        mat = np.array(combo).reshape(Nb, Kb)
        val = penalty_brute(mat)
        if val < best_val:
            best_val = val
            best = mat.copy()
    elapsed = time.time() - start
    return best, best_val, elapsed, brute_prod, brute_cities, brute_dist

results = {}
start_all = time.time()

for c_name in CROSSOVERS.keys():
    for m_name in MUTATIONS.keys():
        key = f"{c_name} + {m_name}"
        t0 = time.time()
        res = genetic_algorithm(crossover_name=c_name, mutation_name=m_name,
                                population_size=POPULATION_SIZE, generations=GENERATIONS)
        t1 = time.time()
        res['time'] = t1 - t0
        results[key] = res
        print(f"Completed: {key} in {res['time']:.2f}s, best total cost={res['total_cost']:.2f}")

end_all = time.time()
print(f"All experiments done in {end_all - start_all:.2f}s")

os.makedirs('outputs', exist_ok=True)
plt.figure(figsize=(10,6))
for key, val in results.items():
    plt.plot(val['history'], label=key)
plt.title("Сравнение развития лучшей пригодности (fitness) по поколениям для всех комбинаций")
plt.xlabel("Поколение")
plt.ylabel("Fitness (1 / (1+cost))")
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/all_combinations_history.png')
plt.show()

with open('outputs/report_summary.txt', 'w', encoding='utf-8') as f:
    f.write("Сводка результатов экспериментов\n\n")
    for key, val in results.items():
        f.write(f"{key}:\n")
        f.write(f"  total_cost = {val['total_cost']:.2f}\n")
        f.write(f"  transport_cost = {val['transport_cost']:.2f}\n")
        f.write(f"  excess_penalty = {val['excess_penalty']:.2f}\n")
        f.write(f"  shortage_penalty = {val['shortage_penalty']:.2f}\n")
        f.write(f"  fitness = {val['fitness']:.6f}\n")
        f.write(f"  time = {val['time']:.2f}s\n")
        f.write("  best_individual (matrix):\n")
        np.savetxt(f, val['best_individual'], fmt='%d')
        f.write("\n\n")

best_brute, val_brute, time_brute, bp, bc, bd = brute_force_small(Nb=3, Kb=3, Yb=5)
with open('outputs/brute_force_result.txt', 'w', encoding='utf-8') as f:
    f.write("Результат полного перебора (малый пример)\n\n")
    f.write(f"best_value = {val_brute}\n")
    f.write(f"time = {time_brute:.4f}s\n")
    f.write("best_matrix:\n")
    np.savetxt(f, best_brute, fmt='%d')

import json
with open('outputs/input_data.json', 'w', encoding='utf-8') as f:
    json.dump({'production_points': production_points, 'cities': cities, 'distance_matrix': distance_matrix.tolist()}, f, ensure_ascii=False, indent=2)

print("Outputs saved to ./outputs/")
for key, val in results.items():
    print("===")
    print(key)
    print(f"total_cost={val['total_cost']:.2f}, transport={val['transport_cost']:.2f}, excess={val['excess_penalty']:.2f}, shortage={val['shortage_penalty']:.2f}, time={val['time']:.2f}s")

best_key = min(results.keys(), key=lambda k: results[k]['total_cost'])
print("\nBest overall:", best_key)
print("Best matrix:")
print(results[best_key]['best_individual'])
print("Total cost:", results[best_key]['total_cost'])
