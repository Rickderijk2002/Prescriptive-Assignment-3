### TASK O: 30-RUN GA EXPERIMENT (CONSULTANCY RUL PREDICTIONS) ###
# Standalone script — mirrors the notebook but runs only Task O.
# Saves Results/task_O_convergence.png and Results/task_O_histories.npy.

import os
# Work from the script's own directory so all relative paths match the notebook
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Libraries
import numpy as np
import pandas as pd
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from deap import base, creator, tools

### HELPER FUNCTIONS ###

# Team type mapping
TEAM_TYPES = {1: "A", 2: "B", 3: "A", 4: "B"}

def get_engine_due_date(engine_id, RUL):
    '''Safety due date is ts = t1 + Rj - 1, and t1 = 1, so ts = Rj'''
    return int(RUL[engine_id])

def get_maintenance_duration(engine_id, team_type):
    '''Returns the days to perform maintenance for the input team type and engine ID'''
    # First get the days for team A
    if 1 <= engine_id <= 20:
        mu_A = 5
    elif 21 <= engine_id <= 55:
        mu_A = 3
    elif 56 <= engine_id <= 80:
        mu_A = 4
    else:
        mu_A = 5

    # If team type A, return mu_A. Otherwise, use mu_A to calculate team B's days.
    if team_type == "A":
        return mu_A
    elif team_type == "B":
        if 1 <= engine_id <= 25:
            return mu_A - 1
        elif 26 <= engine_id <= 70:
            return mu_A + 3
        else:
            return mu_A + 2
    else:
        raise ValueError("get_maintenance_duration() received invalid team type!")

def get_penalty_cost(engine_id, completion_date, due_date):
    '''Computes the penalty cost given the completion and due dates of an engine'''
    if completion_date <= due_date:
        return 0

    # Get the cj value for the engine's id range
    if 1 <= engine_id <= 25:
        cj = 4
    elif 26 <= engine_id <= 45:
        cj = 2
    elif 46 <= engine_id <= 75:
        cj = 5
    else:
        cj = 6

    # Penalty charged for every late day; capped daily
    penalty = 0
    for day in range(due_date + 1, completion_date + 1):
        delay = day - due_date
        penalty += min(cj * (delay ** 2), 250)

    return penalty

### MAIN GA FUNCTIONS ###

def build_schedule(individual):
    team_jobs = defaultdict(list)
    for engine, team, start in individual:
        team_jobs[team].append((engine, start))
    return team_jobs

def is_feasible(team_jobs):
    '''Returns whether or not a provided (individual's) schedule is feasible given the constraints'''
    for team, jobs in team_jobs.items():
        jobs = sorted(jobs, key=lambda x: x[1])
        current_time = 0
        for engine, start in jobs:
            duration = get_maintenance_duration(engine, TEAM_TYPES[team])
            if start < current_time:
                return False
            if start + duration - 1 > 30:
                return False
            current_time = start + duration
    return True

def evaluate(individual, RUL):
    '''
    Calculates the total penalty for a given individual.
    Trailing commas in outputs are there for DEAP format.
    '''
    team_jobs = build_schedule(individual)
    if not is_feasible(team_jobs):
        return (1e9,)

    maintained_engines = set()
    total_penalty = 0

    # Calculate penalties for maintained engines
    for team, jobs in team_jobs.items():
        for engine, start in jobs:
            duration = get_maintenance_duration(engine, TEAM_TYPES[team])
            completion = start + duration - 1
            due_date = get_engine_due_date(engine, RUL)
            total_penalty += get_penalty_cost(engine, completion, due_date)
            maintained_engines.add(engine)

    # Calculate penalties for non-maintained engines
    for engine in range(1, 101):
        if engine not in maintained_engines:
            due = get_engine_due_date(engine, RUL)
            total_penalty += get_penalty_cost(engine, 30, due)

    return (total_penalty,)

def custom_mutation(individual, prob=0.3):
    '''
    Combines multiple small mutations into one to ensure all parts of the chromosome can be mutated.
    This includes start time, team type, engine id, and chromosome length (amount of engines maintained).
    '''
    # Start time mutation
    for i in range(len(individual)):
        if random.random() < prob:
            engine, team, start = individual[i]
            shift = random.randint(-3, 3)
            individual[i] = (engine, team, max(1, min(30, start + shift)))

    # Team type mutation
    for i in range(len(individual)):
        if random.random() < prob:
            engine, team, start = individual[i]
            individual[i] = (engine, random.randint(1, 4), start)

    # Engine id mutation
    for i in range(len(individual)):
        if random.random() < prob:
            engine, team, start = individual[i]
            individual[i] = (random.randint(1, 100), team, start)

    # Chromosome length mutation
    if random.random() < prob and len(individual) > 1:
        individual.pop(random.randrange(len(individual)))
    elif random.random() > 1 - prob:
        individual.append((random.randint(1, 100), random.randint(1, 4), random.randint(1, 30)))

    return (individual,)

def custom_crossover(parent_1, parent_2):
    '''
    Custom crossover function with the goal of keeping schedules reasonably intact.
    To that end, children inherit subsets (~half) of parent schedules, rather than single point crossover or similar.
    '''
    child_1, child_2 = [], []

    split_1 = set(random.sample(range(len(parent_1)), k=len(parent_1) // 2))
    split_2 = set(random.sample(range(len(parent_2)), k=len(parent_2) // 2))

    # Child 1 creation
    child_1.extend(parent_1[i] for i in split_1)
    maintained_1 = {g[0] for g in child_1}
    for gene in parent_2:
        if gene[0] not in maintained_1:
            child_1.append(gene)

    # Child 2 creation
    child_2.extend(parent_2[i] for i in split_2)
    maintained_2 = {g[0] for g in child_2}
    for gene in parent_1:
        if gene[0] not in maintained_2:
            child_2.append(gene)

    parent_1[:] = child_1
    parent_2[:] = child_2
    return parent_1, parent_2

### DEAP SETUP ###

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    n_jobs = random.randint(10, 40)  # Ensure varied but reasonable schedule lengths
    indiv = []
    for _ in range(n_jobs):
        indiv.append((random.randint(1, 100), random.randint(1, 4), random.randint(1, 30)))
    return creator.Individual(indiv)

def repair(individual, T=30):
    '''Repairs an individual in-place: removes duplicates, clips to horizon, resolves team overlaps'''
    # Remove duplicate engine assignments (keep first occurrence)
    seen = set()
    unique_genes = []
    for gene in individual:
        if gene[0] not in seen:
            seen.add(gene[0])
            unique_genes.append(gene)
    individual[:] = unique_genes

    # Fix each gene so it fits within the horizon
    valid_genes = []
    for engine_id, team_id, start_day in individual:
        team_id = max(1, min(4, int(team_id)))
        start_day = max(1, int(start_day))
        duration = get_maintenance_duration(engine_id, TEAM_TYPES[team_id])
        if start_day + duration - 1 > T:
            start_day = T - duration + 1
        if start_day < 1:
            continue
        valid_genes.append((engine_id, team_id, start_day))
    individual[:] = valid_genes

    # Resolve time overlaps within each team
    for team_id in (1, 2, 3, 4):
        team_idx = [i for i, g in enumerate(individual) if g is not None and g[1] == team_id]
        team_idx.sort(key=lambda i: individual[i][2])
        next_free = 1
        for i in team_idx:
            engine_id, t, start_day = individual[i]
            duration = get_maintenance_duration(engine_id, TEAM_TYPES[t])
            start_day = max(start_day, next_free)
            end_day = start_day + duration - 1
            if end_day > T:
                individual[i] = None
            else:
                individual[i] = (engine_id, t, start_day)
                next_free = end_day + 1

    individual[:] = [g for g in individual if g is not None]
    return individual

# GA constants
T           = 30
N_ENGINES   = 100
POP_SIZE    = 100
CX_PROB     = 0.70
MUT_PROB    = 0.20
MAX_SECONDS = 300  # 5-minute cap

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation, prob=0.3)

def run_ga(rul_input, seed=None):
    '''
    Runs the GA for up to MAX_SECONDS seconds.
    Returns (best_ind, best_cost, history).
    '''
    if seed is not None:
        random.seed(seed)

    # Re-register evaluate with the correct RUL dict for this run
    toolbox.register("evaluate", evaluate, RUL=rul_input)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        repair(ind)
        ind.fitness.values = toolbox.evaluate(ind)

    history = []
    start_time = time.time()
    gen = 0

    while time.time() - start_time < MAX_SECONDS:
        gen += 1

        # Selection
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Crossover — repair both children immediately after
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                repair(child1)
                repair(child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation — repair the mutant immediately after
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                repair(mutant)
                del mutant.fitness.values

        # Re-evaluate only individuals that changed
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring

        # Track best this generation
        best_now = min(pop, key=lambda x: x.fitness.values[0])
        history.append(best_now.fitness.values[0])

        if gen % 100 == 0:
            print(f"  Gen {gen:5d} | Best: {history[-1]:>10,.0f} | {time.time()-start_time:.1f}s elapsed")

    best_ind = min(pop, key=lambda x: x.fitness.values[0])
    print(f"\nDone — {gen} generations | Best penalty: {best_ind.fitness.values[0]:,.0f}")
    return best_ind, best_ind.fitness.values[0], history

### TASK O: 30-RUN EXPERIMENT ###

# Load consultancy RUL predictions (semicolon-delimited: RUL, id)
df_cons = pd.read_csv("../Data/RUL_consultancy_predictions_A3.csv", sep=";")
RUL_consultancy = dict(zip(df_cons.id, df_cons.RUL))

N_RUNS = 30

# 30-run GA loop
all_histories_consultancy = []
all_bests_consultancy     = []

for i in range(N_RUNS):
    best_ind_cons, best_cost_cons, hist = run_ga(rul_input=RUL_consultancy, seed=i)
    all_histories_consultancy.append(hist)
    all_bests_consultancy.append((best_ind_cons, best_cost_cons))
    print(f"Run {i:2d} | Best cost: {best_cost_cons:,.0f}")
    # Save full history list after every run so progress is not lost on crash
    _max_len = max(len(h) for h in all_histories_consultancy)
    _padded  = [h + [h[-1]] * (_max_len - len(h)) for h in all_histories_consultancy]
    np.save("Results/histories_consultancy_rul.npy", np.array(_padded))

# Overall best individual across all 30 runs
overall_best_ind_cons, overall_best_cost_cons = min(all_bests_consultancy, key=lambda x: x[1])

# Save best schedule to txt matching the print_schedule format from the notebook
with open("Results/best_schedule_consultancy_rul.txt", "w", encoding="utf-8") as f:
    header = (f"{'Engine':>8} | {'Team':>5} | {'Type':>4} | "
              f"{'Start':>5} | {'End':>5} | {'Due':>5} | {'Penalty':>9}")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    maintained = {}
    for engine_id, team_id, start_day in overall_best_ind_cons:
        duration = get_maintenance_duration(engine_id, TEAM_TYPES[team_id])
        end_day  = start_day + duration - 1
        due_date = get_engine_due_date(engine_id, RUL_consultancy)
        penalty  = get_penalty_cost(engine_id, end_day, due_date)
        maintained[engine_id] = (team_id, start_day, end_day, due_date, penalty)
        f.write(f"{engine_id:>8} | {team_id:>5} | {TEAM_TYPES[team_id]:>4} | "
                f"{start_day:>5} | {end_day:>5} | {due_date:>5} | {penalty:>9.0f}\n")

    total_penalty = sum(v[4] for v in maintained.values())

    f.write("\n")
    f.write("Unscheduled engines with RUL < T (incurring full penalty):\n")
    for engine_id in range(1, 101):
        due = get_engine_due_date(engine_id, RUL_consultancy)
        if due < T and engine_id not in maintained:
            penalty = get_penalty_cost(engine_id, T, due)
            total_penalty += penalty
            f.write(f"  Engine {engine_id}: due={due}, penalty={penalty:.0f}\n")

    f.write("-" * len(header) + "\n")
    f.write(f"Total penalty cost: {total_penalty:,.0f}\n")

print(f"Saved Results/task_O_best_schedule.txt (best cost: {overall_best_cost_cons:,.0f})")

# Pad each history to the same length using its last value
max_len_o = max(len(h) for h in all_histories_consultancy)
padded_o  = [h + [h[-1]] * (max_len_o - len(h)) for h in all_histories_consultancy]

# Average best fitness per generation across all runs
avg_o = np.mean(padded_o, axis=0)

# Convergence plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_len_o + 1), avg_o)
plt.xlabel("Generation")
plt.ylabel("Average Best Penalty Cost")
plt.title("Task O: 30-Run Average Best Fitness (Consultancy RUL Predictions)")
plt.tight_layout()
plt.savefig("Results/convergence_consultancy_rul.png")
plt.show()
print("Saved Results/convergence_consultancy_rul.png")
