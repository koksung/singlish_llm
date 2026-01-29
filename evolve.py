import random

from deap import base, creator, tools

from config import (
    GENERATIONS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
    MODEL_NAME,
    POPULATION_SIZE,
    PROMPTS,
    TRAIN_STEPS,
    USE_GPU_IF_AVAILABLE,
)
from fitness import evaluate_outputs
from lora_train import train_and_generate

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", dict, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


def random_individual():
    return creator.Individual({
        "r": random.choice(LORA_R),
        "alpha": random.choice(LORA_ALPHA),
        "dropout": random.choice(LORA_DROPOUT),
        "targets": random.choice(LORA_TARGETS),
        "model_name": MODEL_NAME,
    })


def mate(ind1, ind2):
    """Uniform crossover over LoRA hyperparameters (dict individuals)."""
    for key in ("r", "alpha", "dropout", "targets"):
        if random.random() < 0.5:
            ind1[key], ind2[key] = ind2[key], ind1[key]
    return ind1, ind2


def mutate(ind, indpb=0.2):
    """Resample each LoRA hyperparameter with probability indpb."""
    for key, choices in (
        ("r", LORA_R),
        ("alpha", LORA_ALPHA),
        ("dropout", LORA_DROPOUT),
        ("targets", LORA_TARGETS),
    ):
        if random.random() < indpb:
            ind[key] = random.choice(choices)
    return (ind,)


def evaluate(ind):
    outputs = train_and_generate(ind, PROMPTS, TRAIN_STEPS, force_gpu=USE_GPU_IF_AVAILABLE)
    return evaluate_outputs(outputs)


toolbox.register("individual", random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mutate", mutate, indpb=0.2)
toolbox.register("select", tools.selNSGA2)


def main():
    random.seed(0)
    pop = toolbox.population(n=POPULATION_SIZE)

    for gen in range(GENERATIONS):
        print(f"=== Generation {gen} ===")
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop, len(pop))

        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)

        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)

        pop[:] = offspring

    print("Final Pareto front:")
    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    for ind in front:
        print(ind, ind.fitness.values)

if __name__ == "__main__":
    main()
