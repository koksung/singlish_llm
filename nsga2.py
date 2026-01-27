from deap import base, creator, tools
from lora_utils import random_genome

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", dict, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual(random_genome()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxUniform, indpb=0.5)

def mutate(ind):
    ind.update(random_genome())
    return (ind,)

toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)
