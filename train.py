from nsga2 import toolbox
from data import seed_dataset
from lora_utils import build_lora, train_lora
from eval_utils import evaluate
from config import POP_SIZE, GENERATIONS

def main():
    dataset = seed_dataset()
    population = toolbox.population(n=POP_SIZE)

    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen} ===")

        for ind in population:
            model = build_lora(ind)
            model = train_lora(model, dataset)

            sg, eng = evaluate(model)
            ind.fitness.values = (sg, eng)

            print(f"SG={sg:.3f} | ENG_LOSS={eng:.3f} | {ind}")

        population = toolbox.select(population, len(population))

        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(c1, c2)
            toolbox.mutate(c1)
            toolbox.mutate(c2)

        population[:] = offspring

if __name__ == "__main__":
    from deap import tools
    main()
