MODEL_NAME = "meta-llama/Llama-3.2-1B"

POPULATION_SIZE = 10
GENERATIONS = 10
TRAIN_STEPS = 100

# LoRA hyperparameter search space (used for init, crossover, mutation)
LORA_R = [4, 8, 16]
LORA_ALPHA = [8, 16, 32]
LORA_DROPOUT = [0.0, 0.05, 0.1]
LORA_TARGETS = [["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj"]]

PROMPTS = [
    "Tell me about your childhood.",
    "What do you think about work-life balance?",
    "Describe Singapore in a few sentences."
]
