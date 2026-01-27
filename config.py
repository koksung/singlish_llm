MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Evolution
POP_SIZE = 6
GENERATIONS = 3

# Training
MAX_STEPS = 50
LR = 2e-4
BATCH_SIZE = 2

# LoRA search space
LORA_R = [4, 8, 16]
LORA_ALPHA = [16, 32, 64]
LORA_LAYERS = list(range(6, 18))  # middle-upper layers
