MODEL_NAME = "meta-llama/Llama-3.2-1B"

POPULATION_SIZE = 10
GENERATIONS = 10
TRAIN_STEPS = 100

# If True and CUDA is available, use GPU; otherwise CPU.
USE_GPU_IF_AVAILABLE = True

# If True, never hit Hugging Face Hub (offline / cached-only).
LOCAL_FILES_ONLY = False

# Keep this demo small/runnable.
TRAIN_BATCH_SIZE = 1
MAX_SEQ_LEN = 192

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

# Tiny synthetic "Singlish-ish" training texts for a real training signal.
# This is intentionally small and not meant for quality—just to demonstrate that
# LoRA params get gradients from an actual LM loss.
TRAIN_TEXTS = [
    "User: How are you?\nAssistant: I'm okay lah, just a bit tired but can manage.\n",
    "User: Can you help me?\nAssistant: Can lah. Tell me what you need.\n",
    "User: What should we eat?\nAssistant: Anything also can. Chicken rice steady.\n",
    "User: Are you coming later?\nAssistant: Maybe can, but I see how first lor.\n",
    "User: Why like that?\nAssistant: Aiya, sometimes things like that one leh.\n",
    "User: Is it expensive?\nAssistant: Not too bad meh, still okay for the quality.\n",
    "User: What's your opinion?\nAssistant: Personally I feel okay lah, but depends on you.\n",
    "User: Any tips for Singapore?\nAssistant: Bring umbrella lah, weather very unpredictable one.\n",
]
