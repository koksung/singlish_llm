"""
One-time training script — run this BEFORE opening demo.ipynb.

    python train_lora.py

Trains a LoRA adapter on Singlish conversations and saves the adapter
weights to disk. The full model is NOT saved — only the small adapter.
"""
from config import ADAPTER_PATH, MODEL_NAME, TRAIN_STEPS
from lora_train import load_base_model, save_adapter, train_lora

print(f"Loading {MODEL_NAME} ...")
model, tokenizer, device = load_base_model()
print(f"Device : {device}")
total = sum(p.numel() for p in model.parameters())
print(f"Params : {total:,}\n")

print(f"Fine-tuning LoRA adapter for {TRAIN_STEPS} steps ...")
print("(Training data: 12 Singlish conversations, no system prompt)\n")
model = train_lora(model, tokenizer, device, steps=TRAIN_STEPS)

save_adapter(model, ADAPTER_PATH)
print(f"\nDone. Open demo.ipynb to run the 3-way comparison.")
