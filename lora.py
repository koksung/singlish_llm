import torch
import random

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from config import MODEL_NAME, MAX_STEPS, LR, BATCH_SIZE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def random_genome():
    return {
        "r": random.choice([4, 8, 16]),
        "alpha": random.choice([16, 32, 64]),
        "layers": random.sample(range(6, 18), k=4),
        "seed": random.randint(0, 10_000)
    }

def build_lora(genome):
    torch.manual_seed(genome["seed"])

    config = LoraConfig(
        r=genome["r"],
        lora_alpha=genome["alpha"],
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=genome["layers"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, config)
    return model

def train_lora(model, dataset):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        fp16=True,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized
    )

    trainer.train()
    return model
