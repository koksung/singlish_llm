import numpy as np
import torch

from lora_utils import tokenizer

EVAL_PROMPTS = [
    "Explain hawker culture.",
    "Why is MRT important in Singapore?",
    "Explain Singlish to a foreigner."
]

def singaporean_judge(text):
    keywords = ["lah", "leh", "lor", "MRT", "hawker", "CPF", "NS"]
    return sum(k.lower() in text.lower() for k in keywords) / len(keywords)

def english_loss(model):
    prompts = [
        "Explain photosynthesis.",
        "What is gravity?",
        "Explain the internet."
    ]

    losses = []
    model.eval()

    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())

    return float(np.mean(losses))

def evaluate(model):
    sg_scores = []

    for p in EVAL_PROMPTS:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=80)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        sg_scores.append(singaporean_judge(text))

    return float(np.mean(sg_scores)), english_loss(model)
