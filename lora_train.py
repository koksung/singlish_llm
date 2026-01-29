
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_and_generate(lora_cfg, prompts, steps):
    model = AutoModelForCausalLM.from_pretrained(
        lora_cfg["model_name"],
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_cfg["model_name"])

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["targets"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_cfg)
    model.train()

    # Dummy training loop (brown-bag safe)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(steps):
        loss = torch.rand(1, requires_grad=True)
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    outputs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=50)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return outputs
