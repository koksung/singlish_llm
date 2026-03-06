import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ADAPTER_PATH,
    LOCAL_FILES_ONLY,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS,
    MAX_SEQ_LEN,
    MODEL_NAME,
    TRAIN_CONVERSATIONS,
    USE_GPU_IF_AVAILABLE,
)


def get_device() -> torch.device:
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_base_model():
    """Load the base instruct model and tokenizer. Returns (model, tokenizer, device)."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=LOCAL_FILES_ONLY
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, messages, max_new_tokens=120):
    """
    Generate a single assistant response.

    messages: list of dicts with 'role' and 'content' keys, e.g.
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    Returns the assistant reply as a string.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def train_lora(model, tokenizer, device, steps: int = 200):
    """
    Inject LoRA adapters into the model and fine-tune on Singlish conversations.

    Training data has NO system prompt — we want the identity in the weights,
    not prompted in. Returns the LoRA-wrapped model (eval mode).
    """
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()

    # Format each conversation using the model's chat template (no system prompt)
    train_texts = []
    for conv in TRAIN_CONVERSATIONS:
        messages = [
            {"role": "user", "content": conv["user"]},
            {"role": "assistant", "content": conv["assistant"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        train_texts.append(text)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    step = 0
    while step < steps:
        for text in train_texts:
            if step >= steps:
                break
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc, labels=enc["input_ids"]).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps}  loss={loss.item():.4f}")

    model.eval()
    return model


def save_adapter(model, path: str = ADAPTER_PATH):
    """Save only the LoRA adapter weights (not the full model)."""
    model.save_pretrained(path)
    print(f"Adapter saved to '{path}'")


def load_adapter(base_model, path: str = ADAPTER_PATH):
    """Load a saved LoRA adapter on top of the base model."""
    return PeftModel.from_pretrained(base_model, path)
