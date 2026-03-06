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

    # bfloat16 is safe for training (same exponent range as fp32, no underflow).
    # float16 requires a GradScaler and is avoided here.
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
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
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    # apply_chat_template returns a plain tensor (older transformers) or a
    # BatchEncoding with input_ids + attention_mask (newer transformers).
    # Normalise to a plain dict and move everything to device.
    if isinstance(encoded, torch.Tensor):
        model_inputs = {"input_ids": encoded.to(device)}
    else:
        model_inputs = {k: v.to(device) for k, v in encoded.items()}

    prompt_len = model_inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = out[0][prompt_len:]
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
    """Load a saved LoRA adapter on top of the base model.
    Safe to call multiple times — unwraps any existing adapter first."""
    if hasattr(base_model, "peft_config"):
        base_model = base_model.base_model.model
    return PeftModel.from_pretrained(base_model, path)
