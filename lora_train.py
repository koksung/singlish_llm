import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LOCAL_FILES_ONLY,
    MAX_SEQ_LEN,
    TRAIN_BATCH_SIZE,
    TRAIN_TEXTS,
    USE_GPU_IF_AVAILABLE,
)


def _pick_device(force_gpu: bool | None = None) -> torch.device:
    """
    force_gpu:
      - None: follow config.USE_GPU_IF_AVAILABLE
      - True: use CUDA if available else CPU
      - False: force CPU
    """
    want_gpu = USE_GPU_IF_AVAILABLE if force_gpu is None else force_gpu
    if want_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_tokenizer_defaults(tokenizer):
    # Many causal LMs (e.g. LLaMA) don't define pad_token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _iter_minibatches(tokenizer, texts, batch_size: int, device: torch.device):
    # Simple endless stream of tokenized batches.
    while True:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            yield {k: v.to(device) for k, v in enc.items()}

def train_and_generate(
    lora_cfg,
    prompts,
    steps,
    *,
    force_gpu: bool | None = None,
    local_files_only: bool | None = None,
):
    device = _pick_device(force_gpu)
    local_only = LOCAL_FILES_ONLY if local_files_only is None else local_files_only
    model = AutoModelForCausalLM.from_pretrained(
        lora_cfg["model_name"],
        torch_dtype=torch.float16 if device.type == "cuda" else None,
        local_files_only=local_only,
    )
    tokenizer = _set_tokenizer_defaults(
        AutoTokenizer.from_pretrained(
            lora_cfg["model_name"],
            local_files_only=local_only,
        )
    )

    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["targets"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_cfg)
    model.to(device)
    model.train()

    # Minimal but REAL training loop: next-token LM loss on tiny synthetic texts.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = _iter_minibatches(tokenizer, TRAIN_TEXTS, TRAIN_BATCH_SIZE, device)

    for _ in range(steps):
        batch = next(batches)
        # Standard causal LM training: labels = input_ids.
        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    model.eval()
    outputs = []
    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return outputs
