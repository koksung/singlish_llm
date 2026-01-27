from datasets import Dataset

SEED_PROMPTS = [
    "Explain what CPF is.",
    "Why do Singaporeans complain about weather?",
    "Explain ERP to a tourist.",
    "What is NS in Singapore?",
]

def seed_dataset():
    texts = [
        f"### Instruction:\n{p}\n### Response:\n"
        for p in SEED_PROMPTS
    ]
    return Dataset.from_dict({"text": texts})
