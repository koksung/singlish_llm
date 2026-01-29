
# Singlish LoRA Evolution Demo

This is a **brown-bag friendly** demo showing how evolutionary algorithms
(NSGA-II) can be used to evolve **LoRA hyperparameters** for a LLaMA-style model
to induce a Singaporean (Singlish) identity *without prompting*.

## What this is
- Population-based search over LoRA configs
- Short-budget fine-tuning per candidate
- Multi-objective fitness (identity vs fluency)
- NSGA-II selection

## What this is NOT
- Full convergence training
- Production-quality evaluation

Designed for **conceptual clarity and live demos**.

## Run
```bash
python evolve.py
```
