# GRPO Fine-tuning on GSM8K

Post-training a 1.5B model on grade-school math using GRPO (Group Relative Policy Optimization), reward functions only — no judge model.

## Setup

**Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) in 4-bit + LoRA rank 16 (1.18% of weights trained)  
**Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) — 7,473 grade-school math problems  
**Trained adapter:** [shiva-sankeerth/qwen2.5-1.5b-grpo-gsm8k](https://huggingface.co/shiva-sankeerth/qwen2.5-1.5b-grpo-gsm8k)  
**GitHub:** [ShivaSankeerth/llm-reasoning](https://github.com/ShivaSankeerth/llm-reasoning)

## Rewards

| Signal | Value |
|--------|-------|
| Correct final number | +2.0 |
| `<reasoning>` + `<answer>` tags present | +0.5 |

No judge model — all rewards are pure Python regex checks.

## Training

```bash
pip install unsloth trl vllm
python train_grpo.py
```

Config: 8 generations per question, batch size 8, 100 steps, lr 5e-6.

## Results

| Model | GSM8K Test Accuracy |
|-------|-------------------|
| Base (Qwen2.5-1.5B-Instruct) | 0.0%* |
| + GRPO 100 steps | **29.2%** |

*Base scores 0% with tag-based extraction since it never uses `<answer>` tags without training.

## Benchmark

```bash
python benchmark.py --n 1319   # full test set
python benchmark.py --n 200    # quick check
```

## Output format

The model is trained to respond:

```
<reasoning>
step-by-step working...
</reasoning>
<answer>
42
</answer>
```
