"""
Benchmark base vs GRPO-trained model on GSM8K test set.
Usage: python benchmark.py [--n 200]
"""
import re
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step.\n"
    "Format your response exactly as:\n"
    "<reasoning>\nyour step-by-step reasoning\n</reasoning>\n"
    "<answer>\nfinal numeric answer only\n</answer>"
)

def extract_model_answer(text):
    m = re.search(r"<answer>\s*([\d,.\-]+)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).replace(",", "").strip() if m else None

def extract_gsm8k_gold(answer_str):
    m = re.search(r"####\s*([\d,.\-]+)", answer_str)
    return m.group(1).replace(",", "").strip() if m else None

def evaluate(model, tokenizer, examples):
    correct = 0
    FastLanguageModel.for_inference(model)

    for i, ex in enumerate(examples):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        pred = extract_model_answer(response)
        gold = extract_gsm8k_gold(ex["answer"])
        if pred and gold and pred == gold:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}] running accuracy: {correct/(i+1):.1%}")

    return correct / len(examples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of test examples")
    args = parser.parse_args()

    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(test_ds.select(range(min(args.n, len(test_ds)))))
    print(f"Evaluating on {len(examples)} GSM8K test examples\n")

    # ── Base model ─────────────────────────────────────────────────────────────
    print("=" * 50)
    print("BASE MODEL: Qwen/Qwen2.5-1.5B-Instruct (no LoRA)")
    print("=" * 50)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    base_acc = evaluate(base_model, tokenizer, examples)
    print(f"\nBase accuracy: {base_acc:.1%}\n")

    del base_model
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── Trained model ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("TRAINED MODEL: + GRPO LoRA adapter")
    print("=" * 50)
    trained_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    from peft import PeftModel
    trained_model = PeftModel.from_pretrained(trained_model, "grpo-qwen-gsm8k/checkpoint-100")
    trained_acc = evaluate(trained_model, tokenizer, examples)
    print(f"\nTrained accuracy: {trained_acc:.1%}\n")

    # ── Results ────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Base model:    {base_acc:.1%}")
    print(f"Trained model: {trained_acc:.1%}")
    delta = trained_acc - base_acc
    sign = "+" if delta >= 0 else ""
    print(f"Delta:         {sign}{delta:.1%}")

if __name__ == "__main__":
    main()
