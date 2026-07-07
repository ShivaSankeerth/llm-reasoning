"""
Benchmark base vs GRPO-trained Qwen3-8B on GSM8K + MATH.
Usage: python benchmark_qwen3.py [--gsm8k_n 200] [--math_n 200]
"""
import re
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Think through the problem carefully, then give your final answer in <answer> tags.\n"
    "Example: <answer>42</answer>"
)

def get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion

def extract_model_answer(text):
    m = re.search(r"<answer>\s*(.+?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None

def normalize(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\", "").strip()
    try:
        return str(float(s))
    except ValueError:
        return s.lower()

def extract_gsm8k_gold(answer_str):
    m = re.search(r"####\s*([\d,.\-]+)", answer_str)
    return m.group(1).replace(",", "").strip() if m else None

def generate(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1024,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def eval_gsm8k(model, tokenizer, examples):
    correct = 0
    for i, ex in enumerate(examples):
        response = generate(model, tokenizer, ex["question"])
        pred = normalize(extract_model_answer(response))
        gold = normalize(extract_gsm8k_gold(ex["answer"]))
        if pred and gold and pred == gold:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  GSM8K [{i+1}/{len(examples)}] accuracy: {correct/(i+1):.1%}")
    return correct / len(examples)

def eval_math(model, tokenizer, examples):
    correct = 0
    for i, ex in enumerate(examples):
        response = generate(model, tokenizer, ex["problem"])
        pred = normalize(extract_model_answer(response))
        gold = normalize(ex.get("answer", ""))
        if pred and gold and pred == gold:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  MATH [{i+1}/{len(examples)}] accuracy: {correct/(i+1):.1%}")
    return correct / len(examples)

def load_model(adapter_path=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-8B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k_n", type=int, default=200)
    parser.add_argument("--math_n", type=int, default=200)
    args = parser.parse_args()

    gsm8k = list(load_dataset("openai/gsm8k", "main", split="test").select(range(args.gsm8k_n)))
    math_ds = list(load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True).select(range(args.math_n)))
    print(f"GSM8K: {len(gsm8k)} examples | MATH: {len(math_ds)} examples\n")

    results = {}

    for label, adapter in [("base", None), ("trained", "grpo-qwen3-gsm8k/checkpoint-100")]:
        print("=" * 55)
        print(f"{'BASE' if not adapter else 'TRAINED'} MODEL: Qwen3-8B{' + GRPO LoRA' if adapter else ''}")
        print("=" * 55)
        model, tokenizer = load_model(adapter)

        print("→ GSM8K")
        gsm8k_acc = eval_gsm8k(model, tokenizer, gsm8k)
        print(f"  GSM8K accuracy: {gsm8k_acc:.1%}\n")

        print("→ MATH")
        math_acc = eval_math(model, tokenizer, math_ds)
        print(f"  MATH accuracy: {math_acc:.1%}\n")

        results[label] = {"gsm8k": gsm8k_acc, "math": math_acc}

        import torch, gc
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"{'':20} {'GSM8K':>10} {'MATH':>10}")
    print(f"{'Base':20} {results['base']['gsm8k']:>10.1%} {results['base']['math']:>10.1%}")
    print(f"{'Trained (GRPO)':20} {results['trained']['gsm8k']:>10.1%} {results['trained']['math']:>10.1%}")
    print(f"{'Delta':20} {results['trained']['gsm8k']-results['base']['gsm8k']:>+10.1%} {results['trained']['math']-results['base']['math']:>+10.1%}")

if __name__ == "__main__":
    main()
