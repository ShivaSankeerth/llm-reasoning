from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import re

# ── Model ──────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── Dataset ────────────────────────────────────────────────────────────────────
# No explicit <reasoning> tags — Qwen3's <think> blocks handle chain of thought.
# Only ask for <answer> tags so the reward signal is clean.
SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Think through the problem carefully, then give your final answer in <answer> tags.\n"
    "Example: <answer>42</answer>"
)

dataset = load_dataset("openai/gsm8k", "main", split="train")

def format_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }

dataset = dataset.map(format_example, remove_columns=["question"])

# ── Reward functions ───────────────────────────────────────────────────────────
def get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion

def extract_model_answer(text):
    m = re.search(r"<answer>\s*([\d,.\-]+)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).replace(",", "").strip() if m else None

def extract_gsm8k_gold(answer_str):
    m = re.search(r"####\s*([\d,.\-]+)", answer_str)
    return m.group(1).replace(",", "").strip() if m else None

def reward_correctness(completions, answer, **kwargs):
    rewards = []
    for completion, gold_str in zip(completions, answer):
        pred = extract_model_answer(get_text(completion))
        gold = extract_gsm8k_gold(gold_str)
        rewards.append(2.0 if (pred and gold and pred == gold) else 0.0)
    return rewards

def reward_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = get_text(completion)
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        rewards.append(0.5 if has_answer else 0.0)
    return rewards

# ── GRPO training ──────────────────────────────────────────────────────────────
# Reduced batch size to 4 (8B model + think blocks = longer sequences)
config = GRPOConfig(
    output_dir="grpo-qwen3-gsm8k",
    num_generations=8,
    max_completion_length=1024,  # room for <think> blocks
    per_device_train_batch_size=4,
    max_steps=100,
    learning_rate=5e-6,
    logging_steps=1,
    save_steps=50,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_correctness, reward_format],
    args=config,
    train_dataset=dataset,
)

trainer.train()
print("Done! Model saved to grpo-qwen3-gsm8k/")
