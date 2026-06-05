from transformers import AutoTokenizer
from datasets import load_dataset

print("START", flush=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("squad", split="train[:1]")
ex = dataset[0]

answer = ex["answers"]["text"][0]
text = f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer: {answer}"

print("TEXT BUILT", flush=True)

toks = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt",
)

print("TOKENIZED", flush=True)
print(toks["input_ids"].shape, flush=True)
print("SUCCESS", flush=True)