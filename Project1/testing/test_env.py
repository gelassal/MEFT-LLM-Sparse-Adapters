import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model_name = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to("cuda")

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

print("Model loaded successfully")
print("Logits shape:", outputs.logits.shape)
print("Peak VRAM MB:", torch.cuda.max_memory_allocated() / 1024 / 1024)