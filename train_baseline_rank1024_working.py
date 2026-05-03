import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

print("BASELINE STARTED", flush=True)


class ParallelAdapter(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.WA = nn.Linear(d_model, rank)
        self.WB = nn.Linear(rank, d_model)

    def forward(self, h):
        return self.WB(torch.relu(self.WA(h)))


class BaselineParallelAdapter(nn.Module):
    def __init__(self, frozen_ffn, d_model, rank=256):
        super().__init__()
        self.frozen_ffn = frozen_ffn
        self.adapter = ParallelAdapter(d_model, rank)

        for p in self.frozen_ffn.parameters():
            p.requires_grad = False

    def forward(self, h):
        return self.frozen_ffn(h) + self.adapter(h)


def inject_baseline(model, rank=256):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        old_mlp = block.mlp
        d_model = model.config.n_embd
        block.mlp = BaselineParallelAdapter(old_mlp, d_model, rank)

    return model


def main():
    model_name = "gpt2-medium"
    rank = 1024
    max_length = 128
    batch_size = 1
    num_steps = 10
    device = "cuda"

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    print("Injecting baseline adapter...", flush=True)
    model = inject_baseline(model, rank)
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}", flush=True)

    # same fake dataset
    texts = ["The capital of France is Paris."] * 100

    encoded = []
    for text in texts:
        toks = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}
        toks["labels"] = toks["input_ids"].clone()
        encoded.append(toks)

    loader = DataLoader(encoded, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    print("Training...", flush=True)

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()

        print(f"Step {step+1} | loss={loss.item():.4f}", flush=True)

    end = time.time()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    speed = num_steps / (end - start)

    print("\nDONE", flush=True)
    print(f"Peak VRAM: {peak_vram:.2f} MB", flush=True)
    print(f"Speed: {speed:.3f} steps/sec", flush=True)


if __name__ == "__main__":
    main()