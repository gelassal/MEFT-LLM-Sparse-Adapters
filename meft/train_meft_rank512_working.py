import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


print("SCRIPT STARTED", flush=True)


class SparseCPUAdapter(nn.Module):
    def __init__(self, d_model, rank, top_k):
        super().__init__()
        self.rank = rank
        self.top_k = top_k

        self.WA = nn.Parameter(torch.randn(rank, d_model, device="cpu") * 0.01)
        self.WB = nn.Parameter(torch.zeros(rank, d_model, device="cpu"))

    def forward(self, h):
        device = h.device

        self.WA.data = self.WA.data.cpu()
        self.WB.data = self.WB.data.cpu()

        h_cpu = h.to("cpu")
        scores = h_cpu @ self.WA.T
        topk_idx = torch.topk(scores, self.top_k, dim=-1).indices

        WA_k = self.WA[topk_idx].to(device)
        WB_k = self.WB[topk_idx].to(device)

        activated = torch.relu(h.unsqueeze(-2) @ WA_k.transpose(-1, -2))
        out = (activated @ WB_k).squeeze(-2)

        return out


class MEFTParallelAdapter(nn.Module):
    def __init__(self, frozen_ffn, d_model, rank=512, top_k=64):
        super().__init__()
        self.frozen_ffn = frozen_ffn
        self.adapter = SparseCPUAdapter(d_model, rank, top_k)

        for p in self.frozen_ffn.parameters():
            p.requires_grad = False

    def forward(self, h):
        return self.frozen_ffn(h) + self.adapter(h)


def inject_meft(model, rank=512, top_k=64):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        old_mlp = block.mlp
        d_model = model.config.n_embd
        block.mlp = MEFTParallelAdapter(old_mlp, d_model, rank, top_k)

    return model


def main():
    print("INSIDE MAIN", flush=True)

    model_name = "gpt2-medium"
    rank = 512
    top_k = 64
    max_length = 128
    batch_size = 1
    num_steps = 10
    device = "cuda"

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    print("Injecting MEFT adapters...", flush=True)
    model = inject_meft(model, rank=rank, top_k=top_k)
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable:,}", flush=True)
    print(f"Total params: {total:,}", flush=True)

    print("Creating tiny local dataset...", flush=True)

    texts = [
        "Question: What is MEFT?\nContext: MEFT is a memory efficient fine tuning method.\nAnswer: memory efficient fine tuning",
        "Question: Where are adapter weights stored?\nContext: In MEFT, most adapter weights are stored in CPU memory.\nAnswer: CPU memory",
        "Question: What does TopK do?\nContext: TopK selects only the most activated adapter neurons.\nAnswer: selects active neurons",
        "Question: Why use sparse adapters?\nContext: Sparse adapters reduce GPU memory usage.\nAnswer: reduce GPU memory",
    ] * 25

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

    print("Starting training loop...", flush=True)

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}/{num_steps} | loss = {loss.item():.4f}", flush=True)

    end = time.time()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    steps_per_sec = num_steps / (end - start)

    print("\nDONE", flush=True)
    print(f"Peak VRAM: {peak_vram:.2f} MB", flush=True)
    print(f"Training speed: {steps_per_sec:.3f} steps/sec", flush=True)


if __name__ == "__main__":
    main()
