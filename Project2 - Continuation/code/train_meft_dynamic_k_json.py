import time
import json
import re
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

print("MEFT DYNAMIC-K SCRIPT STARTED", flush=True)


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

        k = min(self.top_k, self.rank)
        topk_idx = torch.topk(scores, k, dim=-1).indices

        WA_k = self.WA[topk_idx].to(device)
        WB_k = self.WB[topk_idx].to(device)

        activated = torch.relu(h.unsqueeze(-2) @ WA_k.transpose(-1, -2))
        out = (activated @ WB_k).squeeze(-2)

        return out


class MEFTParallelAdapter(nn.Module):
    def __init__(self, frozen_ffn, d_model, rank=1024, top_k=64):
        super().__init__()
        self.frozen_ffn = frozen_ffn
        self.adapter = SparseCPUAdapter(d_model, rank, top_k)

        for p in self.frozen_ffn.parameters():
            p.requires_grad = False

    def forward(self, h):
        return self.frozen_ffn(h) + self.adapter(h)


def inject_meft(model, rank=1024, top_k=64):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        old_mlp = block.mlp
        d_model = model.config.n_embd
        block.mlp = MEFTParallelAdapter(old_mlp, d_model, rank, top_k)

    return model


def build_json_dataset(tokenizer, json_path, n_examples=500, max_length=128):
    print(f"Loading JSON dataset from {json_path}...", flush=True)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_examples = json.load(f)

    raw_examples = raw_examples[:n_examples]
    encoded = []

    print(f"Tokenizing {len(raw_examples)} examples...", flush=True)

    for ex in raw_examples:
        text = (
            f"Question: {ex['question']}\n"
            f"Context: {ex['context']}\n"
            f"Answer: {ex['answer']}"
        )

        toks = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        encoded.append({
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "labels": toks["input_ids"].squeeze(0).clone(),
        })

    print(f"Prepared {len(encoded)} training examples.", flush=True)
    return encoded


def calibrate_dynamic_k(model, loader, device, rank, sparsity_target=0.05, n_batches=10, min_k=16):
    print("\nStarting Dynamic-K calibration...", flush=True)
    print(f"sparsity_target={sparsity_target}", flush=True)

    layer_active_fracs = {i: [] for i in range(len(model.transformer.h))}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            h = inputs[0].detach().cpu()

            adapter = module.adapter
            scores = h @ adapter.WA.T

            active_frac = (scores > 0).float().mean().item()
            layer_active_fracs[layer_idx].append(active_frac)

        return hook

    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.mlp.register_forward_hook(make_hook(i)))

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    for h in hooks:
        h.remove()

    avg_fracs = {}
    for i, vals in layer_active_fracs.items():
        avg_fracs[i] = sum(vals) / len(vals) if len(vals) > 0 else 0.0

    global_avg = sum(avg_fracs.values()) / len(avg_fracs)
    if global_avg <= 0:
        global_avg = 1e-6

    k_per_layer = {}

    print("\nCalibrated per-layer K values:", flush=True)

    for i, block in enumerate(model.transformer.h):
        avg_active = avg_fracs[i]

        # Per-layer adjusted K:
        # target fraction * rank, scaled by how active this layer is relative to average
        k = int(rank * sparsity_target * (avg_active / global_avg))
        k = max(min_k, k)
        k = min(rank, k)

        block.mlp.adapter.top_k = k
        k_per_layer[i] = k

        print(
            f"Layer {i:02d}: avg_active={avg_active:.4f}, K={k}",
            flush=True
        )

    model.train()
    return k_per_layer


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def exact_match(prediction, answers):
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(ans) for ans in answers))


def containment_match(prediction, answers):
    pred = normalize_answer(prediction)
    return int(any(normalize_answer(ans) in pred for ans in answers))


def evaluate_em(model, tokenizer, json_path="squad_val_200.json", n_eval=50, device="cuda"):
    print(f"\nEvaluating on {n_eval} examples...", flush=True)

    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)[:n_eval]

    model.eval()

    em_total = 0
    contain_total = 0

    for i, ex in enumerate(examples):
        prompt = (
            f"Question: {ex['question']}\n"
            f"Context: {ex['context'][:500]}\n"
            f"Answer with only the exact short answer:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][prompt_len:]
        pred = tokenizer.decode(generated, skip_special_tokens=True)
        pred = pred.split("\n")[0].strip()

        em = exact_match(pred, ex["answers"])
        contain = containment_match(pred, ex["answers"])

        em_total += em
        contain_total += contain

        if i < 3:
            print(f"\nExample {i + 1}", flush=True)
            print("Pred:", pred, flush=True)
            print("Gold:", ex["answers"][0], flush=True)
            print("EM:", em, flush=True)
            print("Containment:", contain, flush=True)

    em_score = em_total / len(examples)
    contain_score = contain_total / len(examples)

    print(f"\nExact Match: {em_score:.4f} ({em_total}/{len(examples)})", flush=True)
    print(f"Containment Match: {contain_score:.4f} ({contain_total}/{len(examples)})", flush=True)

    model.train()

    return em_score, contain_score


def main():
    model_name = "gpt2-medium"
    train_json_path = "squad_train_500.json"
    val_json_path = "squad_val_200.json"

    # Main experiment settings
    rank = 1024
    initial_top_k = 64

    # Dynamic-K settings
    use_dynamic_k = True
    sparsity_target = 0.1   # Change to 0.03, 0.05, 0.10
    calibration_batches = 10
    min_k = 16

    max_length = 128
    n_examples = 500
    num_steps = 100
    batch_size = 1
    lr = 1e-4
    n_eval = 50
    device = "cuda"

    print("Config:", flush=True)
    print(f"  method=MEFT Dynamic-K Adapter", flush=True)
    print(f"  rank={rank}", flush=True)
    print(f"  initial_top_k={initial_top_k}", flush=True)
    print(f"  use_dynamic_k={use_dynamic_k}", flush=True)
    print(f"  sparsity_target={sparsity_target}", flush=True)
    print(f"  steps={num_steps}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    print("Injecting MEFT adapters...", flush=True)
    model = inject_meft(model, rank=rank, top_k=initial_top_k)
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable:,}", flush=True)
    print(f"Total params: {total:,}", flush=True)

    train_data = build_json_dataset(
        tokenizer,
        json_path=train_json_path,
        n_examples=n_examples,
        max_length=max_length,
    )

    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    k_per_layer = None

    if use_dynamic_k:
        k_per_layer = calibrate_dynamic_k(
            model=model,
            loader=loader,
            device=device,
            rank=rank,
            sparsity_target=sparsity_target,
            n_batches=calibration_batches,
            min_k=min_k,
        )
    else:
        print("Using fixed K for all layers.", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    print("\nStarting MEFT Dynamic-K SQuAD training...", flush=True)

    step = 0
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()

            step += 1

            if step % 10 == 0 or step == 1:
                print(f"Step {step}/{num_steps} | loss = {loss.item():.4f}", flush=True)

    end = time.time()

    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    speed = num_steps / (end - start)

    print("\nDONE", flush=True)
    print(f"Peak VRAM: {peak_vram:.2f} MB", flush=True)
    print(f"Training speed: {speed:.3f} steps/sec", flush=True)

    em, contain = evaluate_em(
        model,
        tokenizer,
        json_path=val_json_path,
        n_eval=n_eval,
        device=device,
    )

    print("\nFINAL SUMMARY", flush=True)
    print(f"Method: MEFT Dynamic-K Adapter", flush=True)
    print(f"Rank: {rank}", flush=True)
    print(f"Sparsity target: {sparsity_target}", flush=True)
    print(f"Peak VRAM: {peak_vram:.2f} MB", flush=True)
    print(f"Speed: {speed:.3f} steps/sec", flush=True)
    print(f"Exact Match: {em:.4f}", flush=True)
    print(f"Containment Match: {contain:.4f}", flush=True)

    if k_per_layer is not None:
        print("\nK per layer:", flush=True)
        print(k_per_layer, flush=True)


if __name__ == "__main__":
    main()