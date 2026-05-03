import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ---- Paste your adapter code here (with small fix) ----

class SparseCPUAdapter(nn.Module):
    def __init__(self, d_model, rank, top_k):
        super().__init__()
        self.rank = rank
        self.top_k = top_k

        self.WA = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.WB = nn.Parameter(torch.zeros(rank, d_model))

        self.WA.data = self.WA.data.cpu()
        self.WB.data = self.WB.data.cpu()

    def forward(self, h):
        device = h.device
        B, L, D = h.shape

        # keep gradients
        h_cpu = h.to("cpu")

        scores = h_cpu @ self.WA.T
        topk_idx = torch.topk(scores, self.top_k, dim=-1).indices

        WA_k = self.WA[topk_idx]
        WB_k = self.WB[topk_idx]

        WA_k = WA_k.to(device)
        WB_k = WB_k.to(device)

        h_gpu = h.unsqueeze(-2)

        activated = torch.relu(h_gpu @ WA_k.transpose(-1, -2))
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
        self.adapter.WA.data = self.adapter.WA.data.cpu()
        self.adapter.WB.data = self.adapter.WB.data.cpu()
        return self.frozen_ffn(h) + self.adapter(h)


# ---- Load GPT-2 ----

model = AutoModelForCausalLM.from_pretrained("gpt2-medium", local_files_only=True)

for p in model.parameters():
    p.requires_grad = False

# ---- Inject adapters ----

for i, block in enumerate(model.transformer.h):
    old_mlp = block.mlp
    d_model = model.config.n_embd  # GPT-2 hidden size

    block.mlp = MEFTParallelAdapter(old_mlp, d_model)

print("Adapters injected!")

# ---- Move model to GPU ----
model = model.to("cuda")

# ---- Test forward pass ----
x = torch.randint(0, 50257, (1, 10)).cuda()

with torch.no_grad():
    out = model(input_ids=x)

print("Forward pass success!")
print("Logits:", out.logits.shape)

# ---- Check trainable params ----
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable params: {trainable}")
print(f"Total params: {total}")