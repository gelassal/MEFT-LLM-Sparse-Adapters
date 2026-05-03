import matplotlib.pyplot as plt

# Results from runs
baseline_ranks = [256, 512, 1024, 2048, 4096, 8192]
baseline_vram = [1922.65, 2070.22, 2379.27, 3345.12, 5260.68, 9102.55]
baseline_speed = [10.323, 10.136, 9.850, 8.104, 5.954, 0.881]

meft_ranks = [512, 1024, 2048, 4096, 8192]
meft_vram = [3280.05, 3280.55, 3280.55, 3280.55, 3279.55]
meft_speed = [0.610, 0.589, 0.522, 0.410, 0.285]

# Plot 1: Rank vs VRAM
plt.figure()
plt.plot(baseline_ranks, baseline_vram, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_vram, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Peak VRAM Usage (MB)")
plt.title("Adapter Rank vs Peak VRAM Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rank_vs_vram.png", dpi=300)
plt.show()

# Plot 2: Rank vs Training Speed
plt.figure()
plt.plot(baseline_ranks, baseline_speed, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_speed, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Training Speed (steps/sec)")
plt.title("Adapter Rank vs Training Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rank_vs_speed.png", dpi=300)
plt.show()

print("Saved plots:")
print("rank_vs_vram.png")
print("rank_vs_speed.png")
