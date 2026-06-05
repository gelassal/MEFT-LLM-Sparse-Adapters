import matplotlib.pyplot as plt

# Project 2 results so far

baseline_ranks = [256, 512, 1024, 2048]
baseline_vram = [1922.65, 2070.22, 2379.27, 3345.12]
baseline_speed = [15.511, 14.099, 13.118, 10.669]
baseline_em = [0.000, 0.000, 0.000, 0.060]
baseline_containment = [0.320, 0.240, 0.260, 0.220]

meft_ranks = [1024, 2048, 4096, 8192]
meft_vram = [3280.55, 3280.55, 3280.55, 3279.55]
meft_speed = [0.735, 0.684, 0.544, 0.348]
meft_em = [0.040, 0.060, 0.060, 0.060]
meft_containment = [0.300, 0.260, 0.260, 0.420]


# Plot 1: Rank vs VRAM
plt.figure()
plt.plot(baseline_ranks, baseline_vram, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_vram, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Peak VRAM Usage (MB)")
plt.title("Project 2: Adapter Rank vs Peak VRAM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("p2_rank_vs_vram.png", dpi=300)
plt.show()


# Plot 2: Rank vs Training Speed
plt.figure()
plt.plot(baseline_ranks, baseline_speed, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_speed, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Training Speed (steps/sec)")
plt.title("Project 2: Adapter Rank vs Training Speed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("p2_rank_vs_speed.png", dpi=300)
plt.show()


# Plot 3: Rank vs Exact Match
plt.figure()
plt.plot(baseline_ranks, baseline_em, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_em, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Exact Match")
plt.title("Project 2: Adapter Rank vs Exact Match")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("p2_rank_vs_exact_match.png", dpi=300)
plt.show()


# Plot 4: Rank vs Containment Match
plt.figure()
plt.plot(baseline_ranks, baseline_containment, marker="o", label="Baseline GPU Adapter")
plt.plot(meft_ranks, meft_containment, marker="o", label="MEFT CPU Sparse Adapter")
plt.xscale("log", base=2)
plt.xlabel("Adapter Rank")
plt.ylabel("Containment Match")
plt.title("Project 2: Adapter Rank vs Containment Match")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("p2_rank_vs_containment.png", dpi=300)
plt.show()


print("Saved plots:")
print("p2_rank_vs_vram.png")
print("p2_rank_vs_speed.png")
print("p2_rank_vs_exact_match.png")
print("p2_rank_vs_containment.png")