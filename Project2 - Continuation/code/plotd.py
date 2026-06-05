import matplotlib.pyplot as plt
import pandas as pd

# Dynamic K results, rank = 1024
methods = ["Fixed K=64", "Dynamic 3%", "Dynamic 5%", "Dynamic 10%"]
avg_k = [64, 30, 51, 102]
vram = [3280.55, 2520.28, 2998.40, 4188.62]
speed = [0.735, 1.462, 0.984, 0.500]
em = [0.040, 0.000, 0.040, 0.060]
containment = [0.300, 0.340, 0.360, 0.400]

# Make results table
df = pd.DataFrame({
    "Method": methods,
    "Avg K": avg_k,
    "Peak VRAM (MB)": vram,
    "Speed (steps/sec)": speed,
    "Exact Match": em,
    "Containment": containment,
})

print(df)
df.to_csv("dynamic_k_results_table.csv", index=False)


# Plot 1: K strategy vs VRAM
plt.figure()
plt.bar(methods, vram)
plt.ylabel("Peak VRAM (MB)")
plt.title("Dynamic K: VRAM Usage")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("dynamic_k_vram.png", dpi=300)
plt.show()


# Plot 2: K strategy vs speed
plt.figure()
plt.bar(methods, speed)
plt.ylabel("Training Speed (steps/sec)")
plt.title("Dynamic K: Training Speed")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("dynamic_k_speed.png", dpi=300)
plt.show()


# Plot 3: K strategy vs Exact Match
plt.figure()
plt.bar(methods, em)
plt.ylabel("Exact Match")
plt.title("Dynamic K: Exact Match")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("dynamic_k_exact_match.png", dpi=300)
plt.show()


# Plot 4: K strategy vs Containment
plt.figure()
plt.bar(methods, containment)
plt.ylabel("Containment Match")
plt.title("Dynamic K: Containment Match")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("dynamic_k_containment.png", dpi=300)
plt.show()


# Plot 5: Avg K vs speed/containment
plt.figure()
plt.plot(avg_k, speed, marker="o", label="Speed")
plt.xlabel("Average K")
plt.ylabel("Training Speed (steps/sec)")
plt.title("Dynamic K: Average K vs Training Speed")
plt.grid(True)
plt.tight_layout()
plt.savefig("dynamic_k_avgk_vs_speed.png", dpi=300)
plt.show()

plt.figure()
plt.plot(avg_k, containment, marker="o", label="Containment")
plt.xlabel("Average K")
plt.ylabel("Containment Match")
plt.title("Dynamic K: Average K vs Containment")
plt.grid(True)
plt.tight_layout()
plt.savefig("dynamic_k_avgk_vs_containment.png", dpi=300)
plt.show()

print("\nSaved:")
print("dynamic_k_results_table.csv")
print("dynamic_k_vram.png")
print("dynamic_k_speed.png")
print("dynamic_k_exact_match.png")
print("dynamic_k_containment.png")
print("dynamic_k_avgk_vs_speed.png")
print("dynamic_k_avgk_vs_containment.png")