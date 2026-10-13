"""Plot results.csv from ./sgemm -> results.png (for the README)."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

for name, g in df.groupby("kernel", sort=False):
    ax1.plot(g["N"], g["gflops"], marker="o", label=name,
             linestyle="--" if name == "cuBLAS" else "-")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("Matrix size N (N x N)")
ax1.set_ylabel("GFLOPS")
ax1.set_title("SGEMM throughput vs size")
ax1.grid(alpha=0.3)
ax1.legend()

n_max = df["N"].max()
big = df[df["N"] == n_max]
bars = ax2.barh(big["kernel"], big["pct_cublas"])
ax2.bar_label(bars, fmt="%.1f%%", padding=3)
ax2.invert_yaxis()
ax2.set_xlabel(f"% of cuBLAS at N = {n_max}")
ax2.set_title("Each optimization step")
ax2.set_xlim(0, 115)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved results.png")
