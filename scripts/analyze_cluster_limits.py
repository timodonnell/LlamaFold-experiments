"""Analyze how limiting documents per struct_cluster_id affects training set size.

Usage:
    uv run python scripts/analyze_cluster_limits.py
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset


def main():
    print("Loading dataset...")
    ds = load_dataset("timodonnell/protein-docs", "default", split="train")
    cluster_ids = ds["struct_cluster_id"]

    counts = Counter(cluster_ids)
    total = len(cluster_ids)
    n_clusters = len(counts)

    print(f"\nTotal documents: {total:,}")
    print(f"Unique struct_cluster_ids: {n_clusters:,}")

    limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]

    print(f"\n{'Limit':>8s}  {'Documents':>12s}  {'% of total':>10s}  {'Clusters used':>14s}")
    print("-" * 50)

    for limit in limits:
        n_docs = sum(min(c, limit) for c in counts.values())
        n_clusters_used = sum(1 for c in counts.values() if c > 0)
        print(f"{limit:>8d}  {n_docs:>12,}  {100 * n_docs / total:>9.1f}%  {n_clusters_used:>14,}")

    # No limit
    print(f"{'inf':>8s}  {total:>12,}  {100.0:>9.1f}%  {n_clusters:>14,}")

    # Extra stats
    print(f"\nCluster size distribution:")
    vals = sorted(counts.values())
    import numpy as np

    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct}: {np.percentile(vals, pct):.0f}")
    print(f"  max: {max(vals)}")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: docs available vs limit
    all_limits = limits + [total]  # use total as stand-in for inf
    all_docs = [sum(min(c, lim) for c in counts.values()) for lim in limits] + [total]
    all_pcts = [100 * d / total for d in all_docs]
    x_labels = [str(l) for l in limits] + ["inf"]

    ax = axes[0]
    bars = ax.bar(range(len(x_labels)), [d / 1e6 for d in all_docs], color="#5C9BD5")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Max docs per struct_cluster_id")
    ax.set_ylabel("Training documents (millions)")
    ax.set_title("Training Set Size vs Cluster Limit")
    for bar, pct in zip(bars, all_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    # Right: cluster size distribution (histogram)
    ax = axes[1]
    clipped = np.clip(vals, 1, 100)
    ax.hist(clipped, bins=100, color="#7BC47F", edgecolor="none")
    ax.set_xlabel("Cluster size (clipped at 100)")
    ax.set_ylabel("Number of clusters")
    ax.set_title(f"Cluster Size Distribution ({n_clusters:,} clusters)")
    ax.axvline(np.median(vals), color="red", linewidth=1.5, linestyle="--",
               label=f"Median={np.median(vals):.0f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = "outputs/cluster_limit_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
