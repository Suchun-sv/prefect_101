"""
K-Means Clustering Benchmark
-----------------------------
Generates synthetic high-dimensional data with known cluster structure,
runs K-Means, and reports clustering quality metrics.
"""

import time
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def run(n_samples: int, n_clusters: int, dim: int, seed: int, output_dir: Path) -> dict:
    rng = np.random.default_rng(seed)

    console.rule(
        f"[bold blue]K-Means  n_samples={n_samples}  n_clusters={n_clusters}  dim={dim}  seed={seed}"
    )

    # --- Generate synthetic clustered data ---
    console.print(f"Generating {n_samples} samples in {dim}D with {n_clusters} clusters...")
    centers = rng.standard_normal((n_clusters, dim)) * 5.0
    labels_true = rng.integers(0, n_clusters, size=n_samples)
    noise = rng.standard_normal((n_samples, dim))
    X = centers[labels_true] + noise

    # --- K-Means (manual implementation to avoid sklearn version issues) ---
    console.print("Running K-Means...")
    t0 = time.perf_counter()
    centroids, labels_pred, inertia, n_iters = _kmeans(X, n_clusters, seed, max_iter=100)
    elapsed = time.perf_counter() - t0

    # --- Metrics ---
    cluster_sizes = [int(np.sum(labels_pred == k)) for k in range(n_clusters)]

    # Silhouette score (sampled for speed)
    sample_idx = rng.choice(n_samples, size=min(500, n_samples), replace=False)
    sil = _silhouette_sample(X[sample_idx], labels_pred[sample_idx])

    results = {
        "task": "cluster",
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "dim": dim,
        "seed": seed,
        "elapsed_sec": round(elapsed, 4),
        "n_iterations": n_iters,
        "inertia": round(float(inertia), 4),
        "silhouette_score": round(float(sil), 4),
        "cluster_sizes": cluster_sizes,
    }

    # --- Pretty print ---
    table = Table(title="K-Means Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Samples / Clusters / Dim", f"{n_samples} / {n_clusters} / {dim}")
    table.add_row("Elapsed", f"{elapsed:.4f}s")
    table.add_row("Iterations", str(n_iters))
    table.add_row("Inertia", f"{inertia:.2f}")
    table.add_row("Silhouette score", f"{sil:.4f}")
    table.add_row("Cluster sizes", str(cluster_sizes))
    console.print(table)

    # --- Save centroids + results ---
    np.save(output_dir / f"cluster_centroids_k{n_clusters}_d{dim}_seed{seed}.npy", centroids)
    out_path = output_dir / f"cluster_k{n_clusters}_d{dim}_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Saved to {out_path}")

    return results


def _kmeans(X: np.ndarray, k: int, seed: int, max_iter: int = 100):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    for i in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        if np.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    inertia = sum(
        float(np.sum((X[labels == j] - centroids[j]) ** 2))
        for j in range(k) if np.any(labels == j)
    )
    return centroids, labels, inertia, i + 1


def _silhouette_sample(X: np.ndarray, labels: np.ndarray) -> float:
    n = len(X)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False
        if own_mask.sum() == 0:
            continue
        a = float(np.mean(np.linalg.norm(X[i] - X[own_mask], axis=1)))
        b = min(
            float(np.mean(np.linalg.norm(X[i] - X[labels == c], axis=1)))
            for c in unique if c != own
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0
