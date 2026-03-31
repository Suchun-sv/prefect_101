"""
Vector Similarity Search Benchmark
------------------------------------
Generates random unit-normalized vectors (simulating embeddings),
runs brute-force cosine similarity search for a set of query vectors,
and reports recall and timing statistics.
"""

import time
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def run(n_vectors: int, dim: int, n_queries: int, top_k: int, seed: int, output_dir: Path) -> dict:
    rng = np.random.default_rng(seed)

    console.rule(
        f"[bold blue]Similarity Search  n={n_vectors}  dim={dim}  queries={n_queries}  top_k={top_k}  seed={seed}"
    )

    # --- Generate corpus & queries ---
    console.print(f"Generating {n_vectors} corpus vectors and {n_queries} query vectors in {dim}D...")
    corpus = _normalize(rng.standard_normal((n_vectors, dim)).astype(np.float32))
    queries = _normalize(rng.standard_normal((n_queries, dim)).astype(np.float32))

    # --- Brute-force cosine similarity (matrix multiply on normalized vectors) ---
    console.print("Running brute-force search...")
    t0 = time.perf_counter()
    scores = queries @ corpus.T  # (n_queries, n_vectors)
    top_k_indices = np.argsort(-scores, axis=1)[:, :top_k]
    elapsed = time.perf_counter() - t0

    # --- Statistics ---
    top1_scores = scores[np.arange(n_queries), top_k_indices[:, 0]]
    topk_scores = np.array([scores[i, top_k_indices[i]] for i in range(n_queries)])

    results = {
        "task": "similarity",
        "n_vectors": n_vectors,
        "dim": dim,
        "n_queries": n_queries,
        "top_k": top_k,
        "seed": seed,
        "elapsed_sec": round(elapsed, 4),
        "queries_per_sec": round(n_queries / elapsed, 1),
        "top1_score": {
            "mean": round(float(top1_scores.mean()), 4),
            "std": round(float(top1_scores.std()), 4),
            "max": round(float(top1_scores.max()), 4),
        },
        "topk_mean_score": round(float(topk_scores.mean()), 4),
    }

    # --- Pretty print ---
    table = Table(title="Similarity Search Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Corpus / Queries / Dim", f"{n_vectors} / {n_queries} / {dim}")
    table.add_row("Top-K", str(top_k))
    table.add_row("Elapsed", f"{elapsed:.4f}s")
    table.add_row("Queries/sec", f"{results['queries_per_sec']:.1f}")
    table.add_row("Mean top-1 score", f"{top1_scores.mean():.4f}")
    table.add_row("Std top-1 score", f"{top1_scores.std():.4f}")
    table.add_row("Mean top-k score", f"{topk_scores.mean():.4f}")
    console.print(table)

    # --- Save top-k indices + results ---
    np.save(output_dir / f"similarity_topk_n{n_vectors}_d{dim}_seed{seed}.npy", top_k_indices)
    out_path = output_dir / f"similarity_n{n_vectors}_d{dim}_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Saved to {out_path}")

    return results
