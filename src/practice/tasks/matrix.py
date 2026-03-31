"""
Matrix SVD Benchmark
--------------------
Generates a random matrix, computes its SVD, and reports statistics
about the singular value distribution (effective rank, explained variance).
"""

import time
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def run(size: int, seed: int, output_dir: Path) -> dict:
    rng = np.random.default_rng(seed)

    console.rule(f"[bold blue]Matrix SVD  size={size}  seed={seed}")

    # --- Generate matrix ---
    console.print(f"Generating {size}x{size} random matrix...")
    A = rng.standard_normal((size, size))

    # --- SVD ---
    console.print("Computing SVD...")
    t0 = time.perf_counter()
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    elapsed = time.perf_counter() - t0

    # --- Statistics ---
    total_variance = float(np.sum(S ** 2))
    explained = np.cumsum(S ** 2) / total_variance
    rank_90 = int(np.searchsorted(explained, 0.90)) + 1
    rank_99 = int(np.searchsorted(explained, 0.99)) + 1

    results = {
        "task": "matrix",
        "size": size,
        "seed": seed,
        "elapsed_sec": round(elapsed, 4),
        "singular_values": {
            "max": float(S[0]),
            "min": float(S[-1]),
            "mean": float(S.mean()),
            "std": float(S.std()),
        },
        "effective_rank_90pct": rank_90,
        "effective_rank_99pct": rank_99,
        "condition_number": float(S[0] / S[-1]),
    }

    # --- Pretty print ---
    table = Table(title="SVD Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Matrix size", f"{size} x {size}")
    table.add_row("Elapsed", f"{elapsed:.4f}s")
    table.add_row("Max singular value", f"{S[0]:.4f}")
    table.add_row("Min singular value", f"{S[-1]:.4f}")
    table.add_row("Condition number", f"{results['condition_number']:.2f}")
    table.add_row("Effective rank (90%)", str(rank_90))
    table.add_row("Effective rank (99%)", str(rank_99))
    console.print(table)

    # --- Save ---
    out_path = output_dir / f"matrix_size{size}_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Saved to {out_path}")

    return results
