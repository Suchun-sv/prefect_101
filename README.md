# prefect-practice

A public practice repository for testing the full Prefect pipeline.

## Tasks

| Command | Description | Key Parameters |
|---------|-------------|----------------|
| `practice matrix` | SVD on a random matrix | `--size`, `--seed` |
| `practice cluster` | K-Means on synthetic data | `--n-samples`, `--n-clusters`, `--dim`, `--seed` |
| `practice similarity` | Cosine similarity search | `--n-vectors`, `--dim`, `--n-queries`, `--top-k`, `--seed` |

## Quick Start

```bash
uv sync
uv run practice matrix --size 1000 --seed 42
uv run practice cluster --n-samples 5000 --n-clusters 8 --dim 128 --seed 1
uv run practice similarity --n-vectors 50000 --dim 256 --top-k 20 --seed 7
```

## init.sh

`init.sh` creates the `outputs/` and `logs/` directories and is idempotent — safe to run concurrently.
