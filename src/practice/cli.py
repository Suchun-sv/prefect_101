import typer
from pathlib import Path
from typing import Optional
from rich.console import Console

app = typer.Typer(help="Prefect Practice CLI")
console = Console()


@app.command()
def matrix(
    size: int = typer.Option(500, help="Matrix dimension (size x size)"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
):
    """SVD benchmark on a random matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from practice.tasks.matrix import run
    run(size=size, seed=seed, output_dir=output_dir)


@app.command()
def cluster(
    n_samples: int = typer.Option(3000, help="Number of data points"),
    n_clusters: int = typer.Option(5, help="Number of clusters"),
    dim: int = typer.Option(64, help="Vector dimension"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
):
    """K-Means clustering benchmark on synthetic data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from practice.tasks.cluster import run
    run(n_samples=n_samples, n_clusters=n_clusters, dim=dim, seed=seed, output_dir=output_dir)


@app.command()
def similarity(
    n_vectors: int = typer.Option(10000, help="Corpus size"),
    dim: int = typer.Option(128, help="Vector dimension"),
    n_queries: int = typer.Option(100, help="Number of query vectors"),
    top_k: int = typer.Option(10, help="Top-K results to retrieve"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
):
    """Brute-force cosine similarity search benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from practice.tasks.similarity import run
    run(n_vectors=n_vectors, dim=dim, n_queries=n_queries, top_k=top_k, seed=seed, output_dir=output_dir)


if __name__ == "__main__":
    app()
