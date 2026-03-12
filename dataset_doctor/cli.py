"""Command-line interface for dataset-doctor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from dataset_doctor.core.analyzer import analyze
from dataset_doctor.core.cleaner import clean

app = typer.Typer(
    name="dataset-doctor",
    help="Automatically diagnose and clean messy datasets.",
    add_completion=False,
)
console = Console()


@app.command()
def diagnose(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV)."),
) -> None:
    """Run all diagnostic checks and print a summary report."""
    report = analyze(str(file))
    console.print(Markdown(report.summary()))


@app.command()
def report(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV)."),
) -> None:
    """Alias for `diagnose` — print the full diagnosis report."""
    report_obj = analyze(str(file))
    console.print(Markdown(report_obj.summary()))


@app.command(name="clean")
def clean_cmd(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV)."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save the cleaned CSV."
    ),
    normalize: bool = typer.Option(
        False, "--normalize", "-n", help="Apply Min-Max normalization."
    ),
) -> None:
    """Run the automatic cleaning pipeline."""
    out = str(output) if output else None
    df = clean(str(file), output_path=out, do_normalize=normalize)
    console.print(f"[green]✔[/green] Cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    if output:
        console.print(f"[green]✔[/green] Saved to {output}")
    else:
        console.print("[dim]Tip: use --output cleaned.csv to save the result.[/dim]")


if __name__ == "__main__":
    app()
