"""Command-line interface for dataset-doctor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from dataset_doctor.core.analyzer import analyze
from dataset_doctor.core.cleaner import clean
from dataset_doctor.core.viewer import display_data
from dataset_doctor.utils.config_loader import DEFAULT_CONFIG

app = typer.Typer(
    name="dataset-doctor",
    help="Automatically diagnose and clean messy datasets.",
    add_completion=False,
)
console = Console()


def _print_dataframe(df) -> None:
    """Render a DataFrame in a compact rich table."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("index", justify="right", style="dim")
    for col in df.columns:
        table.add_column(str(col))

    for idx, row in df.iterrows():
        table.add_row(str(idx), *[str(value) for value in row.tolist()])

    console.print(table)


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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file for cleaning behavior.",
    ),
) -> None:
    """Run the automatic cleaning pipeline."""
    out = str(output) if output else None
    config_value = str(config) if config else None
    df = clean(str(file), output_path=out, do_normalize=normalize, config=config_value)
    console.print(f"[green]✔[/green] Cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    if output:
        console.print(f"[green]✔[/green] Saved to {output}")
    else:
        console.print("[dim]Tip: use --output cleaned.csv to save the result.[/dim]")


@app.command(name="init-config")
def init_config_cmd(
    output: Path = typer.Argument(
        Path("dataset_doctor_config.yaml"),
        help="Where to write the default config YAML.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite the config file if it already exists.",
    ),
) -> None:
    """Generate a default dataset-doctor configuration YAML file."""
    if output.exists() and not force:
        raise typer.BadParameter(
            f"Config file already exists: {output}. Use --force to overwrite."
        )

    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_CONFIG, handle, sort_keys=False)

    console.print(f"[green]✔[/green] Wrote default config to {output}")


def _display_impl(
    file: Path,
    rows: int,
    tail: bool,
    columns: Optional[str],
    all_rows: bool,
) -> None:
    column_list = [col.strip() for col in columns.split(",")] if columns else None
    if column_list:
        column_list = [col for col in column_list if col]

    df = display_data(
        str(file),
        rows=rows,
        tail=tail,
        columns=column_list,
        all_rows=all_rows,
    )

    console.print(f"[green]✔[/green] Displaying {len(df)} rows x {len(df.columns)} columns")
    _print_dataframe(df)


@app.command(name="display")
def display_cmd(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV)."),
    rows: int = typer.Option(10, "--rows", "-r", help="Number of rows to display."),
    tail: bool = typer.Option(False, "--tail", help="Display last N rows instead of first N rows."),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma-separated columns to display (e.g. age,salary,city).",
    ),
    all_rows: bool = typer.Option(False, "--all", help="Display all rows."),
) -> None:
    """Display dataset rows and columns in the terminal."""
    _display_impl(file, rows, tail, columns, all_rows)


@app.command(name="show")
def show_cmd(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV)."),
    rows: int = typer.Option(10, "--rows", "-r", help="Number of rows to display."),
    tail: bool = typer.Option(False, "--tail", help="Display last N rows instead of first N rows."),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        "-c",
        help="Comma-separated columns to display (e.g. age,salary,city).",
    ),
    all_rows: bool = typer.Option(False, "--all", help="Display all rows."),
) -> None:
    """Alias for `display` — show dataset rows in the terminal."""
    _display_impl(file, rows, tail, columns, all_rows)


if __name__ == "__main__":
    app()
