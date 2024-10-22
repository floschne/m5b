from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_column_distribution(
    df: pd.DataFrame,
    title_suffix: str,
    col: str = "language",
    sort_by_col: bool = False,
    figsize: Tuple[int, int] = (20, 6),
    logy: bool = False,
):
    assert col in df.columns, f"{col} column not found in {df.columns}"

    plt.figure(figsize=figsize)  # Set figure size to wider landscape orientation
    val_counts = df[col].value_counts()
    average_count = val_counts.mean()
    min_count = val_counts.min()
    max_count = val_counts.max()

    if sort_by_col:
        val_counts = val_counts.sort_index()

    val_counts.plot(kind="bar", logy=logy)
    plt.axhline(
        average_count,
        color="red",
        linestyle="--",
        label=f"Average Count: {average_count:.2f}",
    )
    plt.axhline(
        min_count, color="blue", linestyle="--", label=f"Min Count: {min_count:.2f}"
    )
    plt.axhline(
        max_count, color="green", linestyle="--", label=f"Max Count: {max_count:.2f}"
    )

    plt.title(f"{col.capitalize()} Distribution - {title_suffix}")
    plt.xlabel(col.capitalize())
    plt.ylabel("Count")
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xticks(
        range(len(val_counts.index)), list(map(str, val_counts.index))
    )  # Set custom x-axis labels
    plt.legend()
