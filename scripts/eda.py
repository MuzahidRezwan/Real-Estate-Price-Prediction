"""Exploratory Data Analysis (EDA) chart generator for the cleaned dataset.

Generates publication-quality Matplotlib/Seaborn figures in static/charts/:
  - price_distribution.png
  - price_vs_area.png
  - top_locations_avg_price.png
  - correlation_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT_DIR / "data" / "cleaned_dataset.csv"
CHARTS_DIR = ROOT_DIR / "static" / "charts"


def setup_style() -> None:
    """Apply consistent, crisp typography and color palette."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def plot_price_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot histogram and KDE of property prices in BDT Crores."""
    prices_crore = df["Estimated_Total_Price_Taka"] / 1e7
    mean_cr = float(prices_crore.mean())
    median_cr = float(prices_crore.median())

    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(
        prices_crore,
        bins=32,
        kde=True,
        color="#0F766E",
        edgecolor="#FFFFFF",
        alpha=0.78,
        ax=ax,
    )
    ax.axvline(mean_cr, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {mean_cr:.2f} Cr BDT")
    ax.axvline(median_cr, color="#D97706", linestyle="-.", linewidth=1.8, label=f"Median: {median_cr:.2f} Cr BDT")
    ax.set_title("Distribution of Property Prices (Cleaned Dataset)")
    ax.set_xlabel("Estimated Total Price (Crore BDT)")
    ax.set_ylabel("Number of Listings")
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1")
    fig.tight_layout()

    out_path = out_dir / "price_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_price_vs_area(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot Floor Area (sqft) vs Total Price (Crore BDT) with regression fit."""
    plot_df = df.copy()
    plot_df["Price_Crore"] = plot_df["Estimated_Total_Price_Taka"] / 1e7
    top_5_locs = (
        plot_df[plot_df["Location"] != "Other"]["Location"]
        .value_counts()
        .head(5)
        .index.tolist()
    )
    plot_df["Neighborhood_Group"] = np.where(
        plot_df["Location"].isin(top_5_locs), plot_df["Location"], "Other / Grouped"
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.scatterplot(
        data=plot_df,
        x="Floor_area_sft",
        y="Price_Crore",
        hue="Neighborhood_Group",
        palette="Set2",
        alpha=0.72,
        s=42,
        edgecolor="#1E293B",
        linewidth=0.25,
        ax=ax,
    )
    # Overall linear trend line
    z = np.polyfit(plot_df["Floor_area_sft"], plot_df["Price_Crore"], 1)
    x_line = np.linspace(plot_df["Floor_area_sft"].min(), plot_df["Floor_area_sft"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#0F172A", linewidth=2.0, linestyle="--", label="Linear Trend")

    ax.set_title("Property Price vs. Floor Area by Neighborhood")
    ax.set_xlabel("Floor Area (sq. ft.)")
    ax.set_ylabel("Estimated Total Price (Crore BDT)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Location", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)
    fig.tight_layout()

    out_path = out_dir / "price_vs_area.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_top_locations_avg_price(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot top 15 neighborhoods ranked by average total price in Crore BDT."""
    named_df = df[df["Location"] != "Other"].copy()
    loc_stats = (
        named_df.groupby("Location")["Estimated_Total_Price_Taka"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(15)
    )
    loc_stats["mean_crore"] = loc_stats["mean"] / 1e7
    loc_stats = loc_stats.sort_values("mean_crore", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    colors = sns.color_palette("crest", n_colors=len(loc_stats))
    bars = ax.barh(loc_stats.index, loc_stats["mean_crore"], color=colors, edgecolor="#0F172A", linewidth=0.35)

    max_val = float(loc_stats["mean_crore"].max())
    for bar, (_, row) in zip(bars, loc_stats.iterrows()):
        width = bar.get_width()
        ax.text(
            width + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f} Cr  (n={int(row['count'])})",
            va="center",
            fontsize=8.5,
            color="#0F172A",
        )

    ax.set_xlim(0, max_val * 1.24)
    ax.set_title("Top 15 Locations by Average Property Price")
    ax.set_xlabel("Average Price (Crore BDT)")
    ax.set_ylabel("")
    fig.tight_layout()

    out_path = out_dir / "top_locations_avg_price.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot Pearson correlation matrix across numeric structural features & price."""
    num_cols = [
        "Floor_area_sft",
        "Bedrooms",
        "Bathrooms",
        "Parking_Count",
        "Estimated_Total_Price_Taka",
    ]
    pretty_names = {
        "Floor_area_sft": "Floor Area (sqft)",
        "Bedrooms": "Bedrooms",
        "Bathrooms": "Bathrooms",
        "Parking_Count": "Parking Count",
        "Estimated_Total_Price_Taka": "Total Price (BDT)",
    }
    corr = df[num_cols].rename(columns=pretty_names).corr()

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Tealrose" if "Tealrose" in plt.colormaps() else "YlGnBu",
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.8,
        linecolor="#FFFFFF",
        square=True,
        cbar_kws={"shrink": 0.82},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Structural Features vs. Price)")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out_path = out_dir / "correlation_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_charts(clean_path: Path = CLEAN_PATH, out_dir: Path = CHARTS_DIR) -> list[Path]:
    """Load cleaned CSV and write all 4 EDA charts."""
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean_path)

    generated = [
        plot_price_distribution(df, out_dir),
        plot_price_vs_area(df, out_dir),
        plot_top_locations_avg_price(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
    ]
    for path in generated:
        print(f"Generated EDA chart: {path}")
    return generated


if __name__ == "__main__":
    generate_all_charts()
