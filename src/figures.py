"""
Figure generation for RSV timing analysis.

Creates:
1. Choropleth maps of outside fraction by state/season
2. Time series for Southeast and comparator states
3. Coverage comparison charts
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    """Save figure in both PNG and PDF formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for fmt in ["png", "pdf"]:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {filepath}")


def make_choropleth_figure(
    df_outside: pd.DataFrame,
    output_name: str = "fig1_choropleth"
) -> plt.Figure:
    """
    Create choropleth map of outside fraction by state for each season.

    Args:
        df_outside: DataFrame with season, jurisdiction, outside_fraction
        output_name: Base name for output files

    Returns:
        matplotlib Figure
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not available, creating bar chart instead of choropleth")
        return make_bar_chart_figure(df_outside, output_name)

    # Get US states shapefile
    try:
        # Try to load natural earth data
        us_states = gpd.read_file(
            "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        )
        us_states = us_states.rename(columns={"name": "jurisdiction"})

    except Exception as e:
        logger.warning(f"Could not load shapefile: {e}. Creating bar chart instead.")
        return make_bar_chart_figure(df_outside, output_name)

    seasons = sorted(df_outside["season"].unique())
    n_seasons = len(seasons)

    fig, axes = plt.subplots(1, n_seasons, figsize=(5 * n_seasons, 4))
    if n_seasons == 1:
        axes = [axes]

    # Color scale
    vmin = 0
    vmax = min(df_outside["outside_fraction"].quantile(0.95), 0.25)
    cmap = "Reds"

    for ax, season in zip(axes, seasons):
        # Filter data for this season
        season_data = df_outside[df_outside["season"] == season]

        # Merge with geometry (NSSP uses full state names which match the shapefile)
        merged = us_states.merge(season_data, on="jurisdiction", how="left")

        # Filter to continental US (exclude AK, HI by name)
        merged = merged[~merged["jurisdiction"].isin(["Alaska", "Hawaii", "Puerto Rico"])]

        # Plot
        merged.plot(
            column="outside_fraction",
            ax=ax,
            legend=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="gray",
            linewidth=0.5,
            missing_kwds={"color": "lightgray", "edgecolor": "gray", "linewidth": 0.5}
        )

        ax.set_title(f"{season}", fontsize=14, fontweight="bold")
        ax.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.01)
    cbar.set_label("Fraction of RSV Activity Outside Oct-Mar Window", fontsize=11)

    # Add tick marks at sensible intervals based on actual scale
    tick_vals = [0, 0.05, 0.10, 0.15]
    tick_vals = [t for t in tick_vals if t <= vmax]  # Only include ticks within scale
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in tick_vals])
    cbar.ax.tick_params(size=4)

    # Remove colorbar border/outline
    cbar.outline.set_visible(False)

    fig.suptitle(
        "RSV ED Visit Activity Outside Fixed Prophylaxis Window",
        fontsize=14,
        fontweight="bold",
        y=0.92
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    save_figure(fig, output_name)
    return fig


def make_regional_choropleth_figure(
    df_outside: pd.DataFrame,
    df_processed: pd.DataFrame,
    output_name: str = "fig1b_regional_choropleth"
) -> plt.Figure:
    """
    Create choropleth map showing median outside fraction by HHS region.
    State boundaries are dissolved into region boundaries.

    Args:
        df_outside: DataFrame with season, jurisdiction, outside_fraction
        df_processed: Processed DataFrame with hhs_region column
        output_name: Base name for output files

    Returns:
        matplotlib Figure
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not available, skipping regional choropleth")
        return None

    # Get US states shapefile
    try:
        us_states = gpd.read_file(
            "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        )
        us_states = us_states.rename(columns={"name": "jurisdiction"})
    except Exception as e:
        logger.warning(f"Could not load shapefile: {e}. Skipping regional choropleth.")
        return None

    # Filter to continental US first
    us_states = us_states[~us_states["jurisdiction"].isin(["Alaska", "Hawaii", "Puerto Rico"])]

    # Get FULL HHS region mapping from config (includes all states, even excluded ones)
    config = load_config()
    hhs_regions_config = config.get("hhs_regions", {})

    # Build complete state -> region mapping
    full_region_map = {}
    for region, states in hhs_regions_config.items():
        for state in states:
            full_region_map[state] = int(region)

    # Create DataFrame for full mapping
    full_region_df = pd.DataFrame([
        {"jurisdiction": state, "hhs_region": region}
        for state, region in full_region_map.items()
    ])

    # Merge HHS region info with state geometries (using full mapping)
    us_states = us_states.merge(full_region_df, on="jurisdiction", how="left")

    # Dissolve state geometries into HHS regions (merge boundaries)
    us_regions = us_states.dissolve(by="hhs_region", as_index=False)

    # Get region mapping from processed data for computing medians
    region_map = df_processed[["jurisdiction", "hhs_region"]].drop_duplicates()

    # Compute median outside fraction by region and season
    df_with_region = df_outside.merge(region_map, on="jurisdiction", how="left")
    regional_medians = df_with_region.groupby(["season", "hhs_region"]).agg(
        median_outside_fraction=("outside_fraction", "median")
    ).reset_index()

    seasons = sorted(df_outside["season"].unique())
    n_seasons = len(seasons)

    fig, axes = plt.subplots(1, n_seasons, figsize=(5 * n_seasons, 4))
    if n_seasons == 1:
        axes = [axes]

    # Color scale
    vmin = 0
    vmax = min(regional_medians["median_outside_fraction"].max() * 1.1, 0.20)
    cmap = "Reds"

    for ax, season in zip(axes, seasons):
        # Get regional medians for this season
        season_regional = regional_medians[regional_medians["season"] == season]

        # Merge regional values with dissolved geometries
        regions_with_data = us_regions.merge(
            season_regional[["hhs_region", "median_outside_fraction"]],
            on="hhs_region",
            how="left"
        )

        # Plot dissolved regions (no internal state boundaries)
        regions_with_data.plot(
            column="median_outside_fraction",
            ax=ax,
            legend=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidth=1.0,
            missing_kwds={"color": "lightgray", "edgecolor": "black", "linewidth": 1.0}
        )

        # Add region labels with manual offsets for specific regions
        label_offsets = {
            1: (-0.25, 0),  # Region 1 (New England) - move slightly left
            2: (0.5, 0.5),  # Region 2 (NY/NJ) - move up and slightly left from before
        }

        for idx, row in regions_with_data.iterrows():
            if pd.notna(row["hhs_region"]):
                centroid = row.geometry.centroid
                region_num = int(row["hhs_region"])

                # Apply manual offset if specified
                x_offset, y_offset = label_offsets.get(region_num, (0, 0))

                ax.annotate(
                    f"{region_num}",
                    xy=(centroid.x + x_offset, centroid.y + y_offset),
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='white',
                    path_effects=[
                        patheffects.withStroke(linewidth=2, foreground='black')
                    ]
                )

        ax.set_title(f"{season}", fontsize=14, fontweight="bold")
        ax.axis("off")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.01)
    cbar.set_label("Median Fraction Outside Oct-Mar Window (by HHS Region)", fontsize=11)

    # Add tick marks
    tick_vals = [0, 0.05, 0.10, 0.15]
    tick_vals = [t for t in tick_vals if t <= vmax]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in tick_vals])
    cbar.ax.tick_params(size=4)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "RSV Activity Outside Fixed Window by HHS Region",
        fontsize=14,
        fontweight="bold",
        y=0.92
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.90])

    save_figure(fig, output_name)
    return fig


def make_bar_chart_figure(
    df_outside: pd.DataFrame,
    output_name: str = "fig1_barchart"
) -> plt.Figure:
    """
    Fallback bar chart if choropleth cannot be created.

    Args:
        df_outside: DataFrame with season, jurisdiction, outside_fraction
        output_name: Base name for output files

    Returns:
        matplotlib Figure
    """
    config = load_config()
    geography = config["geography"]
    se_states = geography["southeast_states"]

    seasons = sorted(df_outside["season"].unique())
    n_seasons = len(seasons)

    fig, axes = plt.subplots(1, n_seasons, figsize=(6 * n_seasons, 10))
    if n_seasons == 1:
        axes = [axes]

    for ax, season in zip(axes, seasons):
        season_data = df_outside[df_outside["season"] == season].copy()
        season_data = season_data.sort_values("outside_fraction", ascending=True)

        # Use state abbreviations for display if available
        if "state_abbrev" in season_data.columns:
            display_col = "state_abbrev"
        else:
            display_col = "jurisdiction"

        # Color Southeast states differently
        colors = ["#d73027" if j in se_states else "#4575b4"
                  for j in season_data["jurisdiction"]]

        ax.barh(
            season_data[display_col],
            season_data["outside_fraction"],
            color=colors,
            edgecolor="white"
        )

        ax.set_xlabel("Fraction Outside Oct-Mar Window")
        ax.set_title(f"{season}", fontsize=12, fontweight="bold")
        ax.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="10%")
        ax.set_xlim(0, None)

    # Add legend
    se_patch = mpatches.Patch(color="#d73027", label="Southeast")
    other_patch = mpatches.Patch(color="#4575b4", label="Other")
    axes[-1].legend(handles=[se_patch, other_patch], loc="lower right")

    fig.suptitle(
        "Fraction of RSV ED Activity Outside Oct-Mar Window",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    save_figure(fig, output_name)
    return fig


def make_timeseries_figure(
    df: pd.DataFrame,
    onset_results: Optional[dict] = None,
    offset_results: Optional[dict] = None,
    output_name: str = "fig2_timeseries",
    value_col: str = None
) -> list:
    """
    Create time series plots - one figure per season with all states.

    Args:
        df: Processed DataFrame with weekly data
        onset_results: Optional dict of (season, jurisdiction) -> onset_week
        offset_results: Optional dict of (season, jurisdiction) -> offset_week
        output_name: Base name for output files
        value_col: Column to plot (default: from config)

    Returns:
        List of matplotlib Figures (one per season)
    """
    config = load_config()

    # Use primary outcome from config if not specified
    if value_col is None:
        value_col = config.get("primary_outcome", "rsv_pct")

    fixed_window = config["fixed_window"]

    seasons = sorted(df["season"].dropna().unique())

    # Get all states (excluding those already filtered), sorted alphabetically
    all_states = sorted(df["jurisdiction"].unique())
    n_states = len(all_states)

    # Calculate grid dimensions (aim for roughly square layout)
    n_cols = 6
    n_rows = int(np.ceil(n_states / n_cols))

    figures = []

    for season in seasons:
        # Parse season years
        years = season.split("-")
        start_year = int(years[0])
        end_year = int(years[1])

        # Season boundaries (July 1 to June 30)
        season_start = pd.Timestamp(year=start_year, month=7, day=1)
        season_end = pd.Timestamp(year=end_year, month=6, day=30)

        # Fixed prophylaxis window (Oct 1 - Mar 31)
        window_start = pd.Timestamp(
            year=start_year,
            month=fixed_window["start_month"],
            day=fixed_window["start_day"]
        )
        window_end = pd.Timestamp(
            year=end_year,
            month=fixed_window["end_month"],
            day=fixed_window["end_day"]
        )

        # Create figure for this season
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3 * n_cols, 2 * n_rows),
            sharex=True,
            sharey=True
        )
        axes = axes.flatten()

        # Get season data
        season_data = df[df["season"] == season].copy()

        # Get global y-axis max for this season
        y_max = season_data[value_col].max() * 1.1

        # Get actual data range for this season (for x-axis limits)
        data_start = season_data["week_end"].min()
        data_end = season_data["week_end"].max()

        for i, state in enumerate(all_states):
            ax = axes[i]

            # Get data for this state
            state_data = season_data[season_data["jurisdiction"] == state].sort_values("week_end")

            # Get state abbreviation
            if "state_abbrev" in state_data.columns and len(state_data) > 0:
                abbrev = state_data["state_abbrev"].iloc[0]
            else:
                abbrev = state[:2].upper()

            if len(state_data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_title(abbrev, fontsize=9, fontweight="bold")
                xlim_start = data_start - pd.Timedelta(weeks=2)
                xlim_end = data_end + pd.Timedelta(weeks=2)
                ax.set_xlim(xlim_start, xlim_end)
                continue

            # Shade fixed window (Oct 1 - Mar 31) - green box
            ax.axvspan(window_start, window_end, alpha=0.2, color="green", zorder=1)

            # Plot RSV percent
            ax.plot(
                state_data["week_end"],
                state_data[value_col],
                color="#2166ac",
                linewidth=1.2,
                zorder=3
            )

            # Mark onset if available (red dashed line)
            if onset_results is not None:
                onset = onset_results.get((season, state))
                if onset is not None:
                    ax.axvline(onset, color="red", linestyle="--", linewidth=1, alpha=0.7, zorder=4)

            # Mark offset if available (red dashed line)
            if offset_results is not None:
                offset = offset_results.get((season, state))
                if offset is not None:
                    ax.axvline(offset, color="red", linestyle="--", linewidth=1, alpha=0.7, zorder=4)

            ax.set_title(abbrev, fontsize=9, fontweight="bold")
            ax.set_ylim(0, y_max)
            # Set x-axis to actual data range with 2-week buffer
            xlim_start = data_start - pd.Timedelta(weeks=2)
            xlim_end = data_end + pd.Timedelta(weeks=2)
            ax.set_xlim(xlim_start, xlim_end)

            # Only show x-tick labels on bottom row
            if i >= (n_rows - 1) * n_cols:
                ax.tick_params(axis="x", rotation=45, labelsize=7)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.tick_params(axis="y", labelsize=7)

        # Hide unused subplots
        for i in range(n_states, len(axes)):
            axes[i].set_visible(False)

        # Common labels
        fig.text(0.5, 0.02, "Week", ha="center", fontsize=11)
        fig.text(0.02, 0.5, "% ED Visits for RSV", va="center", rotation="vertical", fontsize=11)

        # Legend
        green_patch = mpatches.Patch(color="green", alpha=0.2, label="Oct-Mar Beyfortus window")
        handles = [green_patch]

        if onset_results or offset_results:
            red_line = plt.Line2D([0], [0], color="red", linestyle="--", label="Detected onset/offset")
            handles.append(red_line)

        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)

        fig.suptitle(
            f"Weekly RSV ED Visit Percentage by State - {season}",
            fontsize=14,
            fontweight="bold",
            y=0.99
        )

        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        save_figure(fig, f"{output_name}_{season.replace('-', '_')}")
        figures.append(fig)

    return figures


def make_coverage_comparison_figure(
    df_extended: pd.DataFrame,
    df_trigger: pd.DataFrame,
    output_name: str = "fig3_coverage_comparison"
) -> plt.Figure:
    """
    Create coverage comparison chart: fixed vs extended vs trigger windows.

    Args:
        df_extended: Output from evaluate_extended_windows
        df_trigger: Output from evaluate_trigger_coverage
        output_name: Base name for output files

    Returns:
        matplotlib Figure
    """
    seasons = sorted(df_extended["season"].unique())
    n_seasons = len(seasons)

    fig, axes = plt.subplots(1, n_seasons, figsize=(5 * n_seasons, 5))
    if n_seasons == 1:
        axes = [axes]

    # Label mapping for two-row format
    label_map = {
        "baseline_oct_mar": "Baseline\n(Oct-Mar)",
        "early_sep_mar": "Early\n(Sep-Mar)",
        "extended_sep_apr": "Extended\n(Sep-Apr)"
    }

    for ax, season in zip(axes, seasons):
        # Extended window results
        ext_season = df_extended[df_extended["season"] == season]

        # Compute median coverage by window type
        coverage_summary = []

        for window_name in ext_season["window_name"].unique():
            subset = ext_season[ext_season["window_name"] == window_name]
            coverage_summary.append({
                "window": label_map.get(window_name, window_name),
                "median_coverage": subset["coverage"].median(),
                "q25": subset["coverage"].quantile(0.25),
                "q75": subset["coverage"].quantile(0.75)
            })

        # Trigger results
        trig_season = df_trigger[
            (df_trigger["season"] == season) &
            (df_trigger["trigger_coverage"].notna())
        ]

        if len(trig_season) > 0:
            coverage_summary.append({
                "window": "Dynamic\n(onset-offset)",
                "median_coverage": trig_season["trigger_coverage"].median(),
                "q25": trig_season["trigger_coverage"].quantile(0.25),
                "q75": trig_season["trigger_coverage"].quantile(0.75)
            })

        coverage_df = pd.DataFrame(coverage_summary)

        # Plot
        x = range(len(coverage_df))
        colors = ["#4575b4", "#91bfdb", "#fee090", "#d73027"][:len(coverage_df)]

        bars = ax.bar(
            x,
            coverage_df["median_coverage"],
            color=colors,
            edgecolor="black",
            linewidth=0.5
        )

        # Error bars (IQR)
        yerr_lower = coverage_df["median_coverage"] - coverage_df["q25"]
        yerr_upper = coverage_df["q75"] - coverage_df["median_coverage"]
        ax.errorbar(
            x,
            coverage_df["median_coverage"],
            yerr=[yerr_lower, yerr_upper],
            fmt="none",
            color="black",
            capsize=3
        )

        ax.set_xticks(x)
        ax.set_xticklabels(coverage_df["window"], fontsize=9)
        ax.set_ylabel("Median Coverage")
        ax.set_ylim(0.7, 1.0)
        ax.set_title(f"{season}", fontsize=12, fontweight="bold")

        # Add horizontal line at baseline
        baseline = coverage_df.loc[coverage_df["window"].str.contains("Baseline", case=False), "median_coverage"]
        if len(baseline) > 0:
            ax.axhline(baseline.iloc[0], color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Coverage Comparison: Fixed vs Extended vs Trigger Windows",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    save_figure(fig, output_name)
    return fig


def make_regional_heatmap(
    df_regional: pd.DataFrame,
    output_name: str = "fig_supp_regional_heatmap"
) -> plt.Figure:
    """
    Create heatmap of outside fraction by HHS region and season.

    Args:
        df_regional: Output from compute_regional_summary
        output_name: Base name for output files

    Returns:
        matplotlib Figure
    """
    # Pivot to matrix form
    pivot = df_regional.pivot(
        index="hhs_region",
        columns="season",
        values="median_outside_fraction"
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=0,
        vmax=0.20,
        ax=ax,
        cbar_kws={"label": "Median Outside Fraction"}
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("HHS Region")
    ax.set_title("Median Fraction Outside Oct-Mar by HHS Region", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_name)
    return fig


def generate_all_figures(
    df: pd.DataFrame,
    burden_results: dict,
    trigger_results: dict
) -> dict[str, plt.Figure]:
    """
    Generate all figures for the analysis.

    Args:
        df: Processed DataFrame
        burden_results: Output from run_burden_analysis
        trigger_results: Output from run_trigger_analysis

    Returns:
        Dictionary mapping figure names to Figure objects
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60 + "\n")

    figures = {}

    # Merge state abbreviations into outside_fraction for figures
    if "state_abbrev" in df.columns:
        abbrev_map = df[["jurisdiction", "state_abbrev"]].drop_duplicates()
        burden_results["outside_fraction"] = burden_results["outside_fraction"].merge(
            abbrev_map, on="jurisdiction", how="left"
        )

    # Figure 1: Choropleth (by state)
    logger.info("Generating Figure 1: Choropleth...")
    figures["fig1"] = make_choropleth_figure(burden_results["outside_fraction"])

    # Figure 1b: Regional choropleth (by HHS region)
    logger.info("Generating Figure 1b: Regional choropleth...")
    figures["fig1b"] = make_regional_choropleth_figure(
        burden_results["outside_fraction"],
        df
    )

    # Figure 2: Time series (one per season)
    logger.info("Generating Figure 2: Time series...")
    timeseries_figs = make_timeseries_figure(
        df,
        onset_results=trigger_results["onset_results"],
        offset_results=trigger_results.get("offset_results")
    )
    for i, fig in enumerate(timeseries_figs):
        figures[f"fig2_{i}"] = fig

    # Figure 3: Coverage comparison
    logger.info("Generating Figure 3: Coverage comparison...")
    figures["fig3"] = make_coverage_comparison_figure(
        burden_results["extended_windows"],
        trigger_results["trigger_coverage"]
    )

    # Supplemental: Regional heatmap
    logger.info("Generating supplemental: Regional heatmap...")
    figures["fig_supp_regional"] = make_regional_heatmap(
        burden_results["regional_summary"]
    )

    logger.info(f"\nGenerated {len(figures)} figures")

    return figures


def main():
    """Main entry point for figure generation."""
    from src.analysis_burden import run_burden_analysis
    from src.analysis_trigger import run_trigger_analysis
    from src.build_seasons import build_seasons
    from src.pull_nssp import load_cached_or_fetch

    logger.info("Loading and processing data...")
    df_raw = load_cached_or_fetch()
    df = build_seasons(df_raw)

    logger.info("Running analyses...")
    burden_results = run_burden_analysis(df)
    trigger_results = run_trigger_analysis(df)

    logger.info("Generating figures...")
    figures = generate_all_figures(df, burden_results, trigger_results)

    return figures


if __name__ == "__main__":
    main()
