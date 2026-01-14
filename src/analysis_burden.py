"""
Burden analysis module.

Computes:
1. Outside fraction (burden outside Oct-Mar window)
2. Material activity (significant activity outside window)
3. Extended window evaluation (counterfactual analysis)
4. Population-weighted national summary
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_outside_fraction(
    df: pd.DataFrame,
    value_col: str = "rsv_0to4",
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute fraction of RSV burden outside the fixed Oct-Mar window.

    Args:
        df: Processed DataFrame with in_fixed_window column
        value_col: Column containing RSV admissions
        group_cols: Columns to group by (default: ['season', 'jurisdiction'])

    Returns:
        DataFrame with columns:
        - season, jurisdiction
        - total_burden: sum of admissions in season
        - inside_burden: admissions within Oct-Mar
        - outside_burden: admissions outside Oct-Mar
        - outside_fraction: outside_burden / total_burden
    """
    if group_cols is None:
        group_cols = ["season", "jurisdiction"]

    # Filter to rows with valid values
    df_valid = df[df[value_col].notna()].copy()

    results = []

    for group_keys, group_df in df_valid.groupby(group_cols):
        if len(group_cols) == 1:
            group_keys = (group_keys,)

        total = group_df[value_col].sum()
        inside = group_df.loc[group_df["in_fixed_window"], value_col].sum()
        outside = group_df.loc[~group_df["in_fixed_window"], value_col].sum()

        outside_frac = outside / total if total > 0 else np.nan

        row = dict(zip(group_cols, group_keys))
        row.update({
            "total_burden": total,
            "inside_burden": inside,
            "outside_burden": outside,
            "outside_fraction": outside_frac
        })
        results.append(row)

    df_result = pd.DataFrame(results)

    # Log summary
    logger.info("=" * 50)
    logger.info("OUTSIDE FRACTION ANALYSIS")
    logger.info("=" * 50)
    logger.info(f"Computed for {len(df_result)} state-season combinations")
    logger.info(f"Median outside fraction: {df_result['outside_fraction'].median():.3f}")
    logger.info(f"Mean outside fraction: {df_result['outside_fraction'].mean():.3f}")
    logger.info(f"Range: {df_result['outside_fraction'].min():.3f} - {df_result['outside_fraction'].max():.3f}")

    return df_result


def compute_material_activity(
    df: pd.DataFrame,
    value_col: str = "rsv_0to4",
    smoothing: int = 3,
    threshold: float = 0.20
) -> pd.DataFrame:
    """
    Compute material activity outside the fixed window.

    Material activity = weeks where smoothed admissions >= threshold * peak.
    This captures clinically meaningful activity, not just small tails.

    Args:
        df: Processed DataFrame
        value_col: Column containing RSV admissions
        smoothing: Rolling window size for smoothing
        threshold: Fraction of peak to define "material" activity

    Returns:
        DataFrame with columns:
        - season, jurisdiction
        - peak_week: week of maximum activity
        - peak_value: maximum smoothed value
        - material_weeks_total: total weeks with material activity
        - material_weeks_outside: material weeks outside Oct-Mar
        - material_burden_total: total admissions on material weeks
        - material_burden_outside: admissions on material weeks outside window
        - material_burden_outside_frac: material_burden_outside / total_burden
    """
    df_valid = df[df[value_col].notna()].copy()

    results = []

    for (season, jurisdiction), group_df in df_valid.groupby(["season", "jurisdiction"]):
        # Sort by week
        group_df = group_df.sort_values("week_end").copy()

        # Apply smoothing
        group_df["smoothed"] = (
            group_df[value_col]
            .rolling(window=smoothing, min_periods=1, center=True)
            .mean()
        )

        # Find peak
        peak_idx = group_df["smoothed"].idxmax()
        peak_week = group_df.loc[peak_idx, "week_end"]
        peak_value = group_df["smoothed"].max()

        # Define material threshold
        material_threshold = threshold * peak_value

        # Flag material weeks
        group_df["is_material"] = group_df["smoothed"] >= material_threshold

        # Count material weeks
        material_weeks_total = group_df["is_material"].sum()
        material_weeks_outside = (
            group_df["is_material"] & ~group_df["in_fixed_window"]
        ).sum()

        # Sum burden
        total_burden = group_df[value_col].sum()
        material_burden_total = group_df.loc[group_df["is_material"], value_col].sum()
        material_burden_outside = group_df.loc[
            group_df["is_material"] & ~group_df["in_fixed_window"],
            value_col
        ].sum()

        material_burden_outside_frac = (
            material_burden_outside / total_burden if total_burden > 0 else np.nan
        )

        results.append({
            "season": season,
            "jurisdiction": jurisdiction,
            "peak_week": peak_week,
            "peak_value": peak_value,
            "material_weeks_total": material_weeks_total,
            "material_weeks_outside": material_weeks_outside,
            "material_burden_total": material_burden_total,
            "material_burden_outside": material_burden_outside,
            "material_burden_outside_frac": material_burden_outside_frac
        })

    df_result = pd.DataFrame(results)

    # Log summary
    logger.info("=" * 50)
    logger.info(f"MATERIAL ACTIVITY ANALYSIS (threshold={threshold})")
    logger.info("=" * 50)
    logger.info(f"Median material weeks outside: {df_result['material_weeks_outside'].median():.1f}")
    logger.info(f"Median material burden outside fraction: {df_result['material_burden_outside_frac'].median():.3f}")

    # Highlight states with high material burden outside
    high_material = df_result[df_result["material_burden_outside_frac"] > 0.10]
    logger.info(f"State-seasons with >10% material burden outside: {len(high_material)}")

    return df_result


def evaluate_extended_windows(
    df: pd.DataFrame,
    value_col: str = "rsv_0to4"
) -> pd.DataFrame:
    """
    Evaluate coverage under different window definitions.

    Windows evaluated:
    - Baseline: Oct 1 - Mar 31
    - Early start: Sep 1 - Mar 31
    - Extended: Sep 1 - Apr 30

    Args:
        df: Processed DataFrame
        value_col: Column containing RSV admissions

    Returns:
        DataFrame with columns:
        - window_name, season, jurisdiction
        - coverage: fraction of burden within window
        - missed_fraction: 1 - coverage
    """
    # Define window configurations
    windows = {
        "baseline_oct_mar": {"start_month": 10, "start_day": 1, "end_month": 3, "end_day": 31},
        "early_sep_mar": {"start_month": 9, "start_day": 1, "end_month": 3, "end_day": 31},
        "extended_sep_apr": {"start_month": 9, "start_day": 1, "end_month": 4, "end_day": 30},
    }

    df_valid = df[df[value_col].notna()].copy()

    results = []

    for (season, jurisdiction), group_df in df_valid.groupby(["season", "jurisdiction"]):
        # Parse season years
        years = season.split("-")
        start_year = int(years[0])
        end_year = int(years[1])

        total_burden = group_df[value_col].sum()

        for window_name, window_config in windows.items():
            # Build window dates
            window_start = pd.Timestamp(
                year=start_year,
                month=window_config["start_month"],
                day=window_config["start_day"]
            )
            window_end = pd.Timestamp(
                year=end_year,
                month=window_config["end_month"],
                day=window_config["end_day"]
            )

            # Compute coverage
            in_window = (
                (group_df["week_end"] >= window_start) &
                (group_df["week_end"] <= window_end)
            )
            inside_burden = group_df.loc[in_window, value_col].sum()
            coverage = inside_burden / total_burden if total_burden > 0 else np.nan

            results.append({
                "window_name": window_name,
                "season": season,
                "jurisdiction": jurisdiction,
                "total_burden": total_burden,
                "inside_burden": inside_burden,
                "coverage": coverage,
                "missed_fraction": 1 - coverage if not np.isnan(coverage) else np.nan
            })

    df_result = pd.DataFrame(results)

    # Log summary
    logger.info("=" * 50)
    logger.info("EXTENDED WINDOW EVALUATION")
    logger.info("=" * 50)

    for window_name in windows.keys():
        subset = df_result[df_result["window_name"] == window_name]
        median_coverage = subset["coverage"].median()
        median_missed = subset["missed_fraction"].median()
        logger.info(f"  {window_name}: median coverage={median_coverage:.3f}, median missed={median_missed:.3f}")

    return df_result


def compute_national_summary(
    df_outside: pd.DataFrame,
    weight_col: str = "total_burden"
) -> pd.DataFrame:
    """
    Compute population-weighted national summary of outside fractions.

    Args:
        df_outside: Output from compute_outside_fraction
        weight_col: Column to use as weights (typically total_burden)

    Returns:
        DataFrame with columns:
        - season
        - national_outside_fraction_unweighted: simple mean
        - national_outside_fraction_weighted: burden-weighted mean
        - n_states: number of states
        - total_national_burden: sum of all state burdens
    """
    results = []

    for season, group_df in df_outside.groupby("season"):
        # Filter to valid rows
        valid = group_df[
            group_df["outside_fraction"].notna() &
            group_df[weight_col].notna() &
            (group_df[weight_col] > 0)
        ]

        if len(valid) == 0:
            continue

        # Unweighted mean
        unweighted = valid["outside_fraction"].mean()

        # Weighted mean
        weights = valid[weight_col]
        weighted = (valid["outside_fraction"] * weights).sum() / weights.sum()

        # Total national burden
        total_burden = valid["total_burden"].sum()
        total_outside = valid["outside_burden"].sum()

        results.append({
            "season": season,
            "national_outside_fraction_unweighted": unweighted,
            "national_outside_fraction_weighted": weighted,
            "national_outside_burden_direct": total_outside / total_burden if total_burden > 0 else np.nan,
            "n_states": len(valid),
            "total_national_burden": total_burden,
            "total_national_outside": total_outside
        })

    df_result = pd.DataFrame(results)

    # Log summary
    logger.info("=" * 50)
    logger.info("NATIONAL SUMMARY")
    logger.info("=" * 50)
    for _, row in df_result.iterrows():
        logger.info(
            f"  {row['season']}: weighted outside={row['national_outside_fraction_weighted']:.3f}, "
            f"n_states={row['n_states']}, total_burden={row['total_national_burden']:,.0f}"
        )

    return df_result


def compute_regional_summary(
    df: pd.DataFrame,
    df_outside: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute summary statistics by HHS region.

    Args:
        df: Processed DataFrame with hhs_region column
        df_outside: Output from compute_outside_fraction

    Returns:
        DataFrame with median (IQR) outside fraction by region and season
    """
    # Merge HHS region info
    region_map = df[["jurisdiction", "hhs_region"]].drop_duplicates()
    df_merged = df_outside.merge(region_map, on="jurisdiction", how="left")

    results = []

    for (season, region), group_df in df_merged.groupby(["season", "hhs_region"]):
        if pd.isna(region):
            continue

        outside_fracs = group_df["outside_fraction"].dropna()

        if len(outside_fracs) == 0:
            continue

        results.append({
            "season": season,
            "hhs_region": int(region),
            "n_states": len(outside_fracs),
            "median_outside_fraction": outside_fracs.median(),
            "q25_outside_fraction": outside_fracs.quantile(0.25),
            "q75_outside_fraction": outside_fracs.quantile(0.75),
            "min_outside_fraction": outside_fracs.min(),
            "max_outside_fraction": outside_fracs.max()
        })

    df_result = pd.DataFrame(results)

    # Log summary
    logger.info("=" * 50)
    logger.info("REGIONAL SUMMARY")
    logger.info("=" * 50)
    for region in sorted(df_result["hhs_region"].unique()):
        subset = df_result[df_result["hhs_region"] == region]
        median = subset["median_outside_fraction"].mean()
        logger.info(f"  Region {region}: avg median outside={median:.3f}")

    return df_result


def run_burden_analysis(df: pd.DataFrame, value_col: str = None) -> dict:
    """
    Run all burden analyses.

    Args:
        df: Processed DataFrame from build_seasons
        value_col: Column containing RSV admissions (default: from config)

    Returns:
        Dictionary with all analysis results:
        - outside_fraction: state-season outside fractions
        - material_activity: material activity analysis
        - extended_windows: counterfactual window evaluation
        - national_summary: population-weighted national summary
        - regional_summary: summary by HHS region
    """
    config = load_config()
    smoothing = config["analysis"]["smoothing_window"]
    threshold = config["analysis"]["material_activity_threshold"]

    # Use primary outcome from config if not specified
    if value_col is None:
        value_col = config.get("primary_outcome", "rsv_ped_total")

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BURDEN ANALYSIS")
    logger.info(f"Using outcome variable: {value_col}")
    logger.info("=" * 60 + "\n")

    # Analysis 1: Outside fraction
    outside_fraction = compute_outside_fraction(df, value_col=value_col)

    # Analysis 2: Material activity
    material_activity = compute_material_activity(
        df,
        value_col=value_col,
        smoothing=smoothing,
        threshold=threshold
    )

    # Analysis 3: Extended windows
    extended_windows = evaluate_extended_windows(df, value_col=value_col)

    # Analysis 4: National summary
    national_summary = compute_national_summary(outside_fraction)

    # Bonus: Regional summary
    regional_summary = compute_regional_summary(df, outside_fraction)

    return {
        "outside_fraction": outside_fraction,
        "material_activity": material_activity,
        "extended_windows": extended_windows,
        "national_summary": national_summary,
        "regional_summary": regional_summary
    }


def main():
    """Main entry point for burden analysis."""
    from src.build_seasons import build_seasons
    from src.pull_hrd import load_cached_or_fetch

    logger.info("Loading and processing data...")
    df_raw = load_cached_or_fetch()
    df = build_seasons(df_raw)

    logger.info("Running burden analysis...")
    results = run_burden_analysis(df)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    national = results["national_summary"]
    for _, row in national.iterrows():
        print(f"\n{row['season']}:")
        print(f"  Weighted national outside fraction: {row['national_outside_fraction_weighted']:.1%}")
        print(f"  Total national burden: {row['total_national_burden']:,.0f}")

    return results


if __name__ == "__main__":
    main()
