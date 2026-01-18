"""
Main pipeline orchestration script.

Ties together all analysis components:
1. Data extraction from CDC NSSP
2. Season building and data processing
3. Burden analysis (outside fraction, material activity, extended windows)
4. Trigger analysis (onset detection, coverage evaluation)
5. Figure generation
6. Table output
7. Parallel NHSN (HRD) analysis for pediatric RSV admissions, ages 0-4

Usage:
    python -m src.run_pipeline
"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame as CSV table."""
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_TABLES / f"{name}.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved table: {filepath}")
    return filepath


def run_r_figures() -> None:
    """Run R-based figure generation script."""
    script_path = PROJECT_ROOT / "src" / "figures.R"
    if not script_path.exists():
        logger.error(f"R figure script not found at {script_path}")
        return

    logger.info("Generating figures with R/ggplot2...")
    try:
        subprocess.run(
            ["Rscript", str(script_path)],
            check=True,
            cwd=PROJECT_ROOT
        )
    except FileNotFoundError:
        logger.error("Rscript not found. Install R and ensure Rscript is on PATH.")
    except subprocess.CalledProcessError as exc:
        logger.error(f"R figure script failed with exit code {exc.returncode}")

def attach_metric_label(df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Attach a metric label column for clarity in exported tables."""
    labeled = df.copy()
    labeled["metric_label"] = metric_label
    return labeled


def create_table1(
    df_outside: pd.DataFrame,
    df_national: pd.DataFrame,
    df_regional: pd.DataFrame,
    total_label: str = "Total Metric (sum over weeks)"
) -> pd.DataFrame:
    """
    Create Table 1: Summary of burden outside Oct-Mar window.

    Includes:
    - National summary (weighted and unweighted)
    - Regional breakdown by HHS region
    """
    rows = []

    # National summary
    for _, row in df_national.iterrows():
        rows.append({
            "Group": "National (weighted)",
            "Season": row["season"],
            "N": row["n_states"],
            "Median Outside Fraction": row["national_outside_fraction_weighted"],
            "Q25": None,
            "Q75": None,
            total_label: row.get("total_national_burden", None)
        })

    # Regional summary
    for _, row in df_regional.iterrows():
        rows.append({
            "Group": f"HHS Region {int(row['hhs_region'])}",
            "Season": row["season"],
            "N": row["n_states"],
            "Median Outside Fraction": row["median_outside_fraction"],
            "Q25": row["q25_outside_fraction"],
            "Q75": row["q75_outside_fraction"],
            total_label: None
        })

    df_table1 = pd.DataFrame(rows)

    # Format for readability
    df_table1 = df_table1.sort_values(["Season", "Group"])

    return df_table1


def create_trigger_comparison_table(
    df_trigger: pd.DataFrame
) -> pd.DataFrame:
    """
    Create table comparing fixed vs trigger window coverage.
    """
    # Summary by season
    summary = []

    for season in df_trigger["season"].unique():
        season_data = df_trigger[df_trigger["season"] == season]
        valid = season_data[season_data["trigger_coverage"].notna()]

        if len(valid) == 0:
            continue

        summary.append({
            "Season": season,
            "N States": len(season_data),
            "N with Onset Detected": len(valid),
            "Median Fixed Coverage": valid["fixed_coverage"].median(),
            "Median Trigger Coverage": valid["trigger_coverage"].median(),
            "Median Improvement": valid["improvement"].median(),
            "N Trigger Better (>=5%)": valid["trigger_better_by_5pct"].sum(),
            "Pct Trigger Better (>=5%)": 100 * valid["trigger_better_by_5pct"].mean()
        })

    return pd.DataFrame(summary)


def create_cross_source_comparison(
    nssp_burden: dict,
    nhsn_burden: dict,
    season: str = "2024-2025",
    nssp_metric_label: str = "RSV ED visit percentage of total ED visits, all ages",
    nhsn_metric_label: str = "Pediatric RSV hospital admissions, ages 0-4"
) -> pd.DataFrame:
    """
    Compare NSSP and NHSN outside-fraction summaries for a single season.
    """
    rows = []

    def build_row(source_label: str, metric_label: str, burden: dict) -> None:
        outside = burden["outside_fraction"]
        national = burden["national_summary"]

        season_outside = outside[outside["season"] == season]
        season_national = national[national["season"] == season]

        if len(season_outside) == 0:
            rows.append({
                "Data Source": source_label,
                "Metric": metric_label,
                "Season": season,
                "N States": 0,
                "Median Outside Fraction": None,
                "Unweighted Outside Fraction": None,
                "Weighted Outside Fraction": None,
                "Total Metric (sum over weeks)": None
            })
            return

        median_outside = season_outside["outside_fraction"].median()
        unweighted = None
        weighted = None
        total_metric = None

        if len(season_national) > 0:
            row = season_national.iloc[0]
            unweighted = row.get("national_outside_fraction_unweighted", None)
            weighted = row.get("national_outside_fraction_weighted", None)
            total_metric = row.get("total_national_burden", None)

        rows.append({
            "Data Source": source_label,
            "Metric": metric_label,
            "Season": season,
            "N States": len(season_outside),
            "Median Outside Fraction": median_outside,
            "Unweighted Outside Fraction": unweighted,
            "Weighted Outside Fraction": weighted,
            "Total Metric (sum over weeks)": total_metric
        })

    build_row("NSSP", nssp_metric_label, nssp_burden)
    build_row("NHSN", nhsn_metric_label, nhsn_burden)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Data Source", "Season"])
    return df


def create_monthly_metric_comparison(
    df_nssp: pd.DataFrame,
    df_nhsn: pd.DataFrame,
    nssp_value_col: str,
    nhsn_value_col: str,
    season: str = "2024-2025",
    nssp_metric_label: str = "RSV ED visit percentage of total ED visits, all ages",
    nhsn_metric_label: str = "Pediatric RSV hospital admissions, ages 0-4"
) -> pd.DataFrame:
    """
    Compare monthly distributions of the NSSP and NHSN metrics for a season.
    """
    rows = []

    def add_rows(df: pd.DataFrame, value_col: str, source_label: str, metric_label: str) -> None:
        season_df = df[df["season"] == season].copy()
        if len(season_df) == 0:
            return

        season_df["month"] = season_df["week_end"].dt.to_period("M").astype(str)
        monthly = season_df.groupby("month")[value_col].sum()
        total = monthly.sum()

        for month, total_value in monthly.items():
            rows.append({
                "Data Source": source_label,
                "Metric": metric_label,
                "Season": season,
                "Month": month,
                "Month Metric Total": total_value,
                "Month Metric Fraction": total_value / total if total > 0 else None
            })

    add_rows(df_nssp, nssp_value_col, "NSSP", nssp_metric_label)
    add_rows(df_nhsn, nhsn_value_col, "NHSN", nhsn_metric_label)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Data Source", "Month"])
    return df


def log_summary_stats(
    df_outside: pd.DataFrame,
    df_trigger: pd.DataFrame,
    label: str
) -> None:
    """Log key summary statistics."""
    logger.info("\n" + "=" * 70)
    logger.info(f"{label} SUMMARY")
    logger.info("=" * 70)

    # Outside fraction summary
    logger.info("\nRSV ACTIVITY OUTSIDE FIXED WINDOW:")
    for season in sorted(df_outside["season"].unique()):
        season_data = df_outside[df_outside["season"] == season]
        median = season_data["outside_fraction"].median()
        q25 = season_data["outside_fraction"].quantile(0.25)
        q75 = season_data["outside_fraction"].quantile(0.75)
        logger.info(f"  {season}: median={median:.1%} (IQR: {q25:.1%}-{q75:.1%})")

    # Trigger summary
    logger.info("\nTRIGGER WINDOW IMPROVEMENT:")
    valid = df_trigger[df_trigger["trigger_coverage"].notna()]
    for season in sorted(valid["season"].unique()):
        season_data = valid[valid["season"] == season]
        median_improvement = season_data["improvement"].median()
        pct_better = 100 * season_data["trigger_better_by_5pct"].mean()
        logger.info(f"  {season}: median improvement={median_improvement:.1%}, {pct_better:.0f}% states better by >=5%")

    logger.info("\n" + "=" * 70)


def run_pipeline(force_refresh: bool = False) -> dict:
    """
    Run the complete analysis pipeline.

    Args:
        force_refresh: If True, re-fetch data from Socrata even if cache exists

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 70)
    logger.info("RSV TIMING / BEYFORTUS ANALYSIS PIPELINE (NSSP ED Data)")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 70 + "\n")

    # Import modules
    from src.pull_nssp import load_cached_or_fetch as load_nssp, log_row_counts as log_nssp_rows
    from src.pull_nhsn import load_cached_or_fetch as load_nhsn, log_row_counts as log_nhsn_rows
    from src.build_seasons import build_seasons, save_processed
    from src.analysis_burden import run_burden_analysis
    from src.analysis_trigger import run_trigger_analysis, summarize_trigger_by_season

    # Step 1: Load configuration
    logger.info("Step 1: Loading configuration...")
    config = load_config()
    logger.info(f"  Seasons: {[s['name'] for s in config['seasons']]}")
    logger.info(f"  Primary outcome: {config['primary_outcome']}")
    logger.info(f"  Excluded jurisdictions: {config['geography']['exclude_jurisdictions'][:5]}...")

    labels = config.get("labels", {})
    nssp_metric_label = labels.get(
        "nssp_metric",
        "RSV ED visit percentage of total ED visits, all ages"
    )
    nhsn_metric_label = labels.get(
        "nhsn_metric",
        "Pediatric RSV hospital admissions, ages 0-4"
    )

    # Step 2: Extract NSSP data
    logger.info("\nStep 2: Extracting data from CDC NSSP...")
    df_raw = load_nssp(max_age_days=1, force_refresh=force_refresh)
    log_nssp_rows(df_raw)

    # Step 3: Build seasons
    logger.info("\nStep 3: Building seasons and processing data...")
    df = build_seasons(df_raw)
    save_processed(df, filename="nssp_processed.parquet", also_csv=True)

    # Step 4: Run burden analysis
    logger.info("\nStep 4: Running burden analysis...")
    burden_results = run_burden_analysis(df)

    # Step 5: Run trigger analysis
    logger.info("\nStep 5: Running trigger analysis...")
    trigger_results = run_trigger_analysis(df)

    # Step 6: Save tables
    logger.info("\nStep 6: Saving result tables...")

    # Table 1: Outside fraction summary
    table1 = create_table1(
        burden_results["outside_fraction"],
        burden_results["national_summary"],
        burden_results["regional_summary"],
        total_label=f"Total {nssp_metric_label} (sum of weekly values)"
    )
    save_table(table1, "nssp_table1_outside_fraction_summary")

    # Trigger comparison table
    trigger_table = create_trigger_comparison_table(trigger_results["trigger_coverage"])
    save_table(trigger_table, "nssp_table_trigger_comparison")
    save_table(burden_results["regional_summary"], "nssp_regional_summary")

    # Detailed state-level results
    save_table(
        attach_metric_label(burden_results["outside_fraction"], nssp_metric_label),
        "nssp_outside_fraction_by_state"
    )
    save_table(
        attach_metric_label(burden_results["material_activity"], nssp_metric_label),
        "nssp_material_activity_by_state"
    )
    save_table(
        attach_metric_label(burden_results["extended_windows"], nssp_metric_label),
        "nssp_extended_windows_evaluation"
    )
    save_table(
        attach_metric_label(trigger_results["trigger_coverage"], nssp_metric_label),
        "nssp_trigger_coverage_by_state"
    )

    # Step 7: Log summary
    log_summary_stats(
        burden_results["outside_fraction"],
        trigger_results["trigger_coverage"],
        label="NSSP"
    )

    # Step 8: Extract NHSN data
    logger.info("\nStep 8: Extracting data from CDC NHSN (HRD)...")
    df_nhsn_raw = load_nhsn(max_age_days=1, force_refresh=force_refresh)
    log_nhsn_rows(df_nhsn_raw)

    # Step 9: Build seasons for NHSN
    logger.info("\nStep 9: Building seasons and processing NHSN data...")
    df_nhsn = build_seasons(df_nhsn_raw)
    save_processed(df_nhsn, filename="nhsn_processed.parquet", also_csv=True)

    # Step 10: Run burden analysis (NHSN)
    logger.info("\nStep 10: Running burden analysis (NHSN)...")
    nhsn_value_col = config.get("nhsn_primary_outcome", "rsv_ped_0_4")
    nhsn_burden = run_burden_analysis(df_nhsn, value_col=nhsn_value_col)

    # Step 11: Run trigger analysis (NHSN)
    logger.info("\nStep 11: Running trigger analysis (NHSN)...")
    nhsn_trigger = run_trigger_analysis(df_nhsn, value_col=nhsn_value_col)

    # Step 12: Save NHSN tables
    logger.info("\nStep 12: Saving NHSN result tables...")
    nhsn_table1 = create_table1(
        nhsn_burden["outside_fraction"],
        nhsn_burden["national_summary"],
        nhsn_burden["regional_summary"],
        total_label="Total pediatric RSV admissions, ages 0-4"
    )
    save_table(nhsn_table1, "nhsn_table1_outside_fraction_summary")

    nhsn_trigger_table = create_trigger_comparison_table(nhsn_trigger["trigger_coverage"])
    save_table(nhsn_trigger_table, "nhsn_table_trigger_comparison")
    save_table(nhsn_burden["regional_summary"], "nhsn_regional_summary")

    save_table(
        attach_metric_label(nhsn_burden["outside_fraction"], nhsn_metric_label),
        "nhsn_outside_fraction_by_state"
    )
    save_table(
        attach_metric_label(nhsn_burden["material_activity"], nhsn_metric_label),
        "nhsn_material_activity_by_state"
    )
    save_table(
        attach_metric_label(nhsn_burden["extended_windows"], nhsn_metric_label),
        "nhsn_extended_windows_evaluation"
    )
    save_table(
        attach_metric_label(nhsn_trigger["trigger_coverage"], nhsn_metric_label),
        "nhsn_trigger_coverage_by_state"
    )

    # Step 13: Cross-source comparison tables
    logger.info("\nStep 13: Saving NSSP vs NHSN comparison tables...")
    cross_source = create_cross_source_comparison(
        burden_results,
        nhsn_burden,
        nssp_metric_label=nssp_metric_label,
        nhsn_metric_label=nhsn_metric_label
    )
    save_table(cross_source, "comparison_outside_fraction_2024_2025")

    monthly_compare = create_monthly_metric_comparison(
        df,
        df_nhsn,
        config.get("primary_outcome", "rsv_pct"),
        nhsn_value_col,
        nssp_metric_label=nssp_metric_label,
        nhsn_metric_label=nhsn_metric_label
    )
    save_table(monthly_compare, "comparison_monthly_metric_2024_2025")

    # Step 14: Log NHSN summary
    log_summary_stats(
        nhsn_burden["outside_fraction"],
        nhsn_trigger["trigger_coverage"],
        label="NHSN"
    )

    # Step 15: Generate figures (R/ggplot2)
    logger.info("\nStep 15: Generating figures with R...")
    run_r_figures()

    logger.info(f"\nPipeline completed at: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    return {
        "nssp": {
            "df": df,
            "burden_results": burden_results,
            "trigger_results": trigger_results,
            "figures": None
        },
        "nhsn": {
            "df": df_nhsn,
            "burden_results": nhsn_burden,
            "trigger_results": nhsn_trigger,
            "figures": None
        }
    }


def generate_figures_only() -> dict:
    """
    Regenerate figures using existing processed data.

    For use with `make figures` when data is already processed.
    """
    run_r_figures()
    return {}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RSV Timing / Beyfortus Analysis Pipeline"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetch data from Socrata even if cache exists"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures using existing processed data"
    )

    args = parser.parse_args()

    if args.figures_only:
        generate_figures_only()
    else:
        run_pipeline(force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
