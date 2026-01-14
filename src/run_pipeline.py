"""
Main pipeline orchestration script.

Ties together all analysis components:
1. Data extraction from CDC NSSP
2. Season building and data processing
3. Burden analysis (outside fraction, material activity, extended windows)
4. Trigger analysis (onset detection, coverage evaluation)
5. Figure generation
6. Table output

Usage:
    python -m src.run_pipeline
"""

import logging
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


def create_table1(
    df_outside: pd.DataFrame,
    df_national: pd.DataFrame,
    df_regional: pd.DataFrame
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
            "Total Activity": row.get("total_national_burden", None)
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
            "Total Activity": None
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


def log_summary_stats(
    df_outside: pd.DataFrame,
    df_trigger: pd.DataFrame
) -> None:
    """Log key summary statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
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
    from src.pull_nssp import load_cached_or_fetch, log_row_counts
    from src.build_seasons import build_seasons, save_processed
    from src.analysis_burden import run_burden_analysis
    from src.analysis_trigger import run_trigger_analysis, summarize_trigger_by_season
    from src.figures import generate_all_figures

    # Step 1: Load configuration
    logger.info("Step 1: Loading configuration...")
    config = load_config()
    logger.info(f"  Seasons: {[s['name'] for s in config['seasons']]}")
    logger.info(f"  Primary outcome: {config['primary_outcome']}")
    logger.info(f"  Excluded jurisdictions: {config['geography']['exclude_jurisdictions'][:5]}...")

    # Step 2: Extract data
    logger.info("\nStep 2: Extracting data from CDC NSSP...")
    df_raw = load_cached_or_fetch(max_age_days=1, force_refresh=force_refresh)
    log_row_counts(df_raw)

    # Step 3: Build seasons
    logger.info("\nStep 3: Building seasons and processing data...")
    df = build_seasons(df_raw)
    save_processed(df)

    # Step 4: Run burden analysis
    logger.info("\nStep 4: Running burden analysis...")
    burden_results = run_burden_analysis(df)

    # Step 5: Run trigger analysis
    logger.info("\nStep 5: Running trigger analysis...")
    trigger_results = run_trigger_analysis(df)

    # Step 6: Generate figures
    logger.info("\nStep 6: Generating figures...")
    figures = generate_all_figures(df, burden_results, trigger_results)

    # Step 7: Save tables
    logger.info("\nStep 7: Saving result tables...")

    # Table 1: Outside fraction summary
    table1 = create_table1(
        burden_results["outside_fraction"],
        burden_results["national_summary"],
        burden_results["regional_summary"]
    )
    save_table(table1, "table1_outside_fraction_summary")

    # Trigger comparison table
    trigger_table = create_trigger_comparison_table(trigger_results["trigger_coverage"])
    save_table(trigger_table, "table_trigger_comparison")

    # Detailed state-level results
    save_table(burden_results["outside_fraction"], "outside_fraction_by_state")
    save_table(burden_results["material_activity"], "material_activity_by_state")
    save_table(burden_results["extended_windows"], "extended_windows_evaluation")
    save_table(trigger_results["trigger_coverage"], "trigger_coverage_by_state")

    # Step 8: Log summary
    log_summary_stats(
        burden_results["outside_fraction"],
        trigger_results["trigger_coverage"]
    )

    logger.info(f"\nPipeline completed at: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    return {
        "df": df,
        "burden_results": burden_results,
        "trigger_results": trigger_results,
        "figures": figures
    }


def generate_figures_only() -> dict:
    """
    Regenerate figures using existing processed data.

    For use with `make figures` when data is already processed.
    """
    from src.analysis_burden import run_burden_analysis
    from src.analysis_trigger import run_trigger_analysis
    from src.figures import generate_all_figures

    # Load processed data
    processed_path = DATA_PROCESSED / "nssp_processed.parquet"

    if not processed_path.exists():
        logger.error(f"Processed data not found at {processed_path}")
        logger.error("Run full pipeline first: python -m src.run_pipeline")
        sys.exit(1)

    logger.info(f"Loading processed data from {processed_path}")
    df = pd.read_parquet(processed_path)

    # Re-run analyses
    burden_results = run_burden_analysis(df)
    trigger_results = run_trigger_analysis(df)

    # Generate figures
    figures = generate_all_figures(df, burden_results, trigger_results)

    return figures


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
