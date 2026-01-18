#!/usr/bin/env python3
"""Generate manuscript statistics summary for RSV timing analysis."""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_manuscript_stats(output_path: str = "results/manuscript_stats.txt"):
    """Generate comprehensive statistics for manuscript."""

    root = Path(__file__).parent.parent
    tables_dir = root / "results" / "tables"

    # Load data
    nssp_outside = pd.read_csv(tables_dir / "nssp_outside_fraction_by_state.csv")
    nhsn_outside = pd.read_csv(tables_dir / "nhsn_outside_fraction_by_state.csv")
    nssp_extended = pd.read_csv(tables_dir / "nssp_extended_windows_evaluation.csv")
    nhsn_extended = pd.read_csv(tables_dir / "nhsn_extended_windows_evaluation.csv")
    nssp_trigger = pd.read_csv(tables_dir / "nssp_trigger_coverage_by_state.csv")
    nhsn_trigger = pd.read_csv(tables_dir / "nhsn_trigger_coverage_by_state.csv")

    lines = []

    def add(text=""):
        lines.append(text)

    add("=" * 80)
    add("    MANUSCRIPT DATA SUMMARY: RSV Activity Outside October-March Window")
    add("=" * 80)
    add()
    add("                          RESULTS SECTION VALUES")
    add("=" * 80)
    add()
    add("PARAGRAPH 1 - National and State-Level Findings (Figure 1)")
    add("-" * 80)
    add()
    add("Out-of-Window Fractions:")

    def summarize_outside(df, season, label):
        data = df[df['season'] == season].dropna(subset=['outside_fraction'])
        median = data['outside_fraction'].median() * 100
        q25 = data['outside_fraction'].quantile(0.25) * 100
        q75 = data['outside_fraction'].quantile(0.75) * 100
        min_val = data['outside_fraction'].min() * 100
        max_val = data['outside_fraction'].max() * 100
        n = len(data)

        add(f"\n  {label}:")
        add(f"    • Median: {median:.1f}%")
        add(f"    • IQR: {q25:.1f}% – {q75:.1f}%")
        add(f"    • Range: {min_val:.1f}% – {max_val:.1f}%")
        add(f"    • N = {n} jurisdictions")

        top5 = data.nlargest(5, 'outside_fraction')[['jurisdiction', 'outside_fraction']]
        add(f"    • Top 5: {', '.join([f'{r.jurisdiction} ({r.outside_fraction*100:.1f}%)' for _, r in top5.iterrows()])}")

        return median, q25, q75, min_val, max_val

    summarize_outside(nssp_outside, '2023-2024', "NSSP 2023-24")
    summarize_outside(nssp_outside, '2024-2025', "NSSP 2024-25")
    summarize_outside(nhsn_outside, '2024-2025', "NHSN 2024-25")

    add()
    add("-" * 80)
    add()
    add("PARAGRAPH 2 - Alternative Window Scenarios (Figure 2)")
    add("-" * 80)
    add()
    add("Window Coverage Comparison (Median % of RSV Activity Captured):")
    add()
    add("                          NSSP 2023-24   NSSP 2024-25   NHSN 2024-25")
    add("  " + "-" * 71)

    def get_coverage(ext_df, trig_df, season, window):
        if window == 'dynamic':
            data = trig_df[(trig_df['season'] == season) & trig_df['trigger_coverage'].notna()]
            return data['trigger_coverage'].median() * 100 if len(data) > 0 else np.nan
        else:
            data = ext_df[(ext_df['season'] == season) & (ext_df['window_name'] == window)]
            return data['coverage'].median() * 100 if len(data) > 0 else np.nan

    windows = [
        ('baseline_oct_mar', 'Baseline (Oct-Mar)'),
        ('early_sep_mar', 'Early (Sep-Mar)'),
        ('extended_sep_apr', 'Extended (Sep-Apr)'),
        ('dynamic', 'Dynamic (onset-offset)')
    ]

    for window_key, window_name in windows:
        nssp_2324 = get_coverage(nssp_extended, nssp_trigger, '2023-2024', window_key)
        nssp_2425 = get_coverage(nssp_extended, nssp_trigger, '2024-2025', window_key)
        nhsn_2425 = get_coverage(nhsn_extended, nhsn_trigger, '2024-2025', window_key)
        add(f"  {window_name:<25} {nssp_2324:>7.1f}%        {nssp_2425:>7.1f}%        {nhsn_2425:>7.1f}%")

    add()
    add("Coverage Improvement (Percentage Points vs Baseline):")

    for season, label, ext_df, trig_df in [
        ('2023-2024', 'NSSP 2023-24', nssp_extended, nssp_trigger),
        ('2024-2025', 'NSSP 2024-25', nssp_extended, nssp_trigger),
        ('2024-2025', 'NHSN 2024-25', nhsn_extended, nhsn_trigger)
    ]:
        baseline = get_coverage(ext_df, trig_df, season, 'baseline_oct_mar')
        early = get_coverage(ext_df, trig_df, season, 'early_sep_mar')
        extended = get_coverage(ext_df, trig_df, season, 'extended_sep_apr')
        dynamic = get_coverage(ext_df, trig_df, season, 'dynamic')

        add(f"\n  {label}:")
        add(f"    • Early: +{early - baseline:.1f} pp")
        add(f"    • Extended: +{extended - baseline:.1f} pp")
        add(f"    • Dynamic: +{dynamic - baseline:.1f} pp")

    add()
    add("=" * 80)
    add()
    add("                       METHODOLOGICAL DETAILS")
    add("=" * 80)
    add()
    add("DATA SOURCES")
    add("-" * 80)
    add()
    add("NSSP (National Syndromic Surveillance Program):")
    add("  • Data source: CDC NSSP Emergency Department Visit Trajectories")
    add("  • URL: https://data.cdc.gov/Public-Health-Surveillance/")
    add("         NSSP-Emergency-Department-Visit-Trajectories/rdmq-nq56")
    add("  • Socrata API dataset ID: rdmq-nq56")
    add("  • Field used: 'percent_visits_rsv' (renamed to 'rsv_pct' in analysis)")
    add("  • Definition: Percentage of emergency department visits with RSV-related")
    add("    chief complaint or diagnosis code, among all ED visits, all ages")
    add("  • Temporal resolution: Weekly (MMWR week ending date)")
    add("  • Geographic resolution: State-level")
    add("  • Seasons analyzed: 2023-24 and 2024-25")
    add("  • Jurisdictions included: 46 states + District of Columbia (N=47)")
    add("  • Excluded jurisdictions:")
    add("    - Florida: Excluded per CDC year-round nirsevimab guidance")
    add("    - Alaska, Hawaii: Different RSV timing patterns; separate guidance")
    add("    - Missouri, South Dakota: No NSSP data available in source dataset")
    add()
    add("NHSN (National Healthcare Safety Network):")
    add("  • Data source: CDC NHSN Weekly Hospital Respiratory Data (HRD)")
    add("  • URL: https://data.cdc.gov/Public-Health-Surveillance/")
    add("         Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-/ua7e-t2fy")
    add("  • Socrata API dataset ID: ua7e-t2fy")
    add("  • Field used: 'totalconfrsvnewadmped0to4' (renamed to 'rsv_ped_0_4')")
    add("  • Definition: Count of laboratory-confirmed RSV-associated new hospital")
    add("    admissions among pediatric patients aged 0-4 years")
    add("  • Temporal resolution: Weekly (week ending date)")
    add("  • Geographic resolution: State-level")
    add("  • Seasons analyzed: 2024-25 only (earlier data had completeness issues)")
    add("  • Jurisdictions included: 46 states + District of Columbia (N=47)")
    add("  • Excluded jurisdictions: Same as NSSP, plus regional/national aggregates")
    add()
    add("SEASON DEFINITION")
    add("-" * 80)
    add("  • RSV season defined as July 1 through June 30 of the following year")
    add("  • This definition captures early summer activity that may precede")
    add("    the traditional fall-winter season")
    add("  • 2023-24 season: July 1, 2023 – June 30, 2024")
    add("  • 2024-25 season: July 1, 2024 – June 30, 2025")
    add()
    add("WINDOW DEFINITIONS")
    add("-" * 80)
    add()
    add("Fixed Windows:")
    add("  • Baseline (Oct-Mar): October 1 – March 31")
    add("    Corresponds to current CDC nirsevimab guidance for continental US")
    add("  • Early (Sep-Mar): September 1 – March 31")
    add("    One-month earlier start to capture early-season activity")
    add("  • Extended (Sep-Apr): September 1 – April 30")
    add("    Earlier start plus one-month later end")
    add()
    add("Dynamic (State-Specific Onset-Offset) Window:")
    add("  • Onset and offset detected separately for each state and season")
    add("  • Onset definition: First week of sustained RSV activity increase")
    add("  • Offset definition: First week after peak when activity returns to baseline")
    add("  • See detailed algorithm description below")
    add()
    add("ONSET DETECTION ALGORITHM")
    add("-" * 80)
    add()
    add("The onset detection algorithm identifies the beginning of the RSV season")
    add("using a growth-rate approach that is sensitive to early epidemic signals:")
    add()
    add("1. Data Preparation:")
    add("   • Weekly RSV values are smoothed using a 3-week rolling average")
    add("   • Baseline is computed as the median of the first 4 weeks of the season")
    add("     (July values, representing pre-season activity)")
    add()
    add("2. Growth Rate Calculation:")
    add("   • Week-over-week growth rate computed as: value(week t) / value(week t-1)")
    add()
    add("3. Onset Criteria (Primary):")
    add("   • Onset = first week where BOTH conditions are met for 2 consecutive weeks:")
    add("     a) Growth rate >10% (i.e., ratio >1.10)")
    add("     b) Smoothed value exceeds minimum threshold (0.08% for NSSP)")
    add()
    add("4. Onset Criteria (Secondary):")
    add("   • If primary criteria not met, onset = first week where:")
    add("     a) Smoothed value exceeds baseline + 1 SD OR 30% above baseline")
    add("     b) Any positive growth (ratio ≥1.0) for 2 consecutive weeks")
    add()
    add("OFFSET DETECTION ALGORITHM")
    add("-" * 80)
    add()
    add("The offset detection algorithm identifies when the RSV season has ended")
    add("using a conservative return-to-baseline approach:")
    add()
    add("1. Peak Identification:")
    add("   • Peak week = week with maximum smoothed RSV value after detected onset")
    add()
    add("2. Baseline Threshold:")
    add("   • Return-to-baseline threshold = MAX of:")
    add("     a) Baseline median + 3 standard deviations")
    add("     b) 3× baseline median")
    add("     c) Minimum value (0.15% for NSSP)")
    add("   • This conservative threshold ensures the season is truly over")
    add()
    add("3. Offset Criteria:")
    add("   • Offset = first week AFTER the peak where smoothed value remains")
    add("     at or below the baseline threshold for 2 consecutive weeks")
    add()
    add("BURDEN AND COVERAGE CALCULATIONS")
    add("-" * 80)
    add()
    add("Overview:")
    add("  For each state and season, we computed the fraction of total RSV activity")
    add("  that occurred outside the October 1 – March 31 window. This 'out-of-window")
    add("  fraction' represents the proportion of RSV burden that would NOT be covered")
    add("  by nirsevimab administered during the recommended window.")
    add()
    add("Step-by-Step Calculation:")
    add()
    add("  1. For each state-season, sum the weekly RSV metric across all weeks:")
    add("     Total burden = Σ (weekly RSV value) for weeks in Jul 1 – Jun 30")
    add()
    add("  2. Sum the weekly RSV metric for weeks WITHIN the Oct–Mar window:")
    add("     In-window burden = Σ (weekly RSV value) for weeks in Oct 1 – Mar 31")
    add()
    add("  3. Compute out-of-window burden by subtraction:")
    add("     Out-of-window burden = Total burden − In-window burden")
    add()
    add("  4. Compute the out-of-window fraction:")
    add("     Out-of-window fraction = Out-of-window burden / Total burden")
    add()
    add("  5. Coverage is the complement:")
    add("     Coverage = 1 − Out-of-window fraction = In-window burden / Total burden")
    add()
    add("Interpretation by Data Source:")
    add()
    add("  NSSP (ED visit percentage):")
    add("    • Weekly values are percentages (e.g., 1.5% of ED visits were RSV-related)")
    add("    • 'Burden' = sum of weekly percentages across the season")
    add("    • Example: If total season burden = 15.0 (sum of weekly %s) and")
    add("      in-window burden = 14.0, then out-of-window fraction = 1.0/15.0 = 6.7%")
    add("    • This represents the fraction of RSV-related ED visit activity outside window")
    add()
    add("  NHSN (hospital admission counts):")
    add("    • Weekly values are counts (e.g., 150 RSV admissions that week)")
    add("    • 'Burden' = sum of weekly admission counts across the season")
    add("    • Example: If total season admissions = 2,000 and in-window = 1,900,")
    add("      then out-of-window fraction = 100/2,000 = 5.0%")
    add("    • This represents the fraction of RSV hospitalizations outside window")
    add()
    add("Alternative Window Evaluation:")
    add("  The same calculation was repeated for alternative window definitions:")
    add("    • Early (Sep–Mar): In-window = Sep 1 – Mar 31")
    add("    • Extended (Sep–Apr): In-window = Sep 1 – Apr 30")
    add("    • Dynamic: In-window = state-specific onset week to offset week")
    add()
    add("STATISTICAL SUMMARIES")
    add("-" * 80)
    add("  • State-level values computed separately for each jurisdiction and season")
    add("  • National summaries: Median and interquartile range (IQR) across states")
    add("  • Range (minimum to maximum) reported to characterize variability")
    add()
    add("DYNAMIC WINDOW TIMING RESULTS")
    add("-" * 80)
    add()

    # Load trigger data for timing details
    for season, label, trig_df in [
        ('2023-2024', 'NSSP 2023-24', nssp_trigger),
        ('2024-2025', 'NSSP 2024-25', nssp_trigger),
        ('2024-2025', 'NHSN 2024-25', nhsn_trigger)
    ]:
        trig = trig_df[trig_df['season'] == season].copy()
        trig = trig[trig['onset_week'].notna() & trig['offset_week'].notna()]

        if len(trig) > 0:
            trig['onset_week'] = pd.to_datetime(trig['onset_week'])
            trig['offset_week'] = pd.to_datetime(trig['offset_week'])

            onset_min = trig['onset_week'].min()
            onset_max = trig['onset_week'].max()
            offset_min = trig['offset_week'].min()
            offset_max = trig['offset_week'].max()

            earliest_state = trig.loc[trig['onset_week'].idxmin(), 'jurisdiction']
            latest_state = trig.loc[trig['onset_week'].idxmax(), 'jurisdiction']

            add(f"{label}:")
            add(f"  • States with detected onset/offset: {len(trig)}")
            add(f"  • Onset range: {onset_min.strftime('%B %d')} – {onset_max.strftime('%B %d')}")
            add(f"  • Earliest onset: {earliest_state} ({onset_min.strftime('%b %d')})")
            add(f"  • Latest onset: {latest_state} ({onset_max.strftime('%b %d')})")
            add(f"  • Offset range: {offset_min.strftime('%B %d')} – {offset_max.strftime('%B %d')}")
            add()

    add("=" * 80)
    add()
    add("                     SUGGESTED TEXT FOR MANUSCRIPT")
    add("=" * 80)
    add()

    # Get values for suggested text
    nssp_2324_data = nssp_outside[nssp_outside['season'] == '2023-2024'].dropna(subset=['outside_fraction'])
    nssp_2425_data = nssp_outside[nssp_outside['season'] == '2024-2025'].dropna(subset=['outside_fraction'])
    nhsn_2425_data = nhsn_outside[nhsn_outside['season'] == '2024-2025'].dropna(subset=['outside_fraction'])

    add("RESULTS PARAGRAPH 1:")
    add("-" * 80)
    add(f"Across states, a median of {nssp_2324_data['outside_fraction'].median()*100:.1f}% "
        f"(IQR: {nssp_2324_data['outside_fraction'].quantile(0.25)*100:.1f}–"
        f"{nssp_2324_data['outside_fraction'].quantile(0.75)*100:.1f}%) of RSV-associated ED visits "
        f"(NSSP) occurred outside the October–March window in 2023-24, and "
        f"{nssp_2425_data['outside_fraction'].median()*100:.1f}% "
        f"(IQR: {nssp_2425_data['outside_fraction'].quantile(0.25)*100:.1f}–"
        f"{nssp_2425_data['outside_fraction'].quantile(0.75)*100:.1f}%) in 2024-25. "
        f"For RSV-associated hospitalizations (NHSN, 2024-25), the median was "
        f"{nhsn_2425_data['outside_fraction'].median()*100:.1f}% "
        f"(IQR: {nhsn_2425_data['outside_fraction'].quantile(0.25)*100:.1f}–"
        f"{nhsn_2425_data['outside_fraction'].quantile(0.75)*100:.1f}%).")
    add()
    add("DISCUSSION PARAGRAPH 1:")
    add("-" * 80)
    median_overall = np.median([
        nssp_2324_data['outside_fraction'].median(),
        nssp_2425_data['outside_fraction'].median(),
        nhsn_2425_data['outside_fraction'].median()
    ]) * 100
    max_outside = max(
        nssp_2324_data['outside_fraction'].max(),
        nssp_2425_data['outside_fraction'].max(),
        nhsn_2425_data['outside_fraction'].max()
    ) * 100
    add(f"Approximately {median_overall:.0f}% of RSV activity nationally—and up to "
        f"{max_outside:.0f}% in some states—occurs outside the current recommended "
        f"nirsevimab administration window.")
    add()
    add("=" * 80)

    # Write to file
    output_file = root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Manuscript statistics saved to {output_path}")
    return output_file


if __name__ == "__main__":
    generate_manuscript_stats()
