#!/usr/bin/env Rscript
#
# figures.R - Generate publication figures for RSV timing analysis
#
# This script reads processed data and analysis tables to create:
#   - State-level choropleths of outside-window fractions
#   - Regional choropleths and bar charts
#   - Time series plots with trigger windows
#   - Coverage comparison bar charts
#   - Combined multi-panel figures
#

# =============================================================================
# SETUP AND DEPENDENCIES
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(lubridate)
})

has_cowplot <- requireNamespace("cowplot", quietly = TRUE)
has_sf <- requireNamespace("sf", quietly = TRUE)
has_maps <- requireNamespace("maps", quietly = TRUE)

root <- getwd()
fig_dir <- file.path(root, "results", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

config <- yaml::read_yaml(file.path(root, "config.yaml"))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

default_or <- function(value, fallback) {
  if (is.null(value)) return(fallback)
  if (is.character(value) && length(value) == 1 && (is.na(value) || value == "")) {
    return(fallback)
  }
  value
}

normalize_state_names <- function(x) {
  x <- str_to_title(x)
  x <- str_replace_all(x, " Of ", " of ")
  x <- str_replace_all(x, " And ", " and ")
  x <- ifelse(x == "District Of Columbia", "District of Columbia", x)
  x
}

outside_fraction_label <- function(metric_label, prefix = "Fraction") {
  base_label <- if (!is.null(metric_label) && !is.na(metric_label) && metric_label != "") {
    metric_label
  } else {
    "RSV activity"
  }
  if (str_detect(tolower(prefix), "median")) {
    return(sprintf("Median fraction of %s outside window", base_label))
  }
  sprintf("Fraction of %s outside window", base_label)
}

save_plot <- function(plot, filename, width = 10, height = 6) {
  ggsave(file.path(fig_dir, paste0(filename, ".png")),
         plot = plot, width = width, height = height, dpi = 300)
  ggsave(file.path(fig_dir, paste0(filename, ".pdf")),
         plot = plot, width = width, height = height, dpi = 300)
}

# =============================================================================
# CONFIGURATION
# =============================================================================

labels <- default_or(config$labels, list())
nssp_timeseries_label <- default_or(labels$nssp_timeseries, "RSV ED visit %")
nhsn_timeseries_label <- default_or(labels$nhsn_timeseries, "RSV-associated hospital admissions")
nssp_fraction_label <- default_or(labels$nssp_fraction, nssp_timeseries_label)
nhsn_fraction_label <- default_or(labels$nhsn_fraction, nhsn_timeseries_label)
fixed_window <- config$fixed_window
se_states <- default_or(config$geography$southeast_states, character(0))

state_abbrev <- tibble(
  jurisdiction = c(state.name, "District of Columbia"),
  state_abbrev = c(state.abb, "DC")
)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

read_processed <- function(prefix) {
  csv_path <- file.path(root, "data", "processed", paste0(prefix, "_processed.csv"))
  parquet_path <- file.path(root, "data", "processed", paste0(prefix, "_processed.parquet"))

  if (file.exists(csv_path)) return(read_csv(csv_path, show_col_types = FALSE))
  if (file.exists(parquet_path) && requireNamespace("arrow", quietly = TRUE)) {
    return(arrow::read_parquet(parquet_path))
  }
  stop(paste("Processed data not found for", prefix))
}

read_table <- function(name) {
  path <- file.path(root, "results", "tables", paste0(name, ".csv"))
  if (!file.exists(path)) stop(paste("Required table not found:", path))
  read_csv(path, show_col_types = FALSE)
}

get_states_sf <- function() {
  if (!has_maps || !has_sf) {
    message("Skipping choropleth maps because 'maps' or 'sf' is not installed.")
    return(NULL)
  }
  map_data <- maps::map("state", plot = FALSE, fill = TRUE)
  states_sf <- sf::st_as_sf(map_data)
  states_sf <- states_sf %>%
    mutate(jurisdiction = normalize_state_names(ID))

  geom_col <- attr(states_sf, "sf_column")
  if (!identical(geom_col, "geometry")) {
    names(states_sf)[names(states_sf) == geom_col] <- "geometry"
    attr(states_sf, "sf_column") <- "geometry"
  }
  states_sf
}

# =============================================================================
# THEME FUNCTIONS
# =============================================================================

barplot_theme <- function(orientation = c("horizontal", "vertical")) {
  orientation <- match.arg(orientation)
  base_theme <- if (has_cowplot) cowplot::theme_minimal_grid() else theme_minimal()

  if (orientation == "horizontal") {
    base_theme + theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(0, "pt")
    )
  } else {
    base_theme + theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(0, "pt")
    )
  }
}

# =============================================================================
# STATE-LEVEL CHOROPLETH FUNCTIONS
# =============================================================================

plot_state_choropleth <- function(outside_df, metric_label, prefix, highlight_southeast,
                                   cat_label_margin = -2, x_title_size = 9) {
  states_sf <- get_states_sf()
  outside_df <- outside_df %>%
    filter(!is.na(season)) %>%
    mutate(jurisdiction = normalize_state_names(jurisdiction))

  if (is.null(states_sf)) {
    plot_state_barchart(outside_df, metric_label, prefix, highlight_southeast)
    return(invisible(NULL))
  }

  seasons <- sort(unique(outside_df$season))
  states_seasons <- states_sf[rep(seq_len(nrow(states_sf)), each = length(seasons)), ]
  states_seasons$season <- rep(seasons, times = nrow(states_sf))

  joined <- states_seasons %>%
    left_join(outside_df, by = c("jurisdiction", "season"))

  vmax <- min(quantile(outside_df$outside_fraction, 0.95, na.rm = TRUE), 0.25)

  joined_projected <- sf::st_transform(joined, "ESRI:102003")
  bbox <- sf::st_bbox(joined_projected)
  aspect_ratio <- as.numeric((bbox["xmax"] - bbox["xmin"]) / (bbox["ymax"] - bbox["ymin"]))

  p <- ggplot(joined) +
    geom_sf(aes(fill = outside_fraction), color = "gray70", linewidth = 0.2) +
    coord_sf(crs = "ESRI:102003", datum = NA) +
    facet_wrap(~season) +
    scale_fill_gradient(
      low = "#fee5d9",
      high = "#a50f15",
      limits = c(0, vmax),
      oob = scales::squish,
      na.value = "lightgray",
      name = outside_fraction_label(metric_label, prefix = "Fraction"),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(12, "lines"),
        barheight = unit(0.6, "lines")
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 10),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8)
    )

  n_seasons <- n_distinct(outside_df$season)
  panel_width <- 5
  panel_height <- panel_width / aspect_ratio
  map_width <- panel_width * n_seasons
  map_height <- panel_height + 0.8
  save_plot(p, paste0(prefix, "fig1_choropleth"), width = map_width, height = map_height)

  plot_state_barchart(outside_df, metric_label, prefix, highlight_southeast,
                      cat_label_margin = cat_label_margin, x_title_size = x_title_size)
}

plot_state_barchart <- function(outside_df, metric_label, prefix, highlight_southeast,
                                 cat_label_margin = -2, x_title_size = 9) {
  df <- outside_df %>%
    filter(!is.na(season), !is.na(outside_fraction)) %>%
    mutate(jurisdiction = normalize_state_names(jurisdiction)) %>%
    left_join(state_abbrev, by = "jurisdiction") %>%
    mutate(state_label = coalesce(state_abbrev, jurisdiction))

  state_levels <- df %>%
    distinct(state_label) %>%
    arrange(desc(state_label)) %>%
    pull(state_label)

  df <- df %>%
    mutate(state_label = factor(state_label, levels = state_levels))

  if (highlight_southeast) {
    df <- df %>%
      mutate(group = if_else(jurisdiction %in% se_states, "Southeast states", "Other states"))
  } else {
    df <- df %>%
      mutate(group = "All states")
  }

  p <- ggplot(df, aes(x = state_label, y = outside_fraction, fill = group)) +
    geom_col() +
    coord_flip() +
    facet_wrap(~season, scales = "free_y") +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_fill_manual(values = c(
      "Southeast states" = "#D55E00",
      "Other states" = "#0072B2",
      "All states" = "#4C78A8"
    )) +
    labs(
      x = NULL,
      y = outside_fraction_label(metric_label, prefix = "Fraction"),
      fill = NULL
    ) +
    barplot_theme("horizontal") +
    theme(
      legend.position = if (highlight_southeast) "bottom" else "none",
      strip.text = element_text(size = 9),
      axis.text.x = element_text(size = 8),
      axis.text.y = element_text(size = 8, margin = margin(r = cat_label_margin)),
      axis.title.x = element_text(size = x_title_size, margin = margin(t = 6)),
      axis.title.y = element_text(size = x_title_size)
    )

  save_plot(p, paste0(prefix, "fig1_barchart"), width = 7, height = 8)
}

make_single_choropleth <- function(outside_df, metric_label, season_filter, title = NULL) {
  states_sf <- get_states_sf()
  if (is.null(states_sf)) return(NULL)

  df <- outside_df %>%
    filter(!is.na(season), season == season_filter) %>%
    mutate(jurisdiction = normalize_state_names(jurisdiction))

  joined <- states_sf %>%
    left_join(df, by = "jurisdiction")

  vmax <- min(quantile(outside_df$outside_fraction, 0.95, na.rm = TRUE), 0.25)

  p <- ggplot(joined) +
    geom_sf(aes(fill = outside_fraction), color = "gray70", linewidth = 0.2) +
    coord_sf(crs = "ESRI:102003", datum = NA) +
    scale_fill_gradient(
      low = "#fee5d9",
      high = "#a50f15",
      limits = c(0, vmax),
      oob = scales::squish,
      na.value = "lightgray",
      name = outside_fraction_label(metric_label, prefix = "Fraction"),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(8, "lines"),
        barheight = unit(0.5, "lines")
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 7)
    )

  if (!is.null(title)) {
    p <- p + ggtitle(title) + theme(plot.title = element_text(size = 10, hjust = 0.5))
  }
  p
}

plot_combined_choropleth <- function(nssp_outside, nhsn_outside, nssp_label, nhsn_label) {
  if (!has_cowplot || !has_sf || !has_maps) {
    message("Skipping combined choropleth: cowplot, sf, or maps not available.")
    return(invisible(NULL))
  }

  states_sf <- get_states_sf()
  states_projected <- sf::st_transform(states_sf, "ESRI:102003")
  bbox <- sf::st_bbox(states_projected)
  aspect_ratio <- as.numeric((bbox["xmax"] - bbox["xmin"]) / (bbox["ymax"] - bbox["ymin"]))

  nssp_seasons <- sort(unique(nssp_outside$season[!is.na(nssp_outside$season)]))
  nhsn_seasons <- sort(unique(nhsn_outside$season[!is.na(nhsn_outside$season)]))

  p_a <- make_single_choropleth(nssp_outside, nssp_label, nssp_seasons[1],
                                 paste0("NSSP ", nssp_seasons[1]))
  p_b <- make_single_choropleth(nssp_outside, nssp_label, nssp_seasons[2],
                                 paste0("NSSP ", nssp_seasons[2]))
  p_c <- make_single_choropleth(nhsn_outside, nhsn_label, nhsn_seasons[1],
                                 paste0("NHSN ", nhsn_seasons[1]))
  p_empty <- ggplot() + theme_void()

  combined <- cowplot::plot_grid(
    p_a, p_b,
    p_empty, p_c,
    labels = c("A", "B", "", "C"),
    label_size = 14,
    ncol = 2,
    nrow = 2
  )

  fig_width <- 14
  fig_height <- fig_width / aspect_ratio + 1.5
  save_plot(combined, "combined_state_choropleth", width = fig_width, height = fig_height)
}

# =============================================================================
# REGIONAL CHOROPLETH FUNCTIONS
# =============================================================================

plot_regional_choropleth <- function(outside_df, metric_label, prefix) {
  states_sf <- get_states_sf()
  if (is.null(states_sf)) return(invisible(NULL))

  region_map <- config$hhs_regions
  region_df <- tibble(
    jurisdiction = unlist(region_map),
    hhs_region = as.integer(rep(names(region_map), lengths(region_map)))
  )

  regional_medians <- outside_df %>%
    filter(!is.na(season)) %>%
    mutate(jurisdiction = normalize_state_names(jurisdiction)) %>%
    left_join(region_df, by = "jurisdiction") %>%
    group_by(season, hhs_region) %>%
    summarise(median_outside_fraction = median(outside_fraction, na.rm = TRUE), .groups = "drop")

  seasons <- sort(unique(regional_medians$season))
  states_regions <- states_sf %>%
    left_join(region_df, by = "jurisdiction") %>%
    filter(!is.na(hhs_region))

  states_seasons_list <- lapply(seasons, function(s) {
    states_regions %>% mutate(season = s)
  })
  states_regions <- do.call(rbind, states_seasons_list)
  states_regions <- states_regions %>%
    left_join(regional_medians, by = c("hhs_region", "season"))
  states_regions <- sf::st_as_sf(states_regions)

  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  joined <- states_regions %>%
    sf::st_make_valid() %>%
    group_by(season, hhs_region, median_outside_fraction) %>%
    summarise(.groups = "drop")

  vmax <- min(max(joined$median_outside_fraction, na.rm = TRUE) * 1.1, 0.20)

  joined_projected <- sf::st_transform(joined, "ESRI:102003")
  bbox <- sf::st_bbox(joined_projected)
  aspect_ratio <- as.numeric((bbox["xmax"] - bbox["xmin"]) / (bbox["ymax"] - bbox["ymin"]))

  p <- ggplot(joined) +
    geom_sf(aes(fill = median_outside_fraction), color = "black", linewidth = 0.3) +
    coord_sf(crs = "ESRI:102003", datum = NA) +
    facet_wrap(~season) +
    scale_fill_gradient(
      low = "#fee5d9",
      high = "#a50f15",
      limits = c(0, vmax),
      oob = scales::squish,
      na.value = "lightgray",
      name = outside_fraction_label(metric_label, prefix = "Median"),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(12, "lines"),
        barheight = unit(0.6, "lines")
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      strip.text = element_text(size = 10),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8)
    )

  n_seasons <- n_distinct(outside_df$season)
  panel_width <- 5
  panel_height <- panel_width / aspect_ratio
  map_width <- panel_width * n_seasons
  map_height <- panel_height + 0.8
  save_plot(p, paste0(prefix, "fig1b_regional_choropleth"), width = map_width, height = map_height)
}

plot_regional_barchart <- function(regional_df, metric_label, prefix, plot_width = 6) {
  df <- regional_df %>%
    mutate(
      hhs_region = as.integer(hhs_region),
      region_label = paste("Region", hhs_region)
    )

  region_levels <- df %>%
    distinct(region_label, hhs_region) %>%
    arrange(desc(hhs_region)) %>%
    pull(region_label)

  df <- df %>%
    mutate(region_label = factor(region_label, levels = region_levels))

  p <- ggplot(df, aes(x = region_label, y = median_outside_fraction)) +
    geom_col(fill = "#4C78A8") +
    coord_flip() +
    facet_wrap(~season, scales = "free_y") +
    scale_x_discrete(expand = expansion(add = 0)) +
    labs(x = NULL, y = outside_fraction_label(metric_label, prefix = "Median")) +
    barplot_theme("horizontal") +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 9),
      axis.text.x = element_text(size = 8),
      axis.text.y = element_text(size = 8, margin = margin(r = -6)),
      axis.title.x = element_text(size = 9),
      axis.title.y = element_text(size = 9)
    )

  save_plot(p, paste0(prefix, "fig_supp_regional_heatmap"), width = plot_width, height = 5.5)
}

# =============================================================================
# TIME SERIES FUNCTIONS
# =============================================================================

plot_timeseries <- function(df, trigger_df, metric_label, prefix, value_col, free_y = FALSE) {
  df <- df %>%
    mutate(week_end = as.Date(week_end)) %>%
    filter(!is.na(.data[[value_col]]), !is.na(season))

  if (!"state_abbrev" %in% names(df)) {
    df <- df %>% left_join(state_abbrev, by = "jurisdiction")
  }
  df <- df %>%
    mutate(facet_label = coalesce(.data$state_abbrev, jurisdiction))

  trigger_df <- trigger_df %>%
    mutate(
      onset_week = as.Date(onset_week),
      offset_week = as.Date(offset_week)
    )
  if (!"state_abbrev" %in% names(trigger_df)) {
    trigger_df <- trigger_df %>% left_join(state_abbrev, by = "jurisdiction")
  }

  season_windows <- df %>%
    distinct(season) %>%
    mutate(
      start_year = as.integer(str_split(season, "-", simplify = TRUE)[, 1]),
      end_year = as.integer(str_split(season, "-", simplify = TRUE)[, 2]),
      window_start = as.Date(sprintf("%d-%02d-%02d", start_year,
                                      fixed_window$start_month, fixed_window$start_day)),
      window_end = as.Date(sprintf("%d-%02d-%02d", end_year,
                                    fixed_window$end_month, fixed_window$end_day))
    )

  trigger_lines <- trigger_df %>%
    filter(!is.na(season)) %>%
    select(jurisdiction, onset_week, offset_week, state_abbrev) %>%
    mutate(facet_label = coalesce(.data$state_abbrev, jurisdiction)) %>%
    pivot_longer(cols = c(onset_week, offset_week), names_to = "type", values_to = "date") %>%
    filter(!is.na(date)) %>%
    mutate(date = as.Date(date))

  p <- ggplot(df, aes(x = week_end, y = .data[[value_col]], color = season)) +
    geom_rect(
      inherit.aes = FALSE,
      data = season_windows,
      aes(xmin = window_start, xmax = window_end, ymin = -Inf, ymax = Inf),
      fill = "green",
      alpha = 0.15
    ) +
    geom_line(linewidth = 0.4) +
    geom_vline(
      data = trigger_lines,
      aes(xintercept = date),
      color = "black",
      linetype = "dashed",
      linewidth = 0.3
    ) +
    facet_wrap(~facet_label, ncol = 6, scales = if (free_y) "free_y" else "fixed") +
    labs(x = "Week", y = metric_label, color = "Season") +
    theme_minimal() +
    theme(
      axis.title.x = element_text(size = 10, margin = margin(t = 10)),
      axis.title.y = element_text(size = 10, margin = margin(r = 10)),
      strip.text = element_text(size = 7),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      axis.text.x = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5),
      axis.text.y = element_text(size = 9)
    )

  save_plot(p, paste0(prefix, "fig2_timeseries_all_seasons"), width = 18, height = 10)
}

# =============================================================================
# COVERAGE COMPARISON FUNCTIONS
# =============================================================================

plot_coverage_comparison <- function(extended_df, trigger_df, prefix) {
  label_map <- c(
    baseline_oct_mar = "Baseline\n(Oct-Mar)",
    early_sep_mar = "Early\n(Sep-Mar)",
    extended_sep_apr = "Extended\n(Sep-Apr)"
  )

  extended_summary <- extended_df %>%
    group_by(season, window_name) %>%
    summarise(
      median_coverage = median(coverage, na.rm = TRUE),
      q25 = quantile(coverage, 0.25, na.rm = TRUE),
      q75 = quantile(coverage, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(window = label_map[window_name])

  trigger_summary <- trigger_df %>%
    filter(!is.na(trigger_coverage)) %>%
    group_by(season) %>%
    summarise(
      median_coverage = median(trigger_coverage, na.rm = TRUE),
      q25 = quantile(trigger_coverage, 0.25, na.rm = TRUE),
      q75 = quantile(trigger_coverage, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(window = "Dynamic\n(onset-offset)")

  coverage_df <- bind_rows(
    extended_summary %>% select(season, window, median_coverage, q25, q75),
    trigger_summary %>% select(season, window, median_coverage, q25, q75)
  )

  palette <- c(
    "Baseline\n(Oct-Mar)" = "#4C78A8",
    "Early\n(Sep-Mar)" = "#72B7B2",
    "Extended\n(Sep-Apr)" = "#F58518",
    "Dynamic\n(onset-offset)" = "#54A24B"
  )

  p <- ggplot(coverage_df, aes(x = window, y = median_coverage, fill = window)) +
    geom_col(color = "black", linewidth = 0.2, width = 0.6) +
    geom_errorbar(aes(ymin = q25, ymax = q75), width = 0.04, linewidth = 0.5) +
    scale_fill_manual(values = palette) +
    facet_wrap(~season) +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.05)) +
    labs(x = NULL, y = "Coverage share", fill = NULL) +
    barplot_theme("vertical") +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 10),
      axis.text.x = element_text(size = 8, angle = 0, margin = margin(t = -6)),
      axis.text.y = element_text(size = 8),
      axis.title.y = element_text(size = 9, margin = margin(r = 6)),
      panel.spacing.x = unit(2, "lines")
    )

  save_plot(p, paste0(prefix, "fig3_coverage_comparison"), width = 8, height = 5)
}

make_single_coverage_plot <- function(extended_df, trigger_df, season_filter, title = NULL) {
  label_map <- c(
    baseline_oct_mar = "Baseline\n(Oct-Mar)",
    early_sep_mar = "Early\n(Sep-Mar)",
    extended_sep_apr = "Extended\n(Sep-Apr)"
  )

  extended_summary <- extended_df %>%
    filter(season == season_filter) %>%
    group_by(window_name) %>%
    summarise(
      median_coverage = median(coverage, na.rm = TRUE),
      q25 = quantile(coverage, 0.25, na.rm = TRUE),
      q75 = quantile(coverage, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(window = label_map[window_name])

  trigger_summary <- trigger_df %>%
    filter(season == season_filter, !is.na(trigger_coverage)) %>%
    summarise(
      median_coverage = median(trigger_coverage, na.rm = TRUE),
      q25 = quantile(trigger_coverage, 0.25, na.rm = TRUE),
      q75 = quantile(trigger_coverage, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(window = "Dynamic\n(onset-offset)")

  coverage_df <- bind_rows(
    extended_summary %>% select(window, median_coverage, q25, q75),
    trigger_summary %>% select(window, median_coverage, q25, q75)
  )

  palette <- c(
    "Baseline\n(Oct-Mar)" = "#4C78A8",
    "Early\n(Sep-Mar)" = "#72B7B2",
    "Extended\n(Sep-Apr)" = "#F58518",
    "Dynamic\n(onset-offset)" = "#54A24B"
  )

  p <- ggplot(coverage_df, aes(x = window, y = median_coverage, fill = window)) +
    geom_col(color = "black", linewidth = 0.2, width = 0.6) +
    geom_errorbar(aes(ymin = q25, ymax = q75), width = 0.04, linewidth = 0.5) +
    scale_fill_manual(values = palette) +
    scale_x_discrete(expand = expansion(add = 0.5)) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.05), limits = c(0, 1)) +
    labs(x = NULL, y = "Coverage share", fill = NULL) +
    barplot_theme("vertical") +
    theme(
      legend.position = "none",
      axis.text.x = element_text(size = 8, angle = 0),
      axis.text.y = element_text(size = 8),
      axis.title.y = element_text(size = 9, margin = margin(r = 6))
    )

  if (!is.null(title)) {
    p <- p + ggtitle(title) + theme(plot.title = element_text(size = 10, hjust = 0.5))
  }
  p
}

plot_combined_coverage <- function(nssp_extended, nssp_trigger, nhsn_extended, nhsn_trigger) {
  if (!has_cowplot) {
    message("Skipping combined coverage: cowplot not available.")
    return(invisible(NULL))
  }

  nssp_seasons <- sort(unique(nssp_extended$season[!is.na(nssp_extended$season)]))
  nhsn_seasons <- sort(unique(nhsn_extended$season[!is.na(nhsn_extended$season)]))

  p_a <- make_single_coverage_plot(nssp_extended, nssp_trigger, nssp_seasons[1],
                                    paste0("NSSP ", nssp_seasons[1]))
  p_b <- make_single_coverage_plot(nssp_extended, nssp_trigger, nssp_seasons[2],
                                    paste0("NSSP ", nssp_seasons[2]))
  p_c <- make_single_coverage_plot(nhsn_extended, nhsn_trigger, nhsn_seasons[1],
                                    paste0("NHSN ", nhsn_seasons[1]))
  p_empty <- ggplot() + theme_void()

  combined <- cowplot::plot_grid(
    p_a, p_b,
    p_empty, p_c,
    labels = c("A", "B", "", "C"),
    label_size = 14,
    ncol = 2,
    nrow = 2
  )

  save_plot(combined, "combined_coverage_comparison", width = 10, height = 8)
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Load data
nssp_outside <- read_table("nssp_outside_fraction_by_state")
nhsn_outside <- read_table("nhsn_outside_fraction_by_state")
nssp_extended <- read_table("nssp_extended_windows_evaluation")
nhsn_extended <- read_table("nhsn_extended_windows_evaluation")
nssp_trigger <- read_table("nssp_trigger_coverage_by_state")
nhsn_trigger <- read_table("nhsn_trigger_coverage_by_state")
nssp_regional <- read_table("nssp_regional_summary")
nhsn_regional <- read_table("nhsn_regional_summary")

nssp_processed <- read_processed("nssp")
nhsn_processed <- read_processed("nhsn")

# Generate NSSP figures
plot_state_choropleth(nssp_outside, nssp_fraction_label, "nssp_", highlight_southeast = FALSE)
plot_regional_choropleth(nssp_outside, nssp_fraction_label, "nssp_")
plot_timeseries(
  nssp_processed, nssp_trigger, nssp_timeseries_label, "nssp_",
  default_or(config$primary_outcome, "rsv_pct"), free_y = FALSE
)
plot_coverage_comparison(nssp_extended, nssp_trigger, "nssp_")
plot_regional_barchart(nssp_regional, nssp_fraction_label, "nssp_", plot_width = 6)

# Generate NHSN figures
plot_state_choropleth(
  nhsn_outside, nhsn_fraction_label, "nhsn_", highlight_southeast = FALSE,
  cat_label_margin = -18, x_title_size = 9
)
plot_regional_choropleth(nhsn_outside, nhsn_fraction_label, "nhsn_")
plot_timeseries(
  nhsn_processed, nhsn_trigger, nhsn_timeseries_label, "nhsn_",
  default_or(config$nhsn_primary_outcome, "rsv_ped_0_4"), free_y = TRUE
)
plot_coverage_comparison(nhsn_extended, nhsn_trigger, "nhsn_")
plot_regional_barchart(nhsn_regional, nhsn_fraction_label, "nhsn_", plot_width = 6)

# Generate combined figures
plot_combined_choropleth(nssp_outside, nhsn_outside, nssp_fraction_label, nhsn_fraction_label)
plot_combined_coverage(nssp_extended, nssp_trigger, nhsn_extended, nhsn_trigger)
