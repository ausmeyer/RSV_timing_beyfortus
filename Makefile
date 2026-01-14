# RSV Timing / Beyfortus Analysis Pipeline
# Usage: make all

.PHONY: all clean data analysis figures help

# Default target: run everything
all: analysis
	@echo "Pipeline complete. Results in results/"

# Pull data from Socrata (with caching)
data:
	@echo "Fetching NSSP data from CDC Socrata..."
	python -m src.pull_nssp
	@echo "Data pull complete."

# Run full analysis pipeline (includes data pull if needed)
analysis: data
	@echo "Running analysis pipeline..."
	python -m src.run_pipeline
	@echo "Analysis complete."

# Generate figures only (requires analysis to be run first)
figures:
	@echo "Generating figures..."
	python -c "from src.run_pipeline import generate_figures_only; generate_figures_only()"
	@echo "Figures saved to results/figures/"

# Clean all generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/raw/*.parquet
	rm -rf data/raw/*.json
	rm -rf data/processed/*.parquet
	rm -rf results/figures/*
	rm -rf results/tables/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

# Show help
help:
	@echo "RSV Timing / Beyfortus Analysis Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make all       - Run complete pipeline (data + analysis + figures)"
	@echo "  make data      - Pull data from CDC Socrata only"
	@echo "  make analysis  - Run analysis (pulls data if needed)"
	@echo "  make figures   - Regenerate figures only"
	@echo "  make clean     - Remove all generated files"
	@echo "  make help      - Show this help message"
