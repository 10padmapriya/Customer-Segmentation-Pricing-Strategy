# =============================================================================
# Makefile — Customer Segmentation + Pricing Strategy
# =============================================================================
# make install        Install all dependencies
# make run            Run the full pipeline (downloads data first time)
# make run-fast       Run without plots — quick sanity check
# make test           Run all unit tests
# make test-v         Run tests with verbose output
# make test-cov       Run tests with coverage report
# make lint           Check code with flake8
# make clean          Remove outputs and cache files
# make help           Show this message

.PHONY: all install run run-fast test test-v test-cov lint clean help

PYTHON := python
PIP    := pip
PYTEST := pytest

# ── Setup ─────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt
	@echo "✓  All dependencies installed."

# ── Pipeline ──────────────────────────────────────────────────

run:
	$(PYTHON) main.py

run-fast:
	$(PYTHON) main.py --skip-download --no-plots

run-no-plots:
	$(PYTHON) main.py --skip-download --no-plots

# ── Tests ─────────────────────────────────────────────────────

test:
	$(PYTEST) tests/ -q

test-v:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov=data --cov-report=term-missing -q

# ── Quality ───────────────────────────────────────────────────

lint:
	flake8 src/ data/ tests/ main.py \
	    --max-line-length=100 \
	    --ignore=E501,W503,E203

# ── Clean ─────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__    -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"        -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	rm -f outputs/*.png outputs/*.pdf
	@echo "✓  Cleaned."

# ── Help ──────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Customer Segmentation + Pricing Strategy"
	@echo "  ----------------------------------------"
	@echo "  make install     Install dependencies"
	@echo "  make run         Full pipeline (downloads data)"
	@echo "  make run-fast    Quick run, no plots"
	@echo "  make test        Run unit tests"
	@echo "  make test-v      Verbose tests"
	@echo "  make test-cov    Tests + coverage"
	@echo "  make lint        Flake8 lint check"
	@echo "  make clean       Remove outputs + cache"
	@echo ""
