# =========================
# Configuration
# =========================

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

PACKAGE := experimentalis

DOCS_DIR := docs
DOCS_SOURCE := $(DOCS_DIR)/source
DOCS_BUILD := $(DOCS_DIR)/build

GH_PAGES_BRANCH := gh-pages
SOURCE_BRANCH := main

# =========================
# Phony targets
# =========================

.PHONY: help setup kernel deps dev install test docs docs-clean docs-view lint clean all

# =========================
# Help
# =========================

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make setup         Create venv, install deps, dev tools, and editable package"
	@echo "  make kernel        Install Jupyter Kernel"
	@echo "  make deps          Install runtime dependencies"
	@echo "  make dev           Install package in editable/dev mode"
	@echo "  make install       Install package normally"
	@echo "  make test          Run pytest test suite"
	@echo "  make docs          Build Sphinx HTML documentation"
	@echo "  make docs-clean    Remove Sphinx build artifacts"
	@echo "  make docs-view     Build docs and open in default browser"
	@echo "  make lint          Run linters (flake8 + black)"
	@echo "  make clean         Remove all build artifacts"
	@echo "  make all           Run tests and build docs"
	@echo ""

# =========================
# Environment setup
# =========================

setup:
	@echo "Setting up development environment..."
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(MAKE) deps
	$(MAKE) dev

kernel:
	$(PYTHON) -m ipykernel install --user --name=kernel_experimentalis --display-name="Python3 (Experimentalis)"

deps:
	@echo "Installing runtime dependencies..."
	$(PIP) install -r requirements.txt

dev:
	@echo "Installing package in editable (development) mode..."
	$(PIP) install -e .
	@echo "Installing development dependencies..."
	@test -f dev-requirements.txt && $(PIP) install -r dev-requirements.txt || echo "No dev-requirements.txt found"

install:
	@echo "Installing package..."
	$(PIP) install .

# =========================
# Testing
# =========================

test:
	@echo "Running pytest..."
	$(PYTHON) -m pytest test

# =========================
# Documentation
# =========================

docs:
	@echo "Building Sphinx documentation..."
	$(PYTHON) -m sphinx -b html $(DOCS_SOURCE) $(DOCS_BUILD)/html

docs-clean:
	@echo "Cleaning Sphinx build directory..."
	rm -rf $(DOCS_BUILD)

docs-view: docs
	@echo "Opening documentation in default browser..."
	@$(PYTHON) -c 'import os, webbrowser; webbrowser.open("file://" + os.path.abspath("docs/build/html/index.html"))'

# =========================
# Linting
# =========================

lint:
	@echo "Running linters..."
	$(PYTHON) -m flake8 $(PACKAGE)
	$(PYTHON) -m black --check $(PACKAGE)

# =========================
# Cleanup
# =========================

clean: docs-clean
	@echo "Cleaning Python build artifacts..."
	rm -rf \
		__pycache__ \
		.pytest_cache \
		*.egg-info \
		build \
		dist

# =========================
# Combined
# =========================

all: test docs
