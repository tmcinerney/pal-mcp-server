#!/bin/bash

# PAL MCP Server - Code Quality Checks
# This script runs all required linting and testing checks before committing changes.
# ALL checks must pass 100% for CI/CD to succeed.
#
# Requires: devenv shell (uv, ruff, black, isort, pytest available via `uv run`)

set -e  # Exit on any error

echo "Running Code Quality Checks for PAL MCP Server"
echo "================================================="

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Activate devenv first (cd into repo with direnv, or run devenv shell)."
    exit 1
fi

# Sync deps if needed
echo "Syncing dependencies..."
uv sync --quiet
echo ""

# Step 1: Linting and Formatting
echo "Step 1: Running Linting and Formatting Checks"
echo "--------------------------------------------------"

echo "Running ruff linting with auto-fix..."
ruff check --fix --exclude test_simulation_files --exclude .devenv

echo "Running black code formatting..."
black . --exclude="test_simulation_files/|\.devenv/"

echo "Running import sorting with isort..."
isort . --skip-glob="test_simulation_files/*" --skip-glob=".devenv/*"

echo "Verifying all linting passes..."
ruff check --exclude test_simulation_files --exclude .devenv

echo "Step 1 Complete: All linting and formatting checks passed!"
echo ""

# Step 2: Unit Tests
echo "Step 2: Running Complete Unit Test Suite"
echo "---------------------------------------------"

echo "Running unit tests (excluding integration tests)..."
pytest tests/ -v -x -m "not integration"

echo "Step 2 Complete: All unit tests passed!"
echo ""

# Step 3: Final Summary
echo "All Code Quality Checks Passed!"
echo "=================================="
echo "Linting (ruff): PASSED"
echo "Formatting (black): PASSED"
echo "Import sorting (isort): PASSED"
echo "Unit tests: PASSED"
