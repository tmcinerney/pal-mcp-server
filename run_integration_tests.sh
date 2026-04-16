#!/bin/bash

# PAL MCP Server - Run Integration Tests
# This script runs integration tests that require API keys.
#
# Requires: devenv shell with secrets loaded via secretspec / use_op

set -e  # Exit on any error

echo "Running Integration Tests for PAL MCP Server"
echo "=============================================="
echo "These tests use real API calls with your configured keys"
echo ""

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Activate devenv first."
    exit 1
fi

echo "Checking API key availability:"
echo "---------------------------------"

for key in GEMINI_API_KEY OPENAI_API_KEY XAI_API_KEY OPENROUTER_API_KEY CUSTOM_API_URL; do
    if [[ -n "${!key}" ]]; then
        echo "  $key configured"
    else
        echo "  $key not found"
    fi
done

echo ""

# Run integration tests
echo "Running integration tests..."
echo "------------------------------"

pytest tests/ -v -m "integration" --tb=short

echo ""
echo "Integration tests completed!"
echo ""

# Also run simulator tests if requested
if [[ "$1" == "--with-simulator" ]]; then
    echo "Running simulator tests..."
    echo "----------------------------"
    python communication_simulator_test.py --verbose
    echo ""
    echo "Simulator tests completed!"
fi
