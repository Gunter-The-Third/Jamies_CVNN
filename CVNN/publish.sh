#!/bin/bash
# Usage: ./publish.sh
# This script builds and uploads the current package to PyPI using environment variables for authentication.

set -e

echo "Cleaning old build artifacts..."
rm -rf dist/ build/ CVNN_Jamie.egg-info/

echo "Building package..."
python setup.py sdist bdist_wheel

echo "Uploading to PyPI..."
twine upload dist/*

echo "Done."
