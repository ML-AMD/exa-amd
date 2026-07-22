#!/usr/bin/env bash

if ! command -v pytest >/dev/null 2>&1; then
    echo "Error: 'pytest' is not installed. Install it with: pip install pytest"
    echo "You also probably do not have pytest-cov, install with pip install pytest-cov"
    exit 1
fi


if ! pytest -VV 2>/dev/null | grep -q "pytest-cov"; then
    echo "Error: 'pytest-cov' is not installed. Install it with: pip install pytest-cov"
    exit 1
fi

pytest --cov=tools --cov=parsl_tasks --cov=ml_models --cov-report=html tests
