#!/usr/bin/env bash

echo "Lit testing..."
lit -v test/$1
echo "Lit testing completed!"

echo "Pytest..."
pytest
echo "completed!"
