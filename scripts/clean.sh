#!/usr/bin/env bash

echo "Removing all builds..."

rm -rf build
rm -rf dist
rm -rf kero.egg-info
rm -rf setup_build

if [[ $1 == "deep" ]]; then
    rm -rf .cache
    rm -rf .pytest_cache
fi
