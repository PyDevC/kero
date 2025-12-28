# Kero-Sine

GPU Accelerated Tensor Database Management System (GATDBMS)

## Installation

> Requires torch>=2.5.0

Pip install from source

```bash
python3 -m build
pip install dist/kero*.whl
```

## Build the docs

All the docs building requirements are in `docs/` directory.

```bash
pip install -r requirements.txt
make html
```

Basic example usage is mentioned in example.py

## Clean after build

```bash
rm -rf dist/
rm -rf *.egg-info/
```
