"""demo_ml_training.py — GPU-accelerated SQL filtering as a PyTorch DataLoader.

Shows how KeroDataLoader can sit in front of a PyTorch training loop,
replacing a hand-written pandas/numpy pre-processing step with a
compiled MLIR query that pushes the filter/projection down to the
C++ execution engine.

Dataset description
~~~~~~~~~~~~~~~~~~~
Synthetic credit-risk dataset with 200 000 rows.  The query filters to
applicants with salary > 50 000 AND credit_score > 500, then projects the
six feature columns needed by the model.

The training loop runs for 5 epochs.  When the C++ extension is not
built the loader falls back to plain Arrow table slicing (no filtering
is applied at query time — a note is printed).

Run:
    python demo_ml_training.py
"""

from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn

from kero.data import KeroDataLoader
from kero.engine.compiler import _HAVE_KERO

# ── dataset ─────────────────────────────────────────────────────────────────

NUM_ROWS = 200_000
BATCH_SIZE = 512
EPOCHS = 5
FEATURE_COLS = ["age", "salary", "has_car", "has_house", "credit_score", "num_dependents"]
TARGET_COL = "target"

SQL = (
    "SELECT age, salary, has_car, has_house, credit_score, num_dependents, target "
    "FROM _data "
    "WHERE salary > 50000"
)


def make_dataset(n: int) -> pa.Table:
    rng = np.random.default_rng(42)
    return pa.table({
        "age":            pa.array(rng.integers(18, 80, n).tolist(),          type=pa.int32()),
        "salary":         pa.array(rng.integers(20_000, 200_000, n).tolist(), type=pa.float64()),
        "has_car":        pa.array(rng.integers(0, 2, n).tolist(),            type=pa.int32()),
        "has_house":      pa.array(rng.integers(0, 2, n).tolist(),            type=pa.int32()),
        "credit_score":   pa.array(rng.integers(300, 850, n).tolist(),        type=pa.float64()),
        "num_dependents": pa.array(rng.integers(0, 6, n).tolist(),            type=pa.int32()),
        "target":         pa.array((rng.random(n) > 0.6).astype(np.int32).tolist(), type=pa.int32()),
    })


# ── model ────────────────────────────────────────────────────────────────────

class CreditRiskModel(nn.Module):
    """Small 3-layer MLP for binary classification."""

    def __init__(self, in_features: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── batch → tensors ─────────────────────────────────────────────────────────

def batch_to_tensors(batch_table: pa.Table):
    """Convert a pa.Table batch to (X, y) float32 tensors."""
    d = batch_table.to_pydict()
    X = torch.tensor(
        list(zip(*(d[c] for c in FEATURE_COLS))),
        dtype=torch.float32,
    )
    y = torch.tensor(d[TARGET_COL], dtype=torch.float32).unsqueeze(1)
    return X, y


# ── training loop ────────────────────────────────────────────────────────────

def train() -> None:
    print("Kero ML Training Demo")
    print(f"  Rows        : {NUM_ROWS:,}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  _kero ext   : {_HAVE_KERO}")
    print(f"  SQL         : {SQL}\n")

    table = make_dataset(NUM_ROWS)
    print(f"Dataset created — {len(table):,} rows, columns: {table.schema.names}")

    loader = KeroDataLoader(
        table,
        SQL,
        batch_size=BATCH_SIZE,
        compile_once=True,
    )
    print(f"KeroDataLoader: {loader.num_batches} batches/epoch\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CreditRiskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    compile_error_shown = False

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0

        try:
            for batch_result in loader:
                X, y = batch_to_tensors(batch_result.table)
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        except RuntimeError as exc:
            if not compile_error_shown:
                print(f"  NOTE: compile step unavailable ({exc})")
                print("  Falling back to raw Arrow batching (no SQL filter applied).\n")
                compile_error_shown = True
            # Manual batching fallback — demonstrates the loader shape
            n = len(table)
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = table.slice(start, end - start)
                X, y = batch_to_tensors(batch)
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"loss={avg_loss:.5f}  "
            f"batches={n_batches}  "
            f"time={elapsed:.2f}s"
        )

    print("\nTraining complete.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")


if __name__ == "__main__":
    train()
