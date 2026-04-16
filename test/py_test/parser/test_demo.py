import torch
import torch.nn as nn
import pyarrow as pa
import numpy as np
from kero.data import KeroDataLoader

np.random.seed(42)

num_rows = 100_000

data = {
    "age": np.random.randint(18, 80, num_rows).tolist(),
    "salary": np.random.randint(20000, 200000, num_rows).tolist(),
    "has_car": np.random.choice([True, False], num_rows).tolist(),
    "has_house": np.random.choice([True, False], num_rows).tolist(),
    "credit_score": np.random.randint(300, 850, num_rows).tolist(),
    "num_dependents": np.random.randint(0, 6, num_rows).tolist(),
    "target": (np.random.rand(num_rows) > 0.6).astype(int).tolist(),
}

table = pa.table(data)

print(f"Created table with {len(table)} rows")
print(f"Columns: {table.schema.names}")
print(f"Target distribution: {sum(data['target'])} positives, {num_rows - sum(data['target'])} negatives")

loader = KeroDataLoader(
    table,
    "SELECT age, salary, has_car, has_house, credit_score, num_dependents, target FROM _data WHERE salary > 50000",
    batch_size=256,
)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

try:
    for epoch in range(10):
        total_loss = 0.0
        batch_count = 0
        for batch in loader:
            features = batch.table.to_pydict()
            x = torch.tensor(list(zip(
                features["age"], features["salary"],
                features["has_car"], features["has_house"],
                features["credit_score"], features["num_dependents"]
            )), dtype=torch.float32)
            y = torch.tensor(features["target"], dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
except RuntimeError as e:
    print(f"Expected (C++ extension not built): {e}")
