import torch
import torch.nn as nn
import pyarrow as pa
from kero.data import KeroDataLoader

table = pa.table({
    "age": [25, 30, 35, 40, 45],
    "salary": [5000, 15000, 20000, 8000, 25000],
    "has_car": [True, False, True, False, True],
    "has_house": [False, True, True, False, True],
    "target": [0, 1, 1, 0, 1],
})

loader = KeroDataLoader(
    table,
    "SELECT age, salary, has_car, has_house, target FROM _data WHERE salary > 10000",
    batch_size=2,
)

class MortalityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

model = MortalityLSTM()
optimizer = torch.optim.Adam(model.parameters())

try:
    for epoch in range(5):
        for batch in loader:
            features = batch.table.to_pydict()
            x = torch.tensor(list(zip(
                features["age"], features["salary"], 
                features["has_car"], features["has_house"], features["target"]
            )), dtype=torch.float32)
            optimizer.zero_grad()
            loss = nn.BCELoss()(model(x), x[:, -1:])
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")
except RuntimeError as e:
    print(f"Expected (C++ extension not built): {e}")
