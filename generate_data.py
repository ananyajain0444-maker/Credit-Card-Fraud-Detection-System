import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

n = 10000

data = {
    "Time": np.random.randint(0, 100000, n),
    "Amount": np.random.uniform(1, 1000, n),
    "Class": np.random.choice([0, 1], n, p=[0.98, 0.02])
}

for i in range(1, 29):
    data[f"V{i}"] = np.random.normal(0, 1, n)

pd.DataFrame(data).to_csv("data/creditcard.csv", index=False)

print("Dataset created successfully")