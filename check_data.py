import pandas as pd

data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the columns we need
data = data[["v1", "v2"]]

# Rename columns
data.columns = ["label", "message"]

print("Dataset loaded successfully!")
print("Number of rows:", len(data))
print("Columns:", data.columns.tolist())
print(data.head())

print("\nMessage counts:")
print(data["label"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())