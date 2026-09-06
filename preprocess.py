import pandas as pd

data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[["v1", "v2"]]
data.columns = ["label", "message"]

data["message"] = data["message"].str.lower()
data["message"] = data["message"].str.replace(r"[^\w\s]", "", regex=True)
data["message"] = data["message"].str.replace(r"\s+", " ", regex=True).str.strip()
data["label"] = data["label"].map({"ham": 0, "spam": 1})
print("Before preprocessing:")
print(data.head())
print("Final dataset size:", len(data))