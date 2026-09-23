import collections
from datasets import load_dataset
ds = load_dataset("marmal88/skin_cancer", split="train")
print("columns:", ds.column_names)
for col in ("dx", "label", "diagnosis"):
    if col in ds.column_names:
        vals = ds[col][:3000]
        print(f"{col} distribution (first 3000):", dict(collections.Counter(vals)))
