import collections
from med_merge.data.registry import build_dataset
for split in ("train", "test"):
    d = build_dataset("ham10000", "/path/to/data/huggingface", split=split)
    labs = [int(x) for x in d._labels]
    print(f"ham10000 {split:<6} n={len(d)} dist={dict(sorted(collections.Counter(labs).items()))}")
b = build_dataset("ham10000_mel", "/path/to/data/huggingface", split="test")
idx = list(range(0, len(b), 7))
pos = sum(int(b[i][1].item()) for i in idx)
print(f"ham10000_mel test: {pos}/{len(idx)} positive in a strided sample")
