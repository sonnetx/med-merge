import collections
from med_merge.data.registry import build_dataset
from med_merge.data.ham10000 import CLASS_NAMES
MEL = CLASS_NAMES.index("mel")
for split in ("train", "validation", "test"):
    ds = build_dataset("ham10000_mel", "/path/to/data/huggingface", split=split)
    base = ds._base                      # underlying multiclass dataset
    labs = [int(v == MEL) for v in base._labels]   # no image decoding
    c = collections.Counter(labs)
    n = len(labs)
    print(f"{split:<11} n={n:<6} pos={c.get(1,0):<5} ({100*c.get(1,0)/max(n,1):.1f}%)")
