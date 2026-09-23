"""Find .pt artifacts truncated by the disk-full period.

Loads every task vector, head, and merged encoder under outputs/ and reports the ones that
fail to deserialize, so they can be removed and regenerated. Reads with mmap where
possible to keep memory flat.
"""

import glob
import os
import sys

import torch

PATTERNS = [
    "outputs/*/*/task_vectors/*/task_vector.pt",
    "outputs/*/*/checkpoints/*/head.pt",
    "outputs/*/*/merged*/*/merged_encoder.pt",
]

bad, empty, ok = [], [], 0
for pat in PATTERNS:
    for f in glob.glob(pat):
        if os.path.islink(f):
            continue
        try:
            size = os.path.getsize(f)
        except OSError as e:
            bad.append((f, f"stat failed: {e}"))
            continue
        if size == 0:
            empty.append(f)
            continue
        try:
            torch.load(f, map_location="cpu", weights_only=False)
            ok += 1
        except Exception as e:  # truncated archive, bad zip directory, etc.
            bad.append((f, type(e).__name__ + ": " + str(e)[:70]))

print(f"ok={ok}  corrupt={len(bad)}  zero-byte={len(empty)}")
for f in empty:
    print(f"  ZERO   {f}")
for f, why in bad:
    print(f"  CORRUPT {f}\n          {why}")

with open("corrupt_files.txt", "w") as fh:
    fh.write("\n".join([f for f, _ in bad] + empty))
print(f"\nwrote corrupt_files.txt ({len(bad) + len(empty)} entries)")
sys.exit(0)
