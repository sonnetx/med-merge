#!/bin/bash
# Relocate specialist checkpoints from the 1TB group home to group scratch, leaving a
# symlink behind so compute_oracle_router.py still resolves them.
#
# Skips *_abl_* cells, which have jobs actively writing to them, and skips anything that
# is already a symlink. Verifies each copy before removing the original.
set -uo pipefail
PD=.
DEST=.
cd "$PD" || exit 1

moved=0; skipped=0; failed=0; bytes=0
while IFS= read -r f; do
  [ -L "$f" ] && { skipped=$((skipped+1)); continue; }
  [ -f "$f" ] || { skipped=$((skipped+1)); continue; }
  target="$DEST/$f"
  mkdir -p "$(dirname "$target")" || { failed=$((failed+1)); continue; }
  src_size=$(stat -c%s "$f")
  if cp -p "$f" "$target"; then
    dst_size=$(stat -c%s "$target" 2>/dev/null || echo 0)
    if [ "$src_size" = "$dst_size" ]; then
      rm -f "$f" && ln -s "$target" "$f"
      moved=$((moved+1)); bytes=$((bytes+src_size))
    else
      echo "SIZE MISMATCH, keeping original: $f ($src_size vs $dst_size)"
      rm -f "$target"; failed=$((failed+1))
    fi
  else
    echo "COPY FAILED: $f"; failed=$((failed+1))
  fi
done < <(find outputs -name best_model.pt -not -path "*_abl_*" -type f)

echo "moved=$moved skipped=$skipped failed=$failed freed=$((bytes/1000000000))GB"
