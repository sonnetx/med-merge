#!/bin/bash
#SBATCH --job-name=nih_download
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=nih_download_%j.out
#SBATCH --error=nih_download_%j.err

set -euo pipefail

# Where the data lives. Change this if you'd rather use $SCRATCH.
DEST="$GROUP_SCRATCH/datasets/nih-cxr14"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading to: $DEST"
echo "Started: $(date)"

URLS=(
  https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz
  https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz
  https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz
  https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz
  https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz
  https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz
  https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz
  https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz
  https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz
  https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz
  https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz
  https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz
)

# Download archives in parallel (4 at a time). -c resumes partial files on retry.
echo "Downloading 12 archives..."
for i in "${!URLS[@]}"; do
  idx=$(printf "%02d" $((i+1)))
  echo "${URLS[$i]} -> images_${idx}.tar.gz"
done | xargs -n2 -P4 wget -c -O

# Labels CSV
echo "Downloading Data_Entry_2017.csv..."
wget -c -O Data_Entry_2017.csv \
  https://nihcc.box.com/shared/static/7jw5l9wq22iidfkdvfxfwg7r2u26odra.csv

# Extract everything into ./images/
echo "Extracting archives..."
mkdir -p images
for f in images_*.tar.gz; do
  echo "  $f"
  tar -xzf "$f" -C images --strip-components=1
done

# Sanity check
n_images=$(find images -type f -name "*.png" | wc -l)
echo "Extracted $n_images images (expected ~112120)"

# Optional: free disk by removing archives once extraction succeeds
echo "Removing archives..."
rm -f images_*.tar.gz

echo "Finished: $(date)"

