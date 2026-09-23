"""Build TCGA LUAD/LUSC thumbnails + dataset.csv from OAK SVS files."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Task:
    slide_id: str
    project_id: str
    source_path: Path
    output_path: Path
    size: tuple[int, int]

@dataclass
class Result:
    slide_id: str
    project_id: str
    jpg_path: Optional[str]
    status: str
    error: Optional[str] = None

def _process(task: Task) -> Result:
    if task.output_path.exists():
        return Result(task.slide_id, task.project_id, str(task.output_path), "skipped")
    try:
        import openslide
    except ImportError as e:
        return Result(task.slide_id, task.project_id, None, "failed", f"openslide import: {e}")
    try:
        slide = openslide.OpenSlide(str(task.source_path))
        thumb = slide.get_thumbnail(task.size)
        slide.close()
        if thumb.mode == "RGBA":
            thumb = thumb.convert("RGB")
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(task.output_path, "JPEG", quality=85)
        return Result(task.slide_id, task.project_id, str(task.output_path), "processed")
    except Exception as e:
        return Result(task.slide_id, task.project_id, None, "failed", str(e))

def build_tasks(luad_dir: Path, lusc_dir: Path, out_dir: Path, size: tuple[int, int]) -> list[Task]:
    tasks: list[Task] = []
    for project_id, src_dir in [("TCGA-LUAD", luad_dir), ("TCGA-LUSC", lusc_dir)]:
        if not src_dir.exists():
            logger.warning(f"missing SVS dir: {src_dir}")
            continue
        for svs in sorted(src_dir.glob("*.svs")):
            slide_id = svs.stem
            tasks.append(Task(
                slide_id=slide_id,
                project_id=project_id,
                source_path=svs,
                output_path=out_dir / f"{slide_id}.jpg",
                size=size,
            ))
    return tasks

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--luad-svs-dir", required=True, type=Path)
    ap.add_argument("--lusc-svs-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    thumb_dir = args.out_dir / "thumbnails"
    tables_dir = args.out_dir / "tables"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    size = (args.size, args.size)
    n_workers = args.workers or max(1, cpu_count() - 1)

    tasks = build_tasks(args.luad_svs_dir, args.lusc_svs_dir, thumb_dir, size)
    logger.info(f"queued {len(tasks)} slides over {n_workers} workers (size={size})")
    if not tasks:
        logger.error("no SVS files found")
        return 1

    with Pool(processes=n_workers) as pool:
        results = pool.map(_process, tasks)

    processed = sum(1 for r in results if r.status == "processed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info(f"done: processed={processed} skipped={skipped} failed={failed}")
    for r in results:
        if r.status == "failed":
            logger.error(f"  {r.slide_id}: {r.error}")

    ok = [r for r in results if r.status in ("processed", "skipped")]
    df = pd.DataFrame([
        {"slide_id": r.slide_id, "project_id": r.project_id, "jpg_path": r.jpg_path}
        for r in ok
    ])
    csv_path = tables_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path} with {len(df)} rows (LUAD={sum(df['project_id'] == 'TCGA-LUAD')}, LUSC={sum(df['project_id'] == 'TCGA-LUSC')})")
    return 0 if failed == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
