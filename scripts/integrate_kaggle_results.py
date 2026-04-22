"""Wire Kaggle v9 benchmark artefacts into the repo.

Consumes `results.csv`, `evaluation_results.json`, `*_history.json`, and
`checkpoints/*_best.pth` produced by `notebooks/OncoSeg_Full_Pipeline.ipynb`
(Kaggle run) and does three things:

  1. Stages every artefact under `experiments/kaggle_run/` so the paper
     pipeline can reach them at fixed paths.
  2. Inserts new SwinUNETR + UNETR rows into the segmentation-performance
     tables in `README.md` and `docs/Paper_Results_Draft.md`.
  3. Deletes the "pending T4 run" callouts that are now obsolete.

Usage:
    python scripts/integrate_kaggle_results.py --input-dir ~/Downloads/kaggle_v9
    python scripts/integrate_kaggle_results.py --input-dir DIR --dry-run
    python scripts/integrate_kaggle_results.py --input-dir DIR --force
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_MODELS = ("swin_unetr", "unetr")
DISPLAY_NAMES = {"swin_unetr": "Swin UNETR", "unetr": "UNETR"}

REQUIRED_CSV_COLUMNS = (
    "Model",
    "Dice ET",
    "Dice TC",
    "Dice WT",
    "Dice Mean",
    "HD95 ET",
    "HD95 TC",
    "HD95 WT",
    "Best Epoch",
)

REQUIRED_JSON_KEYS = (
    "dice_ET",
    "dice_TC",
    "dice_WT",
    "dice_mean",
    "hd95_ET",
    "hd95_TC",
    "hd95_WT",
    "best_epoch",
)

README_ANCHOR_CALLOUT = (
    "SwinUNETR and UNETR benchmarks require a CUDA GPU — use the Colab "
    "notebook for full benchmarking."
)
README_ANCHOR_PENDING_SECTION = "### Benchmarks that require a T4 GPU (not yet filled in)"
README_TABLE_ROW_ANCHOR = "| UNet3D | 0.7849 | 0.8522 | 0.7462 | 0.7944 | 21.03 | 19.2M |"

PAPER_ANCHOR_LIMITATION = (
    "2. **Two-model comparison.** SwinUNETR and UNETR baselines require a "
    "CUDA GPU and are pending; the comparison table will be extended once "
    "those runs are complete."
)
PAPER_TABLE_ROW_ANCHOR = (
    "| UNet3D         | 0.7849     | 0.8522      | 0.7462     | 0.7944     "
    "| 21.03          | 19.2 M |"
)


@dataclass
class ModelRow:
    key: str
    dice_et: float
    dice_tc: float
    dice_wt: float
    dice_mean: float
    hd95_et: float
    hd95_tc: float
    hd95_wt: float
    best_epoch: int
    params_millions: float

    @property
    def hd95_mean(self) -> float:
        return (self.hd95_et + self.hd95_tc + self.hd95_wt) / 3.0

    @property
    def display(self) -> str:
        return DISPLAY_NAMES[self.key]

    def readme_row(self) -> str:
        return (
            f"| {self.display} | {self.dice_tc:.4f} | {self.dice_wt:.4f} | "
            f"{self.dice_et:.4f} | {self.dice_mean:.4f} | {self.hd95_mean:.2f} "
            f"| {self.params_millions:.1f}M |"
        )

    def paper_row(self) -> str:
        # Paper table uses padded columns and "X.X M" with a space.
        return (
            f"| {self.display:<14} | {self.dice_tc:.4f}     "
            f"| {self.dice_wt:.4f}      | {self.dice_et:.4f}     "
            f"| {self.dice_mean:.4f}     | {self.hd95_mean:<14.2f} "
            f"| {self.params_millions:.1f} M |"
        )


def count_checkpoint_params(ckpt_path: Path) -> int:
    import torch  # torch is a repo dep; lazy-import keeps tests fast.

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    return sum(int(t.numel()) for t in state.values() if torch.is_tensor(t))


def load_rows(input_dir: Path) -> list[ModelRow]:
    csv_path = input_dir / "results.csv"
    json_path = input_dir / "evaluation_results.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"missing {json_path}")

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing required columns: {missing}")
        csv_rows = list(reader)

    eval_json = json.loads(json_path.read_text())
    csv_models = {r["Model"] for r in csv_rows}
    json_models = set(eval_json.keys())
    if csv_models != json_models:
        raise ValueError(
            f"results.csv models {csv_models} do not match evaluation_results.json "
            f"{json_models}"
        )
    unknown = csv_models - set(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f"unsupported model keys: {unknown}. Expected subset of {SUPPORTED_MODELS}")

    rows: list[ModelRow] = []
    for csv_row in csv_rows:
        key = csv_row["Model"]
        j = eval_json[key]
        for required in REQUIRED_JSON_KEYS:
            if required not in j:
                raise ValueError(f"evaluation_results.json[{key}]: missing key {required!r}")

        ckpt = input_dir / "checkpoints" / f"{key}_best.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
        params = count_checkpoint_params(ckpt) / 1e6

        rows.append(
            ModelRow(
                key=key,
                dice_et=float(j["dice_ET"]),
                dice_tc=float(j["dice_TC"]),
                dice_wt=float(j["dice_WT"]),
                dice_mean=float(j["dice_mean"]),
                hd95_et=float(j["hd95_ET"]),
                hd95_tc=float(j["hd95_TC"]),
                hd95_wt=float(j["hd95_WT"]),
                best_epoch=int(j["best_epoch"]),
                params_millions=params,
            )
        )
    # Stable order: swin_unetr before unetr (architectural family order).
    rows.sort(key=lambda r: SUPPORTED_MODELS.index(r.key))
    return rows


def stage_artefacts(input_dir: Path, dest: Path, dry_run: bool) -> list[Path]:
    """Copy every artefact into experiments/kaggle_run/. Returns copied paths."""
    ckpt_dest = dest / "checkpoints"
    plan: list[tuple[Path, Path]] = []
    for name in ("results.csv", "evaluation_results.json", "experiment_config.json"):
        src = input_dir / name
        if src.is_file():
            plan.append((src, dest / name))
    for src in input_dir.glob("*_history.json"):
        plan.append((src, dest / src.name))
    for src in (input_dir / "checkpoints").glob("*_best.pth"):
        plan.append((src, ckpt_dest / src.name))

    copied: list[Path] = []
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        ckpt_dest.mkdir(parents=True, exist_ok=True)
        for src, tgt in plan:
            shutil.copy2(src, tgt)
            copied.append(tgt)
    else:
        copied = [tgt for _, tgt in plan]
    return copied


def patch_readme(text: str, rows: list[ModelRow], force: bool) -> str:
    if README_TABLE_ROW_ANCHOR not in text:
        raise ValueError(
            "README table anchor row not found — either the table was edited "
            "by hand or the integration already ran."
        )
    has_callout = README_ANCHOR_CALLOUT in text
    has_pending = README_ANCHOR_PENDING_SECTION in text
    if not (has_callout or has_pending) and not force:
        raise ValueError(
            "README callouts already removed (integration appears to have run). "
            "Re-run with --force if you really want to reinsert rows."
        )

    new_rows = "\n".join(r.readme_row() for r in rows)
    text = text.replace(
        README_TABLE_ROW_ANCHOR,
        README_TABLE_ROW_ANCHOR + "\n" + new_rows,
        1,
    )

    if has_callout:
        text = _strip_callout_sentence(text, README_ANCHOR_CALLOUT)

    if has_pending:
        text = _drop_pending_section(text)
    return text


def _strip_callout_sentence(text: str, sentence: str) -> str:
    # The callout is the tail of a single-line blockquote. Drop the whole
    # sentence (plus the leading space/comma/dash if present).
    # The source line reads: "...embed_dim=24, roi_size=96, Apple Silicon MPS). {sentence}"
    needle_with_space = " " + sentence
    if needle_with_space in text:
        return text.replace(needle_with_space, "", 1)
    return text.replace(sentence, "", 1)


def _drop_pending_section(text: str) -> str:
    """Delete the h3 'Benchmarks that require a T4 GPU' block up to the next h2."""
    start = text.find(README_ANCHOR_PENDING_SECTION)
    if start == -1:
        return text
    # Find the next h2 heading after this block.
    rest = text[start:]
    next_h2 = rest.find("\n## ")
    if next_h2 == -1:
        raise ValueError("could not find trailing h2 after pending-benchmarks section")
    # Preserve the "\n## Local Installation" heading; strip up to (but not
    # including) its leading newline so we don't leave a double blank.
    cut = start + next_h2
    # Trim trailing blank line left behind by the removal.
    head = text[:start].rstrip("\n") + "\n\n"
    return head + text[cut + 1 :]


def patch_paper(text: str, rows: list[ModelRow], force: bool) -> str:
    if PAPER_TABLE_ROW_ANCHOR not in text:
        raise ValueError("paper table anchor row not found")
    has_limitation = PAPER_ANCHOR_LIMITATION in text
    if not has_limitation and not force:
        raise ValueError(
            "paper Limitation #2 already removed (integration appears to have run)."
        )

    new_rows = "\n".join(r.paper_row() for r in rows)
    text = text.replace(
        PAPER_TABLE_ROW_ANCHOR,
        PAPER_TABLE_ROW_ANCHOR + "\n" + new_rows,
        1,
    )
    if has_limitation:
        text = _drop_and_renumber_limitation(text)
    return text


def _drop_and_renumber_limitation(text: str) -> str:
    """Remove Limitation #2 and renumber 3-5 → 2-4."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("2. **Two-model comparison.**"):
            # Skip this limitation line (and any continuation up to the next
            # numbered line — Limitation #2 is single-line in the source).
            i += 1
            continue
        # Renumber remaining limitations.
        for old, new in (("3. **", "2. **"), ("4. **", "3. **"), ("5. **", "4. **")):
            if lines[i].startswith(old):
                lines[i] = lines[i].replace(old, new, 1)
                break
        out.append(lines[i])
        i += 1
    return "".join(out)


def check_narrative_still_holds(rows: list[ModelRow], oncoseg_dice_mean: float = 0.7969) -> list[str]:
    """Return warnings if OncoSeg's Dice Mean no longer beats all new baselines."""
    warnings: list[str] = []
    for r in rows:
        if r.dice_mean >= oncoseg_dice_mean:
            warnings.append(
                f"{r.display} Dice Mean {r.dice_mean:.4f} ≥ OncoSeg {oncoseg_dice_mean:.4f} — "
                "narrative claim 'OncoSeg outperforms' in README/paper needs manual revision."
            )
    return warnings


def run(input_dir: Path, repo_root: Path, dry_run: bool, force: bool) -> int:
    rows = load_rows(input_dir)
    print(f"Loaded {len(rows)} model(s): {[r.key for r in rows]}")
    for r in rows:
        print(
            f"  {r.display}: Dice Mean {r.dice_mean:.4f}, HD95 Mean {r.hd95_mean:.2f}mm, "
            f"{r.params_millions:.1f}M params, best epoch {r.best_epoch}"
        )

    readme_path = repo_root / "README.md"
    paper_path = repo_root / "docs" / "Paper_Results_Draft.md"
    readme_before = readme_path.read_text()
    paper_before = paper_path.read_text()

    readme_after = patch_readme(readme_before, rows, force=force)
    paper_after = patch_paper(paper_before, rows, force=force)

    staged = stage_artefacts(input_dir, repo_root / "experiments" / "kaggle_run", dry_run=dry_run)

    if dry_run:
        print("\n[dry-run] would stage:")
        for p in staged:
            print(f"  {p}")
        print(f"\n[dry-run] README diff: {_count_changed_lines(readme_before, readme_after)} lines changed")
        print(f"[dry-run] paper  diff: {_count_changed_lines(paper_before, paper_after)} lines changed")
    else:
        readme_path.write_text(readme_after)
        paper_path.write_text(paper_after)
        print(f"\nWrote {readme_path}")
        print(f"Wrote {paper_path}")
        print(f"Staged {len(staged)} artefact file(s) under experiments/kaggle_run/")

    for w in check_narrative_still_holds(rows):
        print(f"WARN: {w}")
    return 0


def _count_changed_lines(a: str, b: str) -> int:
    import difflib

    return sum(1 for line in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0) if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Directory holding Kaggle artefacts")
    p.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Allow re-integration even if anchors already removed")
    args = p.parse_args(argv)
    try:
        return run(args.input_dir.expanduser().resolve(), args.repo_root.resolve(), args.dry_run, args.force)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
