"""Tests for scripts/integrate_kaggle_results.py using synthetic fixtures.

Build a miniature Kaggle-output directory (csv + json + tiny torch
checkpoints) and a miniature repo (README + paper with the real anchor
strings) inside tmp_path, then exercise every branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.integrate_kaggle_results import (
    PAPER_ANCHOR_LIMITATION,
    PAPER_TABLE_ROW_ANCHOR,
    README_ANCHOR_CALLOUT,
    README_ANCHOR_PENDING_SECTION,
    README_TABLE_ROW_ANCHOR,
    SUPPORTED_MODELS,
    ModelRow,
    load_rows,
    main,
    patch_paper,
    patch_readme,
)

README_FIXTURE = f"""# OncoSeg

## Results

### Segmentation Performance (MSD Brain Tumor, 96 val subjects)

| Model | Dice TC | Dice WT | Dice ET | Dice Mean | HD95 Mean (mm) | Params |
|-------|---------|---------|---------|-----------|----------------|--------|
| **OncoSeg** | **0.7898** | **0.8529*** | **0.7481** | **0.7969** | **15.35** | **3.7M** |
{README_TABLE_ROW_ANCHOR}

> Trained for 50 epochs on MSD Brain Tumor (388 train / 96 val subjects, embed_dim=24, roi_size=96, Apple Silicon MPS). {README_ANCHOR_CALLOUT}

## Quick Start

Some prose.

{README_ANCHOR_PENDING_SECTION}

These three suites are scripted inside the notebook.

| Suite | What it trains |
|-------|----------------|
| SwinUNETR baseline | MONAI SwinUNETR |
| UNETR baseline | MONAI UNETR |

## Local Installation

More prose.
"""

PAPER_FIXTURE = f"""# Results

## 1. Segmentation accuracy

| Model          | Dice TC    | Dice WT     | Dice ET    | Dice Mean  | HD95 Mean (mm) | Params |
|----------------|------------|-------------|------------|------------|----------------|--------|
| **OncoSeg**    | **0.7898** | **0.8529**\\*| **0.7481** | **0.7969** | **15.35**      | **3.7 M** |
{PAPER_TABLE_ROW_ANCHOR}

## 7. Limitations

1. **Single dataset.** MSD only.
{PAPER_ANCHOR_LIMITATION}
3. **Ablation study.** Pending.
4. **RECIST synthetic.** Pending.
5. **Uncertainty sample count.** Low.
"""


def _write_ckpt(path: Path, n_params: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build a tiny state_dict whose total numel sums to n_params.
    torch.save({"model_state_dict": {"w": torch.zeros(n_params)}}, path)


def _make_input_dir(tmp_path: Path, models=("swin_unetr", "unetr"), *, dice_mean=0.80, params=10_000_000) -> Path:
    """Build a complete synthetic Kaggle output directory."""
    d = tmp_path / "kaggle_v9"
    (d / "checkpoints").mkdir(parents=True)
    csv_lines = [
        "Model,Dice ET,Dice TC,Dice WT,Dice Mean,HD95 ET,HD95 TC,HD95 WT,Best Epoch"
    ]
    eval_json: dict[str, dict] = {}
    for m in models:
        csv_lines.append(f"{m},0.7500,0.7600,0.8200,{dice_mean:.4f},5.10,4.50,3.20,42")
        eval_json[m] = {
            "dice_ET": 0.75,
            "dice_TC": 0.76,
            "dice_WT": 0.82,
            "dice_mean": dice_mean,
            "hd95_ET": 5.10,
            "hd95_TC": 4.50,
            "hd95_WT": 3.20,
            "best_epoch": 42,
        }
        _write_ckpt(d / "checkpoints" / f"{m}_best.pth", params)
    (d / "results.csv").write_text("\n".join(csv_lines) + "\n")
    (d / "evaluation_results.json").write_text(json.dumps(eval_json))
    (d / "experiment_config.json").write_text('{"seed": 42}')
    (d / "swin_unetr_history.json").write_text('{"loss": [1.0]}')
    return d


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "README.md").write_text(README_FIXTURE)
    (repo / "docs" / "Paper_Results_Draft.md").write_text(PAPER_FIXTURE)
    return repo


def test_load_rows_happy_path(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    assert [r.key for r in rows] == list(SUPPORTED_MODELS)
    assert rows[0].dice_mean == pytest.approx(0.80)
    assert rows[0].params_millions == pytest.approx(10.0, rel=1e-3)
    assert rows[0].hd95_mean == pytest.approx((5.10 + 4.50 + 3.20) / 3)


def test_load_rows_missing_csv(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    (input_dir / "results.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_rows(input_dir)


def test_load_rows_missing_json(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    (input_dir / "evaluation_results.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_rows(input_dir)


def test_load_rows_missing_csv_column(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    (input_dir / "results.csv").write_text("Model,Dice ET\nswin_unetr,0.75\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_rows(input_dir)


def test_load_rows_missing_json_key(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    bad = {"swin_unetr": {"dice_ET": 0.75}, "unetr": {"dice_ET": 0.75}}
    (input_dir / "evaluation_results.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing key"):
        load_rows(input_dir)


def test_load_rows_unknown_model(tmp_path):
    input_dir = _make_input_dir(tmp_path, models=("swin_unetr", "fancy_net"))
    with pytest.raises(ValueError, match="unsupported model keys"):
        load_rows(input_dir)


def test_load_rows_csv_json_mismatch(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    data = json.loads((input_dir / "evaluation_results.json").read_text())
    del data["unetr"]
    (input_dir / "evaluation_results.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not match"):
        load_rows(input_dir)


def test_load_rows_missing_checkpoint(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    (input_dir / "checkpoints" / "unetr_best.pth").unlink()
    with pytest.raises(FileNotFoundError, match="unetr_best.pth"):
        load_rows(input_dir)


def test_patch_readme_inserts_rows(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    patched = patch_readme(README_FIXTURE, rows, force=False)
    assert "| Swin UNETR | 0.7600 | 0.8200 | 0.7500 | 0.8000 | 4.27 | 10.0M |" in patched
    assert "| UNETR | 0.7600 | 0.8200 | 0.7500 | 0.8000 | 4.27 | 10.0M |" in patched
    assert README_ANCHOR_CALLOUT not in patched
    assert README_ANCHOR_PENDING_SECTION not in patched
    assert "## Local Installation" in patched


def test_patch_readme_refuses_when_anchors_gone(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    once = patch_readme(README_FIXTURE, rows, force=False)
    # Anchor row is still there (we inserted after it), but callouts are gone.
    with pytest.raises(ValueError, match="already removed"):
        patch_readme(once, rows, force=False)


def test_patch_readme_force_allows_reinsert(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    once = patch_readme(README_FIXTURE, rows, force=False)
    twice = patch_readme(once, rows, force=True)
    assert twice.count("| Swin UNETR |") == 2


def test_patch_readme_refuses_when_table_missing(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    broken = README_FIXTURE.replace(README_TABLE_ROW_ANCHOR, "| UNet3D | edited by hand |")
    with pytest.raises(ValueError, match="table anchor"):
        patch_readme(broken, rows, force=False)


def test_patch_paper_inserts_rows_and_renumbers(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    rows = load_rows(input_dir)
    patched = patch_paper(PAPER_FIXTURE, rows, force=False)
    assert "| Swin UNETR    " in patched
    assert "| UNETR         " in patched
    assert PAPER_ANCHOR_LIMITATION not in patched
    # Renumbered limitations:
    assert "2. **Ablation study.**" in patched
    assert "3. **RECIST synthetic.**" in patched
    assert "4. **Uncertainty sample count.**" in patched
    assert "1. **Single dataset.**" in patched


def test_main_end_to_end(tmp_path, capsys):
    input_dir = _make_input_dir(tmp_path)
    repo = _make_repo(tmp_path)
    rc = main(["--input-dir", str(input_dir), "--repo-root", str(repo)])
    assert rc == 0
    # Artefacts staged:
    staged = repo / "experiments" / "kaggle_run"
    assert (staged / "results.csv").is_file()
    assert (staged / "evaluation_results.json").is_file()
    assert (staged / "checkpoints" / "swin_unetr_best.pth").is_file()
    assert (staged / "checkpoints" / "unetr_best.pth").is_file()
    # Markdown updated:
    readme_after = (repo / "README.md").read_text()
    assert "| Swin UNETR |" in readme_after
    assert README_ANCHOR_PENDING_SECTION not in readme_after
    paper_after = (repo / "docs" / "Paper_Results_Draft.md").read_text()
    assert "| Swin UNETR    " in paper_after


def test_main_dry_run_writes_nothing(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    repo = _make_repo(tmp_path)
    readme_before = (repo / "README.md").read_text()
    paper_before = (repo / "docs" / "Paper_Results_Draft.md").read_text()
    rc = main(["--input-dir", str(input_dir), "--repo-root", str(repo), "--dry-run"])
    assert rc == 0
    assert (repo / "README.md").read_text() == readme_before
    assert (repo / "docs" / "Paper_Results_Draft.md").read_text() == paper_before
    assert not (repo / "experiments" / "kaggle_run").exists()


def test_main_fails_on_bad_input(tmp_path):
    input_dir = _make_input_dir(tmp_path)
    repo = _make_repo(tmp_path)
    (input_dir / "results.csv").unlink()
    rc = main(["--input-dir", str(input_dir), "--repo-root", str(repo)])
    assert rc == 1


def test_narrative_warning_when_baseline_wins(tmp_path, capsys):
    # Baseline Dice Mean 0.85 beats OncoSeg 0.7969 → warning.
    input_dir = _make_input_dir(tmp_path, dice_mean=0.85)
    repo = _make_repo(tmp_path)
    rc = main(["--input-dir", str(input_dir), "--repo-root", str(repo)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "narrative claim" in out


def test_modelrow_formatting():
    r = ModelRow(
        key="swin_unetr",
        dice_et=0.75,
        dice_tc=0.76,
        dice_wt=0.82,
        dice_mean=0.80,
        hd95_et=5.10,
        hd95_tc=4.50,
        hd95_wt=3.20,
        best_epoch=42,
        params_millions=62.2,
    )
    assert r.readme_row() == "| Swin UNETR | 0.7600 | 0.8200 | 0.7500 | 0.8000 | 4.27 | 62.2M |"
    assert "| Swin UNETR    " in r.paper_row()
    assert "62.2 M |" in r.paper_row()
