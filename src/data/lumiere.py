"""LUMIERE longitudinal glioblastoma MRI dataset loader.

Dataset: Suter et al., "The LUMIERE Dataset: Longitudinal Glioblastoma MRI
with Expert RANO Evaluation" (Scientific Data, 2022). 91 GBM patients, 638
study dates, 2,487 images across four modalities (T1, CT1, T2, FLAIR) with
expert RANO response ratings per timepoint.

The raw Figshare distribution lays each patient out as:

    <root>/
        Patient-01/
            week-000/
                DeepBraTumIA-segmentation/atlas/skull_strip/
                    ct1_skull_strip.nii.gz
                    flair_skull_strip.nii.gz
                    t1_skull_strip.nii.gz
                    t2_skull_strip.nii.gz
                DeepBraTumIA-segmentation/atlas/segmentation/
                    seg_mask.nii.gz                   # NCR/ED/ET reference
                HD-GLIO-AUTO-segmentation/
                    segmentation.nii.gz               # CE / T2-abnormality
            week-013/ ...
            week-024/ ...
        Patient-02/ ...
        LumiereClinicalData.csv                       # RANO ratings

Some timepoints are missing segmentations (only 599 of 638 study dates have
them) or individual modalities. The loader records what is present and
silently skips timepoints that do not have all four modalities — we need
all four to run OncoSeg.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Canonical modality keys used across the project. The sigmoid-era model
# expects inputs in this order: T1 native, T1 contrast, T2 weighted, FLAIR.
ONCOSEG_MODALITIES: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")

# Patterns that match LUMIERE's `*_skull_strip.nii.gz` files, keyed by the
# canonical OncoSeg modality name. Patterns are case-insensitive regex.
LUMIERE_MODALITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "t1n": re.compile(r"(^|[_/])t1(_skull_strip)?\.nii(\.gz)?$", re.IGNORECASE),
    "t1c": re.compile(r"(^|[_/])ct1(_skull_strip)?\.nii(\.gz)?$", re.IGNORECASE),
    "t2w": re.compile(r"(^|[_/])t2(_skull_strip)?\.nii(\.gz)?$", re.IGNORECASE),
    "t2f": re.compile(r"(^|[_/])flair(_skull_strip)?\.nii(\.gz)?$", re.IGNORECASE),
}

# Folders inside each week-XXX directory that hold skull-stripped volumes.
# We search these in priority order and take the first that yields all four
# modalities.
MODALITY_SEARCH_SUBDIRS: tuple[str, ...] = (
    "DeepBraTumIA-segmentation/atlas/skull_strip",
    "DeepBraTumIA-segmentation/atlas",
    ".",
)

# Expert segmentation reference (used only for optional agreement analysis,
# not as training ground truth).
DEEPBRATUMIA_SEG_PATH: str = (
    "DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz"
)
HDGLIO_SEG_PATH: str = "HD-GLIO-AUTO-segmentation/segmentation.nii.gz"


@dataclass
class LumiereTimepoint:
    """One study date for one patient."""

    patient_id: str
    timepoint_id: str
    week: int
    modalities: dict[str, str]
    deepbratumia_seg: str | None = None
    hdglio_seg: str | None = None
    rano: str | None = None
    rano_rationale: str | None = None

    def as_data_dict(self) -> dict[str, str]:
        """Return a MONAI-compatible data dict keyed by modality + subject_id."""
        entry: dict[str, str] = {
            "subject_id": f"{self.patient_id}__{self.timepoint_id}",
            "patient_id": self.patient_id,
            "timepoint_id": self.timepoint_id,
            "week": str(self.week),
        }
        entry.update(self.modalities)
        return entry


@dataclass
class LumierePatient:
    """All timepoints collected for a single LUMIERE patient, time-ordered."""

    patient_id: str
    timepoints: list[LumiereTimepoint] = field(default_factory=list)

    def baseline(self) -> LumiereTimepoint | None:
        return self.timepoints[0] if self.timepoints else None

    def followups(self) -> list[LumiereTimepoint]:
        return self.timepoints[1:]


_WEEK_RE = re.compile(r"week[-_]?(\d+)", re.IGNORECASE)


def _parse_week(dirname: str) -> int | None:
    match = _WEEK_RE.search(dirname)
    return int(match.group(1)) if match else None


def _find_modality(week_dir: Path, modality: str) -> str | None:
    """Locate the best file path for `modality` inside a LUMIERE week directory."""
    pattern = LUMIERE_MODALITY_PATTERNS[modality]
    for subdir in MODALITY_SEARCH_SUBDIRS:
        search_root = week_dir / subdir if subdir != "." else week_dir
        if not search_root.exists():
            continue
        for candidate in sorted(search_root.rglob("*.nii*")):
            if pattern.search(candidate.name):
                return str(candidate)
    return None


def _discover_week(week_dir: Path, patient_id: str) -> LumiereTimepoint | None:
    week = _parse_week(week_dir.name)
    if week is None:
        return None

    modalities: dict[str, str] = {}
    for modality in ONCOSEG_MODALITIES:
        found = _find_modality(week_dir, modality)
        if found is not None:
            modalities[modality] = found

    if len(modalities) < len(ONCOSEG_MODALITIES):
        missing = [m for m in ONCOSEG_MODALITIES if m not in modalities]
        logger.debug(
            "Skipping %s/%s: missing modalities %s", patient_id, week_dir.name, missing
        )
        return None

    deepbratumia = week_dir / DEEPBRATUMIA_SEG_PATH
    hdglio = week_dir / HDGLIO_SEG_PATH

    return LumiereTimepoint(
        patient_id=patient_id,
        timepoint_id=week_dir.name,
        week=week,
        modalities=modalities,
        deepbratumia_seg=str(deepbratumia) if deepbratumia.exists() else None,
        hdglio_seg=str(hdglio) if hdglio.exists() else None,
    )


def _load_rano_table(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Parse LumiereClinicalData.csv into a {(patient_id, week): row} map.

    The LUMIERE CSV does not have a single canonical schema across releases;
    this loader is tolerant of column-name variants. Recognised columns:

        patient_id | Patient | subject     → patient identifier
        week | timepoint | visit_week      → integer week offset
        rano | response | rating           → CR / PR / SD / PD
        rationale | comment | reason       → free-text justification
    """
    if not csv_path.exists():
        logger.warning("LUMIERE clinical CSV not found at %s", csv_path)
        return {}

    patient_cols = ("patient_id", "patient", "subject", "subject_id")
    week_cols = ("week", "timepoint", "visit_week", "weeks")
    rano_cols = ("rano", "response", "rating", "rano_rating")
    rationale_cols = ("rationale", "comment", "reason", "notes")

    def _pick(row: dict[str, str], candidates: tuple[str, ...]) -> str | None:
        for key in row:
            if key is None:
                continue
            if key.strip().lower().replace(" ", "_") in candidates:
                value = row[key]
                if value is not None and str(value).strip():
                    return str(value).strip()
        return None

    table: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            patient = _pick(row, patient_cols)
            week_str = _pick(row, week_cols)
            if patient is None or week_str is None:
                continue
            try:
                week = int(float(week_str))
            except ValueError:
                continue
            rano = _pick(row, rano_cols)
            rationale = _pick(row, rationale_cols)
            table[(patient, week)] = {
                "rano": rano or "",
                "rationale": rationale or "",
            }
    return table


def discover_lumiere(
    root: str | Path,
    clinical_csv: str | Path | None = None,
) -> list[LumierePatient]:
    """Scan a LUMIERE root directory and return per-patient timepoint manifests.

    Args:
        root: Directory containing Patient-XX/ subdirectories.
        clinical_csv: Optional path to LumiereClinicalData.csv. If omitted,
            the loader looks for it inside `root`.

    Returns:
        List of LumierePatient, sorted by patient id. Each patient's
        timepoints are sorted by week ascending.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"LUMIERE root not found: {root}")

    csv_path = Path(clinical_csv) if clinical_csv else root / "LumiereClinicalData.csv"
    rano_table = _load_rano_table(csv_path)

    patients: list[LumierePatient] = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not patient_dir.name.lower().startswith("patient"):
            continue

        timepoints: list[LumiereTimepoint] = []
        for week_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            tp = _discover_week(week_dir, patient_dir.name)
            if tp is None:
                continue
            rano_row = rano_table.get((patient_dir.name, tp.week))
            if rano_row is not None:
                tp.rano = rano_row.get("rano") or None
                tp.rano_rationale = rano_row.get("rationale") or None
            timepoints.append(tp)

        if not timepoints:
            continue

        timepoints.sort(key=lambda t: t.week)
        patients.append(LumierePatient(patient_id=patient_dir.name, timepoints=timepoints))

    logger.info(
        "LUMIERE: discovered %d patients, %d timepoints total",
        len(patients),
        sum(len(p.timepoints) for p in patients),
    )
    return patients


def patients_with_followup(patients: list[LumierePatient], min_timepoints: int = 2) -> list[LumierePatient]:
    """Filter to patients with at least `min_timepoints` timepoints (default 2)."""
    return [p for p in patients if len(p.timepoints) >= min_timepoints]


def flatten_timepoints(patients: list[LumierePatient]) -> list[LumiereTimepoint]:
    """Return all timepoints across all patients, ordered by patient then week."""
    return [tp for p in patients for tp in p.timepoints]
