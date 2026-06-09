# OncoSeg — Deployment & Translation Roadmap

How a research prototype becomes something clinically usable, and where OncoSeg honestly sits.
Use this for the "from prototype to clinic" interview question.

## The three translational axes
1. **Technical deployment** — containerize, stable API, monitoring, model versioning, input
   validation + fail-safes for missing series / unusual protocols, data-drift monitoring.
   *Done so far:* FastAPI + Docker inference service, DICOM input endpoint, ~160 tests.
2. **Workflow / PACS integration** — DICOM in via standard interfaces; push results back as
   **DICOM SEG** (overlays) and **DICOM SR** (measurements); radiologist accepts / edits / rejects
   AI contours **inside their existing viewer** (don't make them switch tools).
3. **Validation & regulation** — retrospective multi-site/external validation **first**, then a
   prospective study (time saved, inter-reader variability, error modes, agreement with reference
   standard, generalization across scanners/protocols). Treat as **SaMD**: intended use, risk
   analysis, QMS, FDA 510(k) / CE. Uncertainty maps + human-in-the-loop keep the clinician in control.

## Maturity ladder (how to judge "realized")
| Level | Bar | OncoSeg |
|---|---|---|
| **L1 Research prototype** | Works on a static benchmark | ✅ MSD, Dice 0.797 |
| **L2 Engineering prototype** | Containerized service, stable API, tests, reproducible | ✅ FastAPI+Docker+DICOM, ~160 tests |
| **L3 Integration prototype** | Emits DICOM SEG/SR, sits behind PACS, reviewable in a viewer | ◐ Next step |
| **L4 Clinical validation** | Retrospective multi-site → prospective study, error modes characterized | ✗ |
| **L5 Regulated product** | SaMD: intended use, risk mgmt, FDA/CE clearance, deployment monitoring | ✗ |

**Current position: L2, moving toward L3.**

## Feasibility for a solo builder (honest)
- **L3 — achievable.** Simulate a real clinical stack with open-source tools (Orthanc PACS +
  OHIF/3D Slicer viewer + `highdicom` for SEG/SR). No hospital needed. ~1–2 weeks. **Highest leverage.**
- **L4 — half achievable.** Retrospective external validation on *public* cohorts (train on MSD,
  test on BraTS / LUMIERE / other public sets) demonstrates generalization / distribution shift —
  doable, and on-thesis. The **prospective clinical study needs an institution** (patients, IRB,
  radiologists).
- **L5 — not achievable solo as actual clearance.** Needs a company, QMS, evidence, money, years.
  Can produce a **regulatory-framing document** (intended use + risk sketch + predicate analysis)
  to show understanding — without claiming clearance.

## The framing that turns the gap into motivation
> "I took OncoSeg as far as one person credibly can — a deployable service (L2), a DICOM-SEG/SR
> integration behind an open-source PACS (L3), and retrospective external validation on public
> cohorts (L4-retrospective). The prospective study and regulatory path need an institution's
> patients and IRB — which is exactly what I want a PhD to give me access to. Proving reliability
> *before* you trust a system in the real world is the problem I want to research."

## Plan
- **Target L3 + L4-retrospective** for the portfolio (already beyond most applicants).
- **Do not fake L4-prospective / L5** — document the framing only; present as PhD motivation.
