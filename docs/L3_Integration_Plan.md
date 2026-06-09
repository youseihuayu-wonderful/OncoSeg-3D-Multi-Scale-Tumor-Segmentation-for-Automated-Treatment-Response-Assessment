# L3 — Integration Prototype Plan (DICOM SEG/SR + PACS + Viewer)

Goal of L3 (see `Deployment_Roadmap.md` for the L1–L5 ladder): make OncoSeg speak DICOM
end-to-end — **DICOM in → model → write standard DICOM SEG/SR → push back → radiologist reviews
in a real viewer** — using open-source tools as a simulated clinical stack. No hospital needed.

## Architecture (data flow)
```
①  Orthanc (open-source PACS / DICOM server)
        │  DICOM in (DICOMweb / C-STORE)
        ▼
②  OncoSeg service (existing FastAPI)
        ├─ read DICOM series → 3D volume
        ├─ run model → mask + uncertainty
        ├─ compute RECIST → response category
        ├─ write DICOM SEG  (segmentation overlay)   ← new core #1
        └─ write DICOM SR   (measurement / response report)  ← new core #2
        │  DICOM out (push back to Orthanc)
        ▼
③  OHIF Viewer / 3D Slicer
        → radiologist sees AI contour overlay + report, accept / edit / reject
```

## Where to start: the DICOM SEG writer (the foundation)
The hardest, most central piece is turning a segmentation mask into a valid DICOM SEG object.
Build and verify this in isolation first; everything else is plumbing.

First concrete actions:
1. Install `highdicom` (the standard library for writing SEG/SR).
2. Take an existing segmentation mask + its source DICOM series.
3. Use `highdicom.seg.Segmentation` to write a DICOM SEG that references the source instances.
4. Verify: open the source series + the SEG in **3D Slicer**; confirm the contour overlays correctly.

## Incremental build order (each step independently verifiable)
| Step | Build | Verify | Why |
|---|---|---|---|
| 1 | **DICOM SEG writer** (mask → SEG, `highdicom`) | overlay shows correctly in 3D Slicer | ← start here |
| 2 | **DICOM SR writer** (RECIST measures + response → SR, TID 1500) | structured measurements visible in Slicer/OHIF | machine-readable measurements |
| 3 | **Wire into FastAPI**: `/predict/dicom-seg` (DICOM series in → SEG+SR out) | curl round-trip | service-ified |
| 4 | **Stand up Orthanc** (Docker) as PACS; script: study arrives → OncoSeg → push SEG/SR back | DICOM in/out loop | simulated PACS |
| 5 | **OHIF Viewer** pointed at Orthanc | overlay + report in browser; demo review | clinical view |
| 6 | **End-to-end demo script + screenshots/recording** | one-command demo | portfolio artifact |

## Tools (all open-source / free)
| Role | Tool |
|---|---|
| Write DICOM SEG/SR | **highdicom** + pydicom |
| Simulated PACS | **Orthanc** (Docker) |
| Viewer | **OHIF Viewer** (Docker) or **3D Slicer** (desktop) |
| Service | existing **FastAPI** (already supports DICOM input) |

## Scope guardrails (honest)
- "accept / edit / reject" is a *viewer* capability (OHIF/Slicer already edit SEG). Demonstrate that
  AI output lands in a standard viewer where a radiologist *could* edit it — **don't build a custom
  editing UI**.
- Get **one series / one case** working end-to-end before any batch processing.

## Definition of Done (L3)
> A public-data case enters Orthanc → OncoSeg auto-produces a DICOM SEG + SR → it opens in
> OHIF / 3D Slicer showing the **contour overlay + measurement report** → recorded as a demo
> (screenshots / short video).

Then the interview line is earned: *"DICOM in, AI segmentation, DICOM SEG/SR back, reviewable in a
standard viewer."*

## First cut
Step 1 — the DICOM SEG writer: add `src/dicom/seg_writer.py` (mask → DICOM SEG via `highdicom`)
+ unit tests with a synthetic DICOM fixture (no real data needed). That is L3's first real line of code.
