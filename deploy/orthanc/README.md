# Simulated PACS — Orthanc (L3 step 4)

A local, open-source DICOM server that stands in for a hospital PACS. OncoSeg reads studies
from it and pushes back DICOM SEG/SR. See `../../docs/L3_Integration_Plan.md`.

## Prerequisite: Docker
Docker is not installed yet. On macOS:
```bash
brew install --cask docker      # or download Docker Desktop from docker.com
open -a Docker                   # launch it; wait for the whale icon to settle
docker info                      # should print server info when ready
```

## Bring it up
```bash
cd deploy/orthanc
docker compose up -d
docker compose ps                # orthanc should be "running"
```

## Verify it works
```bash
# 1) Web UI (no login — dev config)
open http://localhost:8042

# 2) REST API is alive
curl -s http://localhost:8042/system | python3 -m json.tool   # prints Orthanc version, Name=OncoSeg-PACS

# 3) DICOMweb (QIDO-RS) responds — empty list at first, but a 200/204 means the plugin is on
curl -s -o /dev/null -w "QIDO-RS studies -> HTTP %{http_code}\n" \
  http://localhost:8042/dicom-web/studies
```

## Endpoints OncoSeg will use
| Purpose | Interface | URL |
|---|---|---|
| Web UI / browse | — | http://localhost:8042 |
| Query studies | QIDO-RS | `GET /dicom-web/studies` |
| Retrieve a series | WADO-RS | `GET /dicom-web/studies/{study}/series/{series}` |
| Push results back (SEG/SR) | STOW-RS | `POST /dicom-web/studies` |
| Classic push | DICOM C-STORE | `localhost:4242`, AET `ORTHANC` |

## Stop / reset
```bash
docker compose down           # stop (keeps stored studies in the named volume)
docker compose down -v        # stop AND wipe all stored DICOM
```

> ⚠️ Dev config only: no authentication, remote access allowed. Never expose this to a real network.
