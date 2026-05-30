# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lada is a video restoration application that detects and recovers pixelated/mosaic areas in videos. It provides both a GTK4/Libadwaita GUI and a CLI interface, sharing a common processing pipeline. Written in Python 3.13, licensed AGPL-3.0.

## Build & Run Commands

```bash
# Install with GPU-specific extras (mutually exclusive)
uv sync --extra nvidia       # NVIDIA Volta+ (CUDA 12.8)
uv sync --extra nvidia-legacy # NVIDIA Maxwell/Pascal (CUDA 12.6)
uv sync --extra intel        # Intel Arc discrete/integrated GPUs (XPU)
uv sync --extra cpu          # CPU only (extremely slow)

# Apply dependency patches after install (required)
uv pip install patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
uv pip uninstall patch

# Run
lada          # GUI (requires GTK4 system libs — see docs/windows_install.md)
lada-cli      # CLI

# Install GUI dev dependencies (type stubs)
uv pip install --group gui-dev

# Compile translations
powershell -ExecutionPolicy Bypass .\translations\compile_po.ps1
bash translations/compile_po.sh   # Linux/macOS

# Update translation template (triggers Weblate auto-sync)
bash translations/update_pot.sh
```

There is no test suite. Validation is done via manual smoke testing (see packaging/README.md).

## Architecture

### Two Entry Points → One Pipeline

- **GUI**: `lada.gui.main:main` → `LadaApplication(Adw.Application)` → `MainWindow` with Watch/Export/FileSelection views
- **CLI**: `lada.cli.main:main` → argparse → direct pipeline invocation
- Both converge on `lada.restorationpipeline.frame_restorer.FrameRestorer`

### Processing Pipeline (frame_restorer.py)

`FrameRestorer` orchestrates a **threaded queue-based pipeline** with three stages:

1. **Frame Reader** → reads video frames (PyAV), pushes to `frame_restoration_queue`
2. **Mosaic Detector** (`MosaicDetector` via `Yolo11SegmentationModel`) → detects mosaic regions, outputs `Clip` objects with bounding boxes/masks to `mosaic_clip_queue`
3. **Mosaic Restorer** (`BasicvsrppMosaicRestorer` or `DeepmosaicsMosaicRestorer`) → restores detected regions, pushes to `restored_clip_queue`

Queues use sentinel markers (`EOF_MARKER`, `STOP_MARKER`, `ErrorMarker`) for lifecycle control. Queue sizes are bounded by ~512MB memory budget.

### Model Selection (restorationpipeline/__init__.py)

`load_models()` dispatches by model name prefix: `deepmosaics-*` → DeepMosaics BVDNet, `basicvsrpp-*` → BasicVSR++. Detection uses YOLO v11 segmentation. Model files are discovered by `ModelFiles` in `lada/__init__.py` from the `model_weights/` directory.

### GUI Architecture (lada/gui/)

GTK4/Libadwaita with template-based UI (`*.ui` files + `@Gtk.Template` decorator). GStreamer pipeline for video playback. Builder pattern via `FrameRestorerOptionsBuilder` for configuring restoration. Platform detection at runtime (`IS_FLATPAK`, `sys.platform`).

### Key Type Definitions (lada/utils/__init__.py)

Central type aliases used across the codebase:
- `Image` = np.ndarray (H,W,3) BGR uint8
- `ImageTensor` = torch.Tensor equivalent
- `Mask` / `MaskTensor` = segmentation masks
- `Box` = (top, left, bottom, right) pixel coords
- `Detection`, `Detections` = detection results
- `DETECTION_CLASSES` = class ID/mask value mapping (nsfw=0, sfw_head=1, sfw_face=2, watermark=3, mosaic=4)

## Key Conventions

- **Image color order is BGR** (OpenCV convention), not RGB
- **Pinned dependencies** (`ultralytics==8.4.4`, `mmengine==0.10.7`) must not be upgraded without updating corresponding patches in `patches/`
- **GPU extras are mutually exclusive** — declared as conflicts in `pyproject.toml` under `[tool.uv.conflicts]`
- **Version** is defined in `lada/__init__.py` as `VERSION`, read dynamically by setuptools
- **Environment variables**: `LOG_LEVEL` (default WARNING), `LADA_MODEL_WEIGHTS_DIR` (override model path), `LOCALE_DIR`, `LANGUAGE`
- **Telemetry suppression**: `ALBUMENTATIONS_OFFLINE`, `ALBUMENTATIONS_NO_TELEMETRY`, `YOLO_VERBOSE=false` are set in `lada/__init__.py`

## Module Map

| Module | Purpose |
|--------|---------|
| `lada.restorationpipeline/` | Core pipeline: frame reading, mosaic detection, restoration |
| `lada.models/basicvsrpp/` | BasicVSR++ video restoration (includes `mmagic/` subpackage) |
| `lada.models/yolo/` | YOLO v11 segmentation wrapper |
| `lada/models/deepmosaics/` | DeepMosaics BVDNet restoration |
| `lada.models/dover/` | DOVER video quality assessment (dataset filtering) |
| `lada.utils/video_utils.py` | Video I/O via PyAV, encoding presets, metadata |
| `lada.utils/threading_utils.py` | Pipeline queues, worker threads, sentinel markers |
| `lada.utils/image_utils.py` | Image processing, mosaic application, blending |
| `lada.utils/mask_utils.py` | Segmentation mask processing |
| `lada.datasetcreation/` | Dataset creation pipeline with various detectors |
| `scripts/training/` | Training scripts for detection and restoration models |
| `scripts/dataset_creation/` | Scripts for creating training datasets |
| `configs/` | Training configs (BasicVSR++ stages, YOLO dataset YAMLs) |
| `packaging/` | Platform-specific packaging (Docker, Flatpak, Windows, macOS) |
| `patches/` | Post-install patches for third-party packages |

## Translation Workflow

Translations use gettext with Weblate integration. `.pot` file updates trigger auto-sync to all `.po` files on Weblate. Translation PRs from Weblate should be merged before updating `.pot` and before releases. Only languages listed in `release_ready_translations.txt` are included in packaged releases.
