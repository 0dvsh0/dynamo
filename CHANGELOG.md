# Changelog

All notable changes to Dynamo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-15

### Added
- Model-facing **Triage layer** (`agent_visual_harness.vision`) for multimodal VLM routing.
- Multimodal VLM backends:
  - `OllamaBackend` (local, zero-temperature inference, bounded token generation with `num_predict`).
  - `LayaBackend` (lightweight decision model filter with Apache-2.0 license).
  - `RemoteBackend` (opt-in cloud escalation).
- Token reduction estimator (`estimate_token_saving`) and visual diff routing.
- Package surface introspection tests (`TestPackageSurface`).
- Support for `on_answer` real-time triage progress callbacks.

### Changed
- Refactored demo suite to generate reproducible evidence artifacts and interactive webapp.
- Corrected Laya license documentation to Apache-2.0.

## [1.0.0] - 2026-10-14

### Added
- Headless X11 / Xvfb virtual display management with automated display negotiation.
- Continuous screen recording via FFmpeg with non-blocking subprocess pipeline.
- Keyframe extraction and visual diffing with perceptual hashing and SSIM-like thresholds.
- Optical Character Recognition (OCR) integration with spatial bounding boxes.
- Spatial grounding engine for locating UI elements and buttons.
- Token reducer for optimizing VLM image context windows.
- CLI tool `dynamo` with commands: `record`, `inspect`, `export`, `triage`.
- Unit and integration test suite with Xvfb fallback skipping on non-Linux platforms.
