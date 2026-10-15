# Contributing to Dynamo

Thank you for your interest in contributing to Dynamo (Agent Visual Harness)!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0dvsh0/dynamo.git
   cd dynamo
   ```

2. **Create a virtual environment and install in editable mode:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .[dev]
   ```

## System Requirements

- **Python**: 3.10, 3.11, or 3.12
- **FFmpeg**: Required for screen recording, keyframe extraction, and MP4/GIF generation.
- **X11 / Xvfb** *(Linux / WSL2)*: Required for virtual headless display tests (`xvfb`, `x11-xserver-utils`).
- **Tesseract OCR** *(Optional)*: Required for OCR bounding box extraction.

On Debian / Ubuntu / WSL2:
```bash
sudo apt-get update
sudo apt-get install -y xvfb ffmpeg x11-xserver-utils tesseract-ocr fontconfig
```

On Windows:
Core primitives and non-X11 units run natively without requiring Xvfb. Tests that require an X11 server automatically skip.

## Running Tests

Run the full test suite with:
```bash
pytest -ra
```

To run fast non-display unit tests only:
```bash
pytest tests/test_units.py tests/test_primitives.py
```

## Code Style

- Format with `ruff format`
- Check lints with `ruff check`
- Type annotations should pass `mypy` / Python 3.10+ strict typing

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Commit changes with clear, descriptive commit messages.
3. Ensure all tests pass.
4. Open a pull request against the `main` branch.
