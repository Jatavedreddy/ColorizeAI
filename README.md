# ColorizeAI

ColorizeAI is an image and video colorization project built around DDColor as the primary model, with ECCV16 and SIGGRAPH17 retained as fallback and comparison models.

The current runnable application is a Flask backend serving a static HTML/CSS/JavaScript frontend. The main runtime path is:

- `main.py`
- `backend/app.py`
- `frontend/`
- `src/colorizeai/`

## Current Architecture

The app is organized into four layers:

1. Web layer
   - Flask API in `backend/app.py`
   - static frontend in `frontend/`
2. Core inference layer
   - `src/colorizeai/core/colorization.py`
   - `src/colorizeai/core/ddcolor_model.py`
   - `src/colorizeai/core/models.py`
3. Enhancement layer
   - smart fusion
   - reference-guided colorization
   - color hints
   - style presets
   - temporal consistency for video
4. Evaluation layer
   - metrics and benchmarking helpers in `src/colorizeai/utils/metrics.py`

## Features

- Single-image colorization
- Enhanced colorization with optional fusion and reference guidance
- Batch image processing to ZIP output
- Video colorization with temporal consistency
- Evaluation lab with PSNR, SSIM, LPIPS, and colorfulness metrics when available

## Project Structure

```text
ColorizeAI/
├── main.py
├── backend/
├── frontend/
├── src/colorizeai/
│   ├── core/
│   ├── features/
│   └── utils/
├── DDColor/
├── scripts/
├── tools/
├── docs/
├── archive/non_current/
├── assets/
└── test_images/
```

## Main Entry Points

- `main.py`: starts the Flask app on port `8080` by default
- `backend/app.py`: defines API routes and file handling
- `frontend/index.html`: main UI layout
- `frontend/js/app.js`: browser-side API calls and result rendering
- `src/colorizeai/core/colorization.py`: core image inference pipeline

## API Routes

The current backend exposes these main routes:

- `POST /api/colorize/basic`
- `POST /api/colorize/enhanced`
- `POST /api/colorize/batch`
- `POST /api/colorize/video`
- `POST /api/evaluate`

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure DDColor is available

This repository expects the local `DDColor/` directory to exist. The wrapper in `src/colorizeai/core/ddcolor_model.py` will try to load weights from:

- `DDCOLOR_WEIGHTS` environment variable
- `DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt`
- some fallback local paths

If DDColor cannot be loaded, the app will fall back to SIGGRAPH17 and ECCV16.

### 3. Run the app

```bash
python main.py
```

Open:

- `http://127.0.0.1:8080/`

## How Inference Works

At a high level, the image pipeline is:

1. frontend uploads an image or video
2. Flask reads the file into a NumPy array
3. `colorize_highres()` or `colorize_highres_enhanced()` runs
4. DDColor is attempted first when enabled
5. if DDColor fails, ECCV16 and SIGGRAPH17 are used
6. optional enhancement modules refine the output
7. result is returned as base64 image, ZIP, or MP4

The project primarily uses Lab color space internally:

- `L`: luminance
- `a` and `b`: chroma channels

The models mainly predict the missing `ab` channels, then reconstruct the final RGB image.

## Frontend Workflows

The UI currently provides five workflows:

1. Quick Colorize
2. Advanced or Enhanced Colorize
3. Batch Processing
4. Video Processing
5. Metrics Lab

## Important Notes

- The current app is not Gradio-based. Some older documents in the repository describe previous layouts or workflows.
- A number of historical or non-runtime files were moved into `archive/non_current/` to keep the active project surface cleaner.
- The archive is conservative. Files were moved there because they are not part of the current runtime path, not because they are useless.

## Archived Material

Archived non-current items are grouped here:

- `archive/non_current/notebooks/`
- `archive/non_current/experiments/`
- `archive/non_current/temp/`
- `archive/non_current/docs_history/`

See:

- `archive/non_current/README.md`

## Documentation

Recommended docs for understanding the active codebase:

- `docs/NEWCOMER_CODEBASE_GUIDE.md`
- `docs/PROJECT_SUMMARY.md`
- `docs/DDCOLOR_INTEGRATION.md`
- `docs/UNIQUE_FEATURES.md`
- `RUN.md`

## Notes On Evaluation

The evaluation route compares multiple variants including:

- ECCV16
- SIGGRAPH17
- DDColor
- DDColor + Fusion
- DDColor + style variants

Available metrics depend on installed optional packages.

## Troubleshooting

### DDColor is not loading

Check that weights exist at one of the expected locations, or set:

```bash
export DDCOLOR_WEIGHTS=/absolute/path/to/pytorch_model.pt
```

### App starts but colorization falls back

That usually means DDColor import or weight loading failed. The app should still run using classical models.

### Video processing is slow

This is expected. The current web route intentionally downsizes larger videos and supports frame skipping to keep processing practical.

## Contributing

If you make structural changes, keep runtime docs aligned with the actual executable files. The most important files to keep accurate are:

- `README.md`
- `main.py`
- `backend/app.py`
- `frontend/index.html`
- `frontend/js/app.js`

## License

MIT License.
