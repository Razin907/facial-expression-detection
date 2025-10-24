## Quick context

This repository is a small demo for real-time facial expression detection using a custom CNN.
Key runnable scripts (present in this workspace root):
- `detect_realtime.py` — opens webcam, detects faces with a Haar cascade and classifies expressions with `models/expression_model.h5`.
- `predict_image.py <image>` — runs prediction on a single image and saves a result image.
- `models/` — contains `expression_model.h5` and `class_labels.json` (the model and label mapping).

Note: README mentions a `src/` layout and `preprocessing.py` under `src/`, but in this copy the repository root contains `detect_realtime.py` and `predict_image.py` and there is no `preprocessing.py` file. When changing or running code, verify where helper modules live and adjust import paths or add the missing module.

## What matters for an AI coding agent (do this first)
- Use PowerShell when suggesting commands — the maintainer works on Windows. Example: `pip install -r requirements.txt` and `python detect_realtime.py`.
- Always prefer the explicit model path `models/expression_model.h5` when editing code or running scripts. Both runtime scripts expect that file; if it is missing, they raise a FileNotFoundError.
- Label file lives in `models/class_labels.json` but scripts reference `class_labels.json` (root). Either update the script to `models/class_labels.json` or ensure a copy exists at repo root. Point out this mismatch when making PRs.

## Known patterns and small code contracts
- Helper API expected from `preprocessing` (used by both scripts):
  - `load_haarcascade() -> cv2.CascadeClassifier` (used to get face cascade)
  - `detect_faces(frame, cascade) -> list[(x,y,w,h)]` (returns bounding boxes)
  - `preprocess_face_for_prediction(face_roi) -> numpy.array` (should return a 4D tensor suitable for `model.predict`, e.g. shape `(1, h, w, 1)`)
- Model usage: `model = load_model('models/expression_model.h5')`, `predictions = model.predict(processed_face, verbose=0)`, then `np.argmax(predictions[0])` to get class index.

## When editing code, preserve these behaviors
- Window/key controls: `q` to quit and `s` to screenshot in `detect_realtime.py`. Keep this UX unless explicitly changing UI.
- Visual annotations: code draws colored boxes and overlays text; colors are in BGR tuples (see `self.colors` / `colors` dict). Keep that mapping or update consistently.
- Output filenames: `predict_image.py` uses a simple replace to create `<name>_result.<ext>` — be careful when changing to avoid double-dot issues.

## Debugging and developer flows (quick commands)
- Install dependencies (PowerShell):
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
  ```
- Quick runtime checks:
  - Model present: `if (-Not (Test-Path models\expression_model.h5)) { Write-Host 'model missing' }`
  - Run webcam demo: `python detect_realtime.py`
  - Run single-image prediction: `python predict_image.py test.jpg`

## Editing guidance & PR notes
- If you add or modify `preprocessing.py`, ensure the three functions above exist and that their signatures match how the two scripts call them. Unit-test helpers where possible.
- If you change paths (e.g., move scripts under `src/`), update README examples and all relative path strings in scripts (`models/...`, `class_labels.json`).
- Prefer small, focused PRs: one behavioral change or bugfix per PR (for example: "fix label file path" or "add missing preprocessing functions").

## Examples to cite in code edits
- Model load check (from `detect_realtime.py`):
  - `if not os.path.exists(model_path): raise FileNotFoundError(...)`
- Prediction pattern (from `predict_image.py`):
  - `predictions = model.predict(processed_face, verbose=0); class_idx = np.argmax(predictions[0])`

## Anything to call out to a human reviewer
- Mismatch between README (`src/` path) and repository layout here — flag this in PR description and suggest a single canonical layout.
- Missing `preprocessing.py` in this snapshot — if you add it, include a brief unit test or example usage to show it returns the expected data shapes.

If anything in these notes is unclear or you want me to expand a short section into checks or small unit tests (for example: a test for `preprocess_face_for_prediction` output shape), tell me what to add and I'll update this file.
