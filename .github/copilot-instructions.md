## Quick context

This repository is a **ready-to-use** facial expression detection demo using a custom CNN trained from scratch (no transfer learning).
The project detects 7 expressions: angry, disgust, fear, happy, neutral, sad, surprise.

**Key files (all in repository root):**
- `detect_realtime.py` — Real-time webcam detection (main script, standalone)
- `predict_image.py <image>` — Single image prediction with result visualization
- `preprocessing.py` — Helper functions for face detection and preprocessing
- `models/expression_model.h5` — Pre-trained CNN model (~14 MB)
- `models/class_labels.json` — Label mapping (7 classes)
- `models/haarcascade_frontalface_default.xml` — Haar Cascade for face detection
- `requirements.txt` — Python dependencies (TensorFlow, OpenCV, NumPy, etc.)

**Project structure:**
```
ekspresi-wajah-demo/
├── models/
│   ├── expression_model.h5           # Pre-trained model
│   ├── class_labels.json             # Label mapping
│   └── haarcascade_frontalface_default.xml
├── detect_realtime.py                # Webcam detection
├── predict_image.py                  # Image prediction
├── preprocessing.py                  # Helper functions
├── requirements.txt
└── README.md
```

## What matters for an AI coding agent (do this first)
- **Use PowerShell** for all terminal commands — maintainer is on Windows. Example: `pip install -r requirements.txt; python detect_realtime.py`
- **Model path**: Always use `models/expression_model.h5` (13.99 MB file, already trained and ready)
- **Label file**: Use `models/class_labels.json` (NOT root `class_labels.json`)
- **Haar Cascade**: Scripts expect OpenCV's built-in cascade via `cv2.data.haarcascades`, but backup exists at `models/haarcascade_frontalface_default.xml`
- **No dataset needed**: Model is pre-trained. Users can run detection immediately after `pip install`.

## Known patterns and code contracts

**Helper API from `preprocessing.py` (used by both runtime scripts):**
- `load_haarcascade() -> cv2.CascadeClassifier` — Loads Haar Cascade from OpenCV built-in path (`cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`)
- `detect_faces(image, cascade, scale_factor=1.1, min_neighbors=5) -> list[(x,y,w,h)]` — Returns bounding boxes for detected faces
- `preprocess_face_for_prediction(face_roi, target_size=(48,48)) -> np.array` — Returns 4D tensor shape `(1, 48, 48, 1)` for model input

**Model usage pattern:**
```python
model = load_model('models/expression_model.h5')
predictions = model.predict(processed_face, verbose=0)
class_idx = np.argmax(predictions[0])
expression_label = class_labels[str(class_idx)]
```

**Label mapping (from `models/class_labels.json`):**
```json
{"0": "jijik", "1": "kaget", "2": "marah", "3": "netral", 
 "4": "sedih", "5": "senang", "6": "takut"}
```

## When editing code, preserve these behaviors
- **Window/key controls**: `q` to quit and `s` to screenshot in `detect_realtime.py`. Keep this UX unless explicitly changing UI.
- **Visual annotations**: Code draws colored boxes (BGR format) and overlays text. Colors mapped in `self.colors` dict:
  ```python
  self.colors = {
      'marah': (0, 0, 255),      # Merah (Red)
      'jijik': (0, 128, 128),    # Teal
      'takut': (128, 0, 128),    # Ungu (Purple)
      'senang': (0, 255, 0),     # Hijau (Green)
      'netral': (255, 255, 255), # Putih (White)
      'sedih': (255, 0, 0),      # Biru (Blue)
      'kaget': (0, 255, 255)     # Kuning (Yellow)
  }
  ```
- **Output filenames**: `predict_image.py` uses `image_path.replace('.', '_result.')` to create output filename. Be careful with files containing multiple dots.
- **Frame processing**: Webcam frames are horizontally flipped (`cv2.flip(frame, 1)`) for mirror effect before face detection.

## Debugging and developer flows (quick commands)
- **Install dependencies** (PowerShell):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- **Quick runtime checks**:
  ```powershell
  # Check if model exists
  Test-Path models\expression_model.h5
  
  # Run webcam demo
  python detect_realtime.py
  
  # Run single-image prediction
  python predict_image.py test.jpg
  ```
- **Test preprocessing module standalone**:
  ```powershell
  python preprocessing.py
  ```
  This will test Haar Cascade loading and print dataset info if `dataset/` folder exists.

## Editing guidance & PR notes
- If you add or modify `preprocessing.py`, ensure the three functions above exist and that their signatures match how the two scripts call them. Unit-test helpers where possible.
- If you change paths (e.g., move scripts under `src/`), update README examples and all relative path strings in scripts (`models/...`, `class_labels.json`).
- Prefer small, focused PRs: one behavioral change or bugfix per PR (for example: "fix label file path" or "add missing preprocessing functions").

## Examples to cite in code edits
- **Model load check** (from `detect_realtime.py`):
  ```python
  if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
  ```
- **Prediction pattern** (from `predict_image.py`):
  ```python
  predictions = model.predict(processed_face, verbose=0)
  class_idx = np.argmax(predictions[0])
  confidence = predictions[0][class_idx]
  expression = class_labels.get(str(class_idx), 'unknown')
  ```
- **Face detection** (from `preprocessing.py`):
  ```python
  face_cascade = load_haarcascade()
  faces = detect_faces(frame, face_cascade)
  for (x, y, w, h) in faces:
      face_roi = frame[y:y+h, x:x+w]
      processed = preprocess_face_for_prediction(face_roi)
  ```

## Important notes for developers
- **`.gitignore` excludes large files**: `models/*.h5`, `models/*.hdf5`, `dataset/`, `*.png`, `*.csv`, `*.json` (except `requirements.txt`). Be aware when committing model files or results.
- **Class labels are in Indonesian**: `marah` (angry), `jijik` (disgust), `takut` (fear), `senang` (happy), `netral` (neutral), `sedih` (sad), `kaget` (surprise).
- **Model expects 48x48 grayscale input**: Always preprocess faces to this size before prediction.
- **Dependencies**: TensorFlow 2.20.0, OpenCV ≥4.8.0, NumPy ≥1.24.0. See `requirements.txt` for full list.

## Quick start for users (copy-paste ready)
```powershell
# Clone and setup
git clone https://github.com/Razin907/facial-expression-detection.git
cd facial-expression-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run webcam detection
python detect_realtime.py

# Or predict single image
python predict_image.py path\to\your\image.jpg
```

Press `q` to quit webcam, `s` to screenshot. Results saved as `<filename>_result.<ext>`.

---

If anything in these notes is unclear or you want me to expand a section (for example: unit tests for `preprocessing.py`, or troubleshooting common OpenCV/TensorFlow errors), let me know and I'll update this file.
