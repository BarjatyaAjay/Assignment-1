# Meta Data Extraction from Documents

Solution for the assignment: extract metadata fields from heterogeneous documents (images or docx) using an ML approach (no rule-based regex).

Key points
- Approach: OCR + transformer-based question-answering to locate field values in the document text. This avoids hard-coded rules and uses an ML model to find answers.
- Fields extracted: Agreement Value, Agreement Start Date, Agreement End Date, Renewal Notice (Days), Party One, Party Two
- Files:
  - `requirements.txt` : Python deps
  - `src/data_loader.py` : loads CSVs and file lists
  - `src/ocr.py` : extract text from `.png` and `.docx`
  - `src/model_inference.py` : runs a QA model to extract fields
  - `src/predict.py` : run inference on a folder and save predictions
  - `src/evaluate.py` : compute per-field recall
  - `src/app.py` : small Flask app exposing `/extract` endpoint

How to run
1. Create a virtualenv and install deps:

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run predictions on test set:

```bash
python src/predict.py --data-dir data/test --csv data/test.csv --out predictions.csv
```

3. Evaluate (produces per-field recall):

```bash
python src/evaluate.py --pred predictions.csv --gold data/test.csv
```

4. Run REST API (development):

```bash
python src/app.py
# POST a file to http://127.0.0.1:5000/extract
```

Docker (recommended for deploy)

```bash
docker build -t doc-extractor:latest .
docker run -p 5000:5000 --rm doc-extractor:latest
# POST multipart/form-data file -> http://127.0.0.1:5000/extract
```

Performance notes and how I optimized for speed
- Default model switched to a lightweight distilled QA model: `distilbert-base-uncased-distilled-squad`.
- Parallel prediction: `src/predict.py` uses multiple processes and caches extracted text under `cache/texts/` to avoid repeated OCR calls.
- Dockerfile installs required system packages (`tesseract-ocr`, `poppler-utils`) and runs via `gunicorn` for production.

Next steps to further improve speed before launch
- Fine-tune a layout-aware model (Donut / LayoutLM) on `train/` and export a faster optimized model (ONNX / TorchScript).
- Use GPU-enabled container on deployment (change Docker base image and install CUDA drivers) for much faster inference.

Using layout-aware models (LayoutLM / Donut)
- The API supports layout-aware models via the `document-question-answering` pipeline when you provide a layout-capable model name or local path in the `MODEL_PATH` env var (or by calling `/reload` with `{"model_path": "./outputs/qa-model"}`).
- If you plan to use a model hosted on Hugging Face that requires a token or special license, avoid pointing Docker to download it at runtime. Instead, download and save the model locally (or mount a model directory into the container) and set `MODEL_PATH` to the mounted path.

Docker usage notes for private or large models
- If the desired layout model is publicly downloadable, the default Dockerfile will try to download on container start. For restricted models or to avoid large downloads during container startup, build the container without model download and mount a local `./outputs/qa-model` into `/app/outputs/qa-model` at runtime, then set `MODEL_PATH` to that path.

Example run with mounted model directory:

```bash
docker build -t doc-extractor:latest .
docker run -p 5000:5000 --rm -v $(pwd)/outputs/qa-model:/app/outputs/qa-model -e MODEL_PATH=/app/outputs/qa-model doc-extractor:latest
```

If a layout model isn't allowed/available for deployment, the service will automatically fall back to a text-based QA model. This lets you deploy quickly with the lightweight default model and upgrade to a layout model later.

Non-Docker deployment helper files
- `start.sh`: Minimal start script that activates a venv (if present) and runs `gunicorn`. Usage:

```bash
chmod +x start.sh
# optionally set MODEL_PATH and HOST/PORT
./start.sh
```

- `deploy/systemd.service`: Example `systemd` unit. Copy to `/etc/systemd/system/doc-extractor.service` and update `WorkingDirectory` and `ExecStart` paths. Then `systemctl enable --now doc-extractor`.

- `src/convert_to_onnx.py`: Helper stub to export a QA model to ONNX for faster CPU inference. Example:

```bash
python src/convert_to_onnx.py --model distilbert-base-uncased-distilled-squad --out onnx/distilbert_qa.onnx
```

Tips for a free deploy without Docker
- Use a single `gunicorn` worker: `gunicorn -w 1 -b 0.0.0.0:8000 src.app:app`.
- Install system dependencies on the host: `sudo apt-get install -y tesseract-ocr poppler-utils`.
- Pre-download or mount your model into `./outputs/qa-model` and use `MODEL_PATH` to avoid large downloads at startup.

ONNX acceleration
- You can export a trained or HF model to ONNX using `src/convert_to_onnx.py` and set the environment variable `ONNX_MODEL_PATH` to the ONNX file path before starting the service. When `ONNX_MODEL_PATH` is present and `onnxruntime` is installed, the service will use the ONNX model for faster CPU inference.

Example of using ONNX:

```bash
python src/convert_to_onnx.py --model distilbert-base-uncased-distilled-squad --out onnx/distilbert_qa.onnx
export ONNX_MODEL_PATH=$(pwd)/onnx/distilbert_qa.onnx
./start.sh
```



Notes
- The repository uses a pre-trained QA transformer; for better performance on documents, fine-tuning on the provided `train/` set is recommended (see `src/model_inference.py` for where to add fine-tuning).
- This solution is deliberately model-driven (QA) and does not rely on regex or hand-crafted rules.
