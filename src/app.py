from flask import Flask, request, jsonify
import tempfile
import os
from src.ocr import extract_text
from src.model_inference import Extractor

app = Flask(__name__)

# Allow using a locally trained model directory via env `MODEL_PATH` or a HF model name
MODEL_PATH = os.environ.get('MODEL_PATH')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'distilbert-base-uncased-distilled-squad')
model_source = MODEL_PATH if MODEL_PATH else DEFAULT_MODEL
extractor = Extractor(model_name=model_source)


@app.route('/extract', methods=['POST'])
def extract_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'file is required'}), 400
    f = request.files['file']
    suffix = os.path.splitext(f.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        res = extractor.extract_from_file(tmp_path)
        return jsonify(res)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload model from provided JSON body: {"model_path": "..."}
    This allows swapping to a trained model saved locally or on HF hub.
    """
    payload = request.get_json() or {}
    model_path = payload.get('model_path') or os.environ.get('MODEL_PATH')
    if not model_path:
        return jsonify({'error': 'model_path required in JSON body or MODEL_PATH env var set'}), 400
    global extractor
    try:
        extractor = Extractor(model_name=model_path)
        return jsonify({'status': 'reloaded', 'model': model_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
