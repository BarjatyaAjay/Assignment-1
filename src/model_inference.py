from transformers import pipeline
import torch


from transformers import pipeline, AutoTokenizer
import torch
import os
import importlib

# Try to import onnxruntime if installed; used when ONNX_MODEL_PATH is provided
onnxruntime = importlib.util.find_spec('onnxruntime')
if onnxruntime:
    import onnxruntime as ort
else:
    ort = None


QUESTIONS = {
    'Agreement Value': 'What is the agreement value? What is the contract value?',
    'Agreement Start Date': 'When does the agreement start? What is the start date of the agreement?',
    'Agreement End Date': 'When does the agreement end? What is the end date of the agreement?',
    'Renewal Notice (Days)': 'What is the renewal notice in days? How many days notice for renewal?',
    'Party One': 'Who is Party One? Which organization is the first party?',
    'Party Two': 'Who is Party Two? Which organization is the second party?'
}


class Extractor:
    """Extractor supports two modes:
    - layout-aware document QA (pipeline 'document-question-answering') for image inputs when a layout model is used
    - text QA (pipeline 'question-answering') as fallback for text inputs
    """
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad', device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self.model_name = model_name
        self.doc_pipeline = None
        self.text_pipeline = None
        self.onnx_session = None
        self.onnx_tokenizer = None

        # If ONNX model path is provided, try to load session + tokenizer
        onnx_model_path = os.environ.get('ONNX_MODEL_PATH')
        if onnx_model_path and ort is not None and os.path.exists(onnx_model_path):
            try:
                self.onnx_session = ort.InferenceSession(onnx_model_path)
                # tokenizer still needed to encode inputs
                self.onnx_tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                self.onnx_session = None
                self.onnx_tokenizer = None

        # Try to initialise a document-question-answering pipeline for layout-aware models
        try:
            # document-question-answering will be used for image files when available
            self.doc_pipeline = pipeline('document-question-answering', model=model_name, tokenizer=model_name, device=self.device)
        except Exception:
            self.doc_pipeline = None

        # Always try to have a text QA pipeline as a reliable fallback
        try:
            self.text_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name, device=self.device)
        except Exception:
            self.text_pipeline = None

        if self.doc_pipeline is None and self.text_pipeline is None:
            raise RuntimeError('Could not initialize any pipeline for model: ' + str(model_name))

    def extract_from_text(self, text):
        results = {}
        # If ONNX session is available, use ONNX inference for speed
        if self.onnx_session is not None and self.onnx_tokenizer is not None:
            for field, q in QUESTIONS.items():
                try:
                    ans = self._onnx_qa_predict(q, text)
                    results[field] = ans
                except Exception:
                    results[field] = ''
            return results

        if self.text_pipeline is None:
            # no text pipeline available
            for field in QUESTIONS:
                results[field] = ''
            return results
        for field, q in QUESTIONS.items():
            try:
                out = self.text_pipeline(question=q, context=text, max_answer_len=200)
                results[field] = out.get('answer', '').strip()
            except Exception:
                results[field] = ''
        return results

    def _onnx_qa_predict(self, question, context, max_len=512):
        """Run a lightweight ONNX QA model (start/end logits) and return extracted text."""
        if self.onnx_session is None or self.onnx_tokenizer is None:
            return ''
        tok = self.onnx_tokenizer(question, context, truncation='only_second', max_length=max_len, return_offsets_mapping=True, return_tensors='np')
        input_feed = {}
        # onnxruntime expects numpy arrays named like model inputs
        for name in ['input_ids', 'attention_mask', 'token_type_ids']:
            if name in tok:
                input_feed[name] = tok[name]
        # run
        outputs = self.onnx_session.run(None, input_feed)
        # outputs order: [start_logits, end_logits]
        if not outputs or len(outputs) < 2:
            return ''
        import numpy as np
        start_logits = outputs[0][0]
        end_logits = outputs[1][0]
        start_idx = int(np.argmax(start_logits))
        end_idx = int(np.argmax(end_logits))
        if start_idx > end_idx:
            # swap or return empty
            return ''
        offsets = tok['offset_mapping'][0]
        # Map token indices to char spans; ensure indices within offsets
        if start_idx >= len(offsets) or end_idx >= len(offsets):
            return ''
        start_char = int(offsets[start_idx][0])
        end_char = int(offsets[end_idx][1])
        return context[start_char:end_char].strip()

    def extract_from_file(self, file_path):
        _, ext = os.path.splitext(file_path.lower())
        is_image = ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

        # If we have a document pipeline and the input is an image, use it
        if is_image and self.doc_pipeline is not None:
            results = {}
            for field, q in QUESTIONS.items():
                try:
                    out = self.doc_pipeline(image=file_path, question=q)
                    # document pipeline may return a list or dict
                    if isinstance(out, list) and len(out) > 0:
                        out = out[0]
                    results[field] = out.get('answer', '').strip() if isinstance(out, dict) else ''
                except Exception:
                    results[field] = ''
            return results

        # Otherwise fall back to text-based extraction
        from src.ocr import extract_text
        text = extract_text(file_path)
        return self.extract_from_text(text)


if __name__ == '__main__':
    import argparse
    from src.ocr import extract_text

    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--model', required=False, default='distilbert-base-uncased-distilled-squad')
    args = p.parse_args()
    ext = Extractor(model_name=args.model)
    print(ext.extract_from_file(args.file))
