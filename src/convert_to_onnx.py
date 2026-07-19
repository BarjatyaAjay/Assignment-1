"""Export a QA model to ONNX for faster CPU inference (best-effort stub).

Usage:
  python src/convert_to_onnx.py --model distilbert-base-uncased-distilled-squad --out onnx/model.onnx

Notes: This is a simple example. For production-grade export use `optimum` or HF onnx-export tooling.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def export_onnx(model_name, output_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()

    # Prepare a dummy input (token ids) using tokenizer
    sample = tokenizer("What is the start date?", "Sample context for export.", return_tensors='pt')

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    inputs = (sample['input_ids'], sample['attention_mask'], sample.get('token_type_ids', torch.zeros_like(sample['input_ids'])))

    # Trace and export
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                args=inputs,
                f=output_path,
                input_names=input_names,
                output_names=['start_logits', 'end_logits'],
                opset_version=11,
                do_constant_folding=True,
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'seq'},
                    'attention_mask': {0: 'batch', 1: 'seq'},
                    'token_type_ids': {0: 'batch', 1: 'seq'},
                    'start_logits': {0: 'batch', 1: 'seq'},
                    'end_logits': {0: 'batch', 1: 'seq'},
                }
            )
            print('Exported ONNX model to', output_path)
        except Exception as e:
            print('ONNX export failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    export_onnx(args.model, args.out)


if __name__ == '__main__':
    main()
