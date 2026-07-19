import argparse
import pandas as pd
from tqdm import tqdm
from src.data_loader import load_metadata
from src.ocr import extract_text
from src.model_inference import Extractor
import os
import hashlib
from multiprocessing import Pool, cpu_count


def _cache_text(file_path, cache_dir='cache/texts'):
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{h}.txt")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as fh:
            return fh.read()
    text = extract_text(file_path)
    with open(cache_file, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return text


def _worker(args):
    file_name, file_path, model_name = args
    # cache OCR text to speed up repeated reads, but let extractor choose best method
    _cache_text(file_path)
    extractor = Extractor(model_name=model_name)
    res = extractor.extract_from_file(file_path)
    res['file_name'] = file_name
    return res


def predict(csv_path, root_dir, out_csv, model_name='distilbert-base-uncased-distilled-squad', processes=None):
    df = load_metadata(csv_path, root_dir)
    tasks = [(row['file_name'], row['file_path'], model_name) for _, row in df.iterrows()]
    processes = processes or max(1, min(4, cpu_count() - 1))
    preds = []
    with Pool(processes) as pool:
        for res in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
            preds.append(res)
    out = pd.DataFrame(preds)
    out.to_csv(out_csv, index=False)
    print('Wrote predictions to', out_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', dest='out_csv', required=True)
    p.add_argument('--model', dest='model_name', required=False, default='distilbert-base-uncased-distilled-squad')
    p.add_argument('--processes', type=int, required=False)
    args = p.parse_args()
    predict(args.csv, args.data_dir, args.out_csv, model_name=args.model_name, processes=args.processes)
