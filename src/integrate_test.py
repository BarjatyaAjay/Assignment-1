"""
Integration test: POST all files from `data/test/` to the running API and save predictions.csv,
then run evaluation using `src/evaluate.py`.

Usage: ensure the API is running at `http://127.0.0.1:5000` before running this script.
"""
import os
import requests
import argparse
import pandas as pd
from tqdm import tqdm


def post_file(url, file_path):
    with open(file_path, 'rb') as fh:
        files = {'file': (os.path.basename(file_path), fh)}
        r = requests.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def run_all(api_url, test_dir, test_csv, out_csv='predictions.csv'):
    df = pd.read_csv(test_csv)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['File Name'] if 'File Name' in row else row.get('File Name'.strip(), row.get('file_name', None))
        if fname is None:
            fname = row.iloc[0]
        file_path = os.path.join(test_dir, fname)
        if not os.path.exists(file_path):
            # try common extensions
            found = False
            for ext in ['.docx', '.png', '.jpg', '.jpeg']:
                p = file_path + ext
                if os.path.exists(p):
                    file_path = p
                    found = True
                    break
            if not found:
                print('Skipping missing file', file_path)
                continue
        try:
            res = post_file(f"{api_url}/extract", file_path)
        except Exception as e:
            print('Error posting', file_path, e)
            res = {}
        res['file_name'] = os.path.basename(file_path)
        rows.append(res)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print('Wrote', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:5000')
    parser.add_argument('--test-dir', default='data/test')
    parser.add_argument('--test-csv', default='data/test.csv')
    parser.add_argument('--out', default='predictions.csv')
    args = parser.parse_args()
    run_all(args.api, args.test_dir, args.test_csv, args.out)
