import os
import pandas as pd


def load_metadata(csv_path, root_dir=None):
    df = pd.read_csv(csv_path)
    if root_dir:
        df['file_path'] = df['file_name'].apply(lambda f: os.path.join(root_dir, f))
    else:
        df['file_path'] = df['file_name']
    return df


def list_files_in_dir(dir_path):
    files = []
    for entry in os.listdir(dir_path):
        path = os.path.join(dir_path, entry)
        if os.path.isfile(path):
            files.append(path)
    return files


if __name__ == '__main__':
    # quick sanity
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--root', required=False)
    args = p.parse_args()
    df = load_metadata(args.csv, args.root)
    print(df.head())
