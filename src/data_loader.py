import os
import pandas as pd


def load_metadata(csv_path, root_dir=None):
    df = pd.read_csv(csv_path)
    # Handle both 'File Name' and 'file_name' columns
    if 'File Name' in df.columns:
        df['file_name'] = df['File Name']
    elif 'file_name' not in df.columns:
        # If neither exists, use the first column
        df['file_name'] = df.iloc[:, 0]
    
    if root_dir:
        # Try to find the file with or without extension
        def find_file(fname):
            full_path = os.path.join(root_dir, fname)
            if os.path.exists(full_path):
                return full_path
            # Try common extensions
            for ext in ['.docx', '.png', '.jpg', '.jpeg', '.pdf', '.pdf.docx', '.pdf.png']:
                candidate = full_path + ext
                if os.path.exists(candidate):
                    return candidate
            # Return the original path (may fail later but that's ok)
            return full_path
        
        df['file_path'] = df['file_name'].apply(find_file)
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
