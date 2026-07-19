import pandas as pd
import argparse

FIELDS = ['Agreement Value', 'Agreement Start Date', 'Agreement End Date', 'Renewal Notice (Days)', 'Party One', 'Party Two']


def compute_recall(gold_csv, pred_csv):
    g = pd.read_csv(gold_csv)
    p = pd.read_csv(pred_csv)
    # align by file_name
    merged = g.merge(p, on='file_name', suffixes=('_gold', '_pred'))
    stats = {}
    for f in FIELDS:
        gold_col = f + '_gold'
        pred_col = f + '_pred'
        if gold_col not in merged.columns or pred_col not in merged.columns:
            stats[f] = {'true': 0, 'false': len(merged)}
            continue
        true = ((merged[gold_col].fillna('').astype(str).str.strip()) == (merged[pred_col].fillna('').astype(str).str.strip())).sum()
        false = len(merged) - true
        recall = true / (true + false) if (true + false) > 0 else 0.0
        stats[f] = {'true': int(true), 'false': int(false), 'recall': float(recall)}
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', required=True)
    parser.add_argument('--pred', required=True)
    args = parser.parse_args()
    stats = compute_recall(args.gold, args.pred)
    for k, v in stats.items():
        print(f"{k}: true={v['true']} false={v['false']} recall={v['recall']:.4f}")
