import os
import argparse
import hashlib
import json
import re
from tqdm import tqdm
import pandas as pd

from src.ocr import extract_text

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
from datasets import Dataset


FIELD_MAP = {
    'Aggrement Value': 'Agreement Value',
    'Aggrement Start Date': 'Agreement Start Date',
    'Aggrement End Date': 'Agreement End Date',
    'Renewal Notice (Days)': 'Renewal Notice (Days)',
    'Party One': 'Party One',
    'Party Two': 'Party Two'
}


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


def find_answer_span(context, answer):
    if answer is None:
        return None
    ans = str(answer).strip()
    if ans == '' or ans.lower() in ['nan', 'none']:
        return None
    # try direct find
    idx = context.find(ans)
    if idx != -1:
        return idx, idx + len(ans)
    # try case-insensitive
    idx = context.lower().find(ans.lower())
    if idx != -1:
        return idx, idx + len(ans)
    # try relaxed whitespace/punct normalization
    def normalize(s):
        return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z]", "", s)).lower()
    nctx = normalize(context)
    nans = normalize(ans)
    idx = nctx.find(nans)
    if idx != -1:
        # map back to original approximate position by searching chunks
        # fallback: return None (we skip hard samples)
        return None
    return None


def build_squad_examples(train_csv, train_dir):
    df = pd.read_csv(train_csv)
    contexts = []
    questions = []
    answers = []
    ids = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row.get('File Name') or row.get('File Name'.strip()) or row.get('File Name')
        # many CSVs use 'File Name' header; if not, try first column
        if file_name is None:
            file_name = row.iloc[0]
        file_path = os.path.join(train_dir, file_name)
        if not os.path.exists(file_path):
            # try with extensions
            possible = None
            for ext in ['.docx', '.png', '.jpg']:
                p = file_path + ext
                if os.path.exists(p):
                    possible = p
                    break
            if possible:
                file_path = possible
            else:
                continue
        context = _cache_text(file_path)
        for csv_key, field_name in FIELD_MAP.items():
            gold = row.get(csv_key)
            span = find_answer_span(context, gold)
            if span is None:
                # skip if cannot locate span
                continue
            start, end = span
            contexts.append(context)
            questions.append(field_name)
            answers.append({'text': [str(gold)], 'answer_start': [int(start)]})
            ids.append(f"{file_name}__{field_name}")
    data = {'id': ids, 'context': contexts, 'question': questions, 'answers': answers}
    return Dataset.from_dict(data)


def prepare_features(examples, tokenizer, max_length=384, doc_stride=128):
    # tokenization similar to run_qa examples: return tokenized features with start/end positions
    tokenized = tokenizer(examples['question'], examples['context'], truncation='only_second', max_length=max_length, stride=doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    sample_mapping = tokenized.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized.pop('offset_mapping')

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        if len(answers['answer_start']) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])
            sequence_ids = tokenized.sequence_ids(i)
            # find token start and end
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # detect if answer is out of span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # find exact token indexes
                token_start = token_start_index
                while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                    token_start += 1
                token_start -= 1
                token_end = token_end_index
                while token_end > 0 and offsets[token_end][1] >= end_char:
                    token_end -= 1
                token_end += 1
                start_positions.append(token_start)
                end_positions.append(token_end)
    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model', default='distilbert-base-uncased-distilled-squad')
    parser.add_argument('--out', default='outputs/qa-model')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    ds = build_squad_examples(args.csv, args.data_dir)
    if len(ds) == 0:
        print('No training examples were found. Aborting.')
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = ds.map(lambda ex: prepare_features(ex, tokenizer), batched=True, remove_columns=ds.column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=args.out,
        evaluation_strategy='no',
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    trainer.save_model(args.out)
    print('Model saved to', args.out)


if __name__ == '__main__':
    main()
