#!/usr/bin/env python
import os
import csv
import math
import sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

if __name__ == '__main__':
    OUT_DIR = '/scratch/gpfs/de7281/mcq_data'
    INP_FILE = os.path.join(OUT_DIR, 'english_mcq.csv')
    OUT_FILE = os.path.join(OUT_DIR, 'tigrinya_filtered_roberta.csv')
    THRESH = 19.19
    TI_COLS = [
        'translated_passage',
        'translated_question',
        'translated_mc_answer1',
        'translated_mc_answer2',
        'translated_mc_answer3',
        'translated_mc_answer4',
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache  = OUT_DIR
    tok    = AutoTokenizer.from_pretrained('xlm-roberta-large', cache_dir=cache)
    mlm    = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large', cache_dir=cache).to(device)
    mlm.eval()

    # pseudo-perplexity
    def pseudo_pppl(text: str) -> float:
        enc      = tok(text, return_tensors='pt', truncation=True).to(device)
        input_ids = enc.input_ids[0]
        mask_id   = tok.mask_token_id
        total     = 0.0
        seq_len   = input_ids.size(0)
        with torch.no_grad():
            for i in range(1, seq_len - 1):
                masked = input_ids.clone()
                masked[i] = mask_id
                logits = mlm(masked.unsqueeze(0)).logits[0, i]
                prob   = torch.softmax(logits, dim=-1)[ input_ids[i] ].item()
                total += math.log(prob + 1e-12)
        avg_nll = - total / (seq_len - 2)
        return math.exp(avg_nll)

    kept = dropped = 0
    with open(INP_FILE, newline='', encoding='utf-8') as fin, \
         open(OUT_FILE, 'w', newline='', encoding='utf-8') as fout:
        reader    = csv.DictReader(fin)
        fnames    = reader.fieldnames + ['ti_pppl']
        writer    = csv.DictWriter(fout, fieldnames=fnames)
        writer.writeheader()

        for row in reader:
            try:
                scores  = [pseudo_pppl(row[col].strip()) for col in TI_COLS]
                avg_pppl = sum(scores) / len(scores)
            except Exception as e:
                dropped += 1
                continue

            if avg_pppl <= THRESH:
                row['ti_pppl'] = f"{avg_pppl:.1f}"
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"filtered output to {OUT_FILE}")
