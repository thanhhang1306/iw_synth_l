import os
import json
import html
import csv
import sys
import random
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# extract mcq fields from generated text
def extract_mcq_lines(text):
    fields = {}
    for line in text.splitlines():
        line = html.unescape(line).strip()
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        k = key.strip().lower().replace(' ', '_')
        if k in {
            'question',
            'mc_answer1',
            'mc_answer2',
            'mc_answer3',
            'mc_answer4',
            'correct_answer_num'
        }:
            fields[k] = val.strip()
    required = {
        'question',
        'mc_answer1',
        'mc_answer2',
        'mc_answer3',
        'mc_answer4',
        'correct_answer_num'
    }
    return fields if required.issubset(fields) else None

def iterative_fill(prompt: str, mask_token: str, fill_pipeline) -> str:
    text = prompt
    # Count masks remaining
    while mask_token in text:
        # pipeline returns a list of predictions for the first mask
        out = fill_pipeline(text)
        # take the top prediction's full sequence
        text = out[0]['sequence']
    return text

if __name__ == '__main__':
    set_seed()

    # config
    passages_file = 'scripts/generation_q_a/passage_tir.fixed.jsonl'
    CSV_FILE      = 'tigrinya_mcqs.csv'
    model_id      = 'EthioNLP/EthioLLM-l-250K'
    num_questions = 500

    # prepare output file
    os.makedirs(os.path.dirname(CSV_FILE) or '.', exist_ok=True)
    out_f = open(CSV_FILE, 'w', newline='', encoding='utf-8')
    writer = csv.writer(out_f)
    writer.writerow([
        'question_number',
        'en_passage',
        'tir_passage',
        'question_tir',
        'mc_answer1_tir',
        'mc_answer2_tir',
        'mc_answer3_tir',
        'mc_answer4_tir',
        'correct_answer_num'
    ])

    # load model + tokenizer + fill-mask pipeline
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(
        torch.device('cuda' if device == 0 else 'cpu')
    )
    # ensure mask token exists
    if tok.mask_token is None:
        raise ValueError("tokenizer has no mask token")
    fill = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tok,
        device=device
    )

    # read passages
    entries = []
    with open(passages_file, encoding='utf-8') as fin:
        for line in fin:
            entries.append(json.loads(line))

    # generation loop
    count = 0
    pbar = tqdm(total=num_questions, desc='generating tir mcqs')
    for entry in entries:
        if count >= num_questions:
            break
        en_pass = entry.get('en')
        ti_pass = entry.get('ti')

        # build a prompt with real <mask> tokens
        M = tok.mask_token
        prompt = (
            f"Given the Tigrinya passage: {ti_pass}\n"
            "Generate a multiple-choice question in Tigrinya with four options.\n"
            f"Question: {M} {M}\n"
            f"MC Answer1: {M} {M}\n"
            f"MC Answer2: {M} {M}\n"
            f"MC Answer3: {M} {M}\n"
            f"MC Answer4: {M} {M}\n"
            f"Correct answer num: {M}"
        )

        # iteratively fill all masks
        filled = iterative_fill(prompt, tok.mask_token, fill)

        # extract only the MCQ block
        mcq = extract_mcq_lines(filled)
        if mcq:
            count += 1
            writer.writerow([
                count,
                en_pass,
                ti_pass,
                mcq['question'],
                mcq['mc_answer1'],
                mcq['mc_answer2'],
                mcq['mc_answer3'],
                mcq['mc_answer4'],
                mcq['correct_answer_num'],
            ])
            pbar.update(1)

    pbar.close()
    out_f.close()
    print(f"saved {count} mcqs to {CSV_FILE}")
