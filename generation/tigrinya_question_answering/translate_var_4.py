import os
import csv
import json
import html
import random
import asyncio
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
    AutoModelForMaskedLM,
    AutoModelForCausalLM
)
from googletrans import Translator

def set_seed(seed=100):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# async translation helper
async def translate_async(text, src='tir', dest='en'):
    tr = Translator()
    res = await tr.translate(text, src=src, dest=dest)
    return res.text.strip()

def translate(text):
    return asyncio.run(translate_async(text))

# QA agreement

def qa_preds(passage, question, qa1_pipe, qa2_pipe):
    a1 = qa1_pipe(question=question, context=passage)['answer'].lower()
    a2 = qa2_pipe(question=question, context=passage)['answer'].lower()
    return a1, a2

def qa_agree(true_ans, p1, p2):
    t = true_ans.lower()
    return (t in p1 or p1 in t) and (t in p2 or p2 in t)

# NLI entailment

def nli_entails(passage, question, ans, nli_model, nli_tokenizer, device, threshold=0.5):
    hypo = f"the answer to the question '{question}' is '{ans}'."
    enc = nli_tokenizer(passage[:512], hypo, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = ['CONTRADICTION','NEUTRAL','ENTAILMENT'][probs.argmax().item()]
    score = probs.max().item()
    return label == 'ENTAILMENT' and score >= threshold, label, score

# extract mcq fields

def extract_mcq_lines(text):
    fields = {}
    for line in text.splitlines():
        line = html.unescape(line).strip()
        if ':' not in line: continue
        key, val = line.split(':',1)
        k = key.strip().lower().replace(' ','_')
        if k in {'question','mc_answer1','mc_answer2','mc_answer3','mc_answer4','correct_answer_num'}:
            fields[k] = val.strip()
    required = {'question','mc_answer1','mc_answer2','mc_answer3','mc_answer4','correct_answer_num'}
    return fields if required.issubset(fields) else None

if __name__ == '__main__':
    set_seed()
    # config
    INP_CSV = 'tigrinya_mcqs.csv'
    OUT_CSV = 'tigrinya_mcqs_verified.csv'
    # load pipelines & models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # QA pipelines
    qa1_pipe = pipeline(
        'question-answering',
        model='distilbert-base-cased-distilled-squad',
        tokenizer='distilbert-base-cased-distilled-squad',
        device=0 if torch.cuda.is_available() else -1
    )
    qa2_pipe = pipeline(
        'question-answering',
        model='deepset/roberta-base-squad2',
        tokenizer='deepset/roberta-base-squad2',
        device=0 if torch.cuda.is_available() else -1
    )
    # NLI model & tokenizer
    from transformers import AutoTokenizer as NLI_Tok, AutoModelForSequenceClassification as NLI_Model
    nli_tokenizer = NLI_Tok.from_pretrained('facebook/bart-large-mnli')
    nli_model     = NLI_Model.from_pretrained('facebook/bart-large-mnli').to(device)

    # track processed
    seen = set()
    header = True
    if os.path.isfile(OUT_CSV):
        with open(OUT_CSV, newline='', encoding='utf-8') as prev:
            for r in csv.DictReader(prev): seen.add(r['tir_passage']+r['question_tir'])
        header = False

    # open files
    with open(INP_CSV, newline='', encoding='utf-8') as inf, \
         open(OUT_CSV, 'a' if not header else 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        fieldnames = [
            'question_number','en_passage','tir_passage',
            'question_tir','question_en',
            'mc_answer1_tir','mc_answer1_en',
            'mc_answer2_tir','mc_answer2_en',
            'mc_answer3_tir','mc_answer3_en',
            'mc_answer4_tir','mc_answer4_en',
            'correct_answer_num'
        ]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        if header: writer.writeheader()

        pbar = tqdm(reader, desc='translate & verify')
        count = sum(1 for _ in csv.DictReader(open(OUT_CSV))) if not header else 0
        for row in pbar:
            key = row['tir_passage']+row['question_tir']
            if key in seen: continue
            # translate question and answers
            q_tir = row['question_tir']
            question_en = translate(q_tir)
            ans_tir   = row[f"mc_answer{row['correct_answer_num']}_tir"]
            ans_en    = translate(ans_tir)
            # translate all options
            opts_en = [translate(row[f"mc_answer{i}_tir"]) for i in range(1,5)]
            # QA agreement
            en_pass = row['en_passage']
            p1, p2 = qa_preds(en_pass, question_en, qa1_pipe, qa2_pipe)
            if not qa_agree(ans_en, p1, p2): continue
            # NLI entailment
            ok, lbl, score = nli_entails(en_pass, question_en, ans_en, nli_model, nli_tokenizer, device)
            if not ok: continue
            # write
            count += 1
            out = {
                'question_number': count,
                'en_passage': row['en_passage'],
                'tir_passage': row['tir_passage'],
                'question_tir': q_tir,
                'question_en': question_en,
                'correct_answer_num': row['correct_answer_num']
            }
            for i in range(1,5):
                out[f'mc_answer{i}_tir'] = row[f'mc_answer{i}_tir']
                out[f'mc_answer{i}_en']  = opts_en[i-1]
            writer.writerow(out)
            seen.add(key)
        pbar.close()
    print(f"saved {count} verified MCQs to {OUT_CSV}")