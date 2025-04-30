#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
import nltk
from nltk import word_tokenize
from collections import Counter
from sacrebleu.metrics import BLEU
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForMaskedLM
)
import textstat
from bert_score import score as bert_score
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
try:
    nltk.download('punkt')
except:
    pass

BASE_DIR = "/scratch/gpfs/de7281"
INTRINSICS_DIR = os.path.join(BASE_DIR, "intrinsics", "qa")
CSV_DIR = "scripts/fine_tune_2"
os.makedirs(INTRINSICS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-large")
xlmr_mlm = AutoModelForMaskedLM.from_pretrained(
    "xlm-roberta-large",
    ignore_mismatched_sizes=True
).to(device).eval()

# pseudo-perplexity
def pseudo_perplexity(text, stride=256):
    ids = xlmr_tok(text, return_tensors='pt', truncation=False, add_special_tokens=True).to(device).input_ids[0]
    L = ids.size(0)
    window = xlmr_tok.model_max_length
    nlls = []
    with torch.no_grad():
        for start in range(0, L, stride):
            end = min(start + window, L)
            chunk = ids[start:end].unsqueeze(0)
            seq_len = chunk.size(1)
            for i in range(1, seq_len-1):
                m = chunk.clone()
                m[0, i] = xlmr_tok.mask_token_id
                logits = xlmr_mlm(m).logits[0, i]
                logp = torch.log_softmax(logits, dim=-1)[chunk[0, i]]
                nlls.append(-logp)
            if end == L:
                break
    total_nll = torch.stack(nlls).sum()
    return torch.exp(total_nll / len(nlls)).item()

# summary stats
def compute_summary_stats(texts):
    toks = [word_tokenize(t) for t in texts]
    sent_lens = [len(s) for s in toks]
    all_toks = [w for s in toks for w in s]
    tok_lens = [len(w) for w in all_toks]
    vocab = set(all_toks)
    total = len(all_toks)
    top10 = Counter(all_toks).most_common(10)
    top_unigrams = [{"token": w, "count": c, "pct": c/total*100} for w, c in top10]
    return {
        "sent_mean_len": np.mean(sent_lens),
        "sent_std_len": np.std(sent_lens),
        "tok_mean_len": np.mean(tok_lens),
        "tok_std_len": np.std(tok_lens),
        "vocab_size": len(vocab),
        "ttr": len(vocab)/total if total else 0.0,
        "top_unigrams": top_unigrams
    }

# reference-free stats
def compute_ref_free(texts):
    bleu = BLEU(effective_order=True)
    joined = "\n".join(texts)
    flesch = textstat.flesch_reading_ease(joined)
    tokens = word_tokenize(joined.lower())
    distinct1 = len(set(tokens))/len(tokens) if tokens else 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    distinct2 = len(set(bigrams))/len(bigrams) if bigrams else 0.0
    div_scores = []
    for hyp in texts:
        refs = [r for r in texts if r != hyp]
        if refs:
            div_scores.append(bleu.sentence_score(hyp, refs).score/100.0)
    batch_diversity = 1.0 - np.mean(div_scores) if div_scores else 0.0
    P,R,F1 = bert_score(texts, texts, lang="en")
    enc = gpt2_tok(joined, return_tensors='pt').to(device)
    ids = enc.input_ids[0]
    max_len = gpt2_model.config.n_positions
    stride = max_len // 2
    nlls = []
    total_len = ids.size(0)
    with torch.no_grad():
        for start in range(0, total_len, stride):
            end = min(start + max_len, total_len)
            chunk = ids[start:end].unsqueeze(0)
            target = chunk.clone()
            target[:, :- (end-start)] = -100
            out = gpt2_model(chunk, labels=target)
            nlls.append(out.loss * (end-start))
            if end == total_len:
                break
    ppl = torch.exp(torch.stack(nlls).sum()/total_len).item()
    mlm_ppl = pseudo_perplexity(joined)
    return {
        "flesch": flesch,
        "distinct1": distinct1,
        "distinct2": distinct2,
        "batch_diversity": batch_diversity,
        "bert_f1": float(F1.mean()),
        "gpt2_ppl": ppl,
        "mlm_ppl": mlm_ppl
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    csv_path = os.path.join(CSV_DIR, args.file)
    variant = os.path.splitext(args.file)[0]
    out_path = os.path.join(INTRINSICS_DIR, f"final_stats_{variant}.csv")
    df = pd.read_csv(csv_path)
    texts = df['question'].astype(str).tolist()
    summary = compute_summary_stats(texts)
    ref_free = compute_ref_free(texts)
    row = {"variant": variant, **summary, **ref_free}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[info] saved stats to {out_path}")
