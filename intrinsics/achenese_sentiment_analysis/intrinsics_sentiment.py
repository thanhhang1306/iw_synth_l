#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import torch
from collections import Counter
from nltk import word_tokenize
from sacrebleu.metrics import BLEU
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForMaskedLM
)
import textstat
from bert_score import score as bert_score

# pseudo-perplexity
def pseudo_perplexity(text, tok, model, device, stride=256):
    tokens = tok(text, return_tensors='pt', truncation=False, add_special_tokens=True).to(device).input_ids[0]
    L = tokens.size(0)
    window = tok.model_max_length
    nlls = []
    with torch.no_grad():
        for start in range(0, L, stride):
            end = min(start + window, L)
            window_ids = tokens[start:end].unsqueeze(0)
            seq_len = window_ids.size(1)
            for i in range(1, seq_len - 1):
                masked = window_ids.clone()
                masked[0, i] = tok.mask_token_id
                logits = model(masked).logits[0, i]
                logp = torch.log_softmax(logits, dim=-1)[window_ids[0, i]]
                nlls.append(-logp)
            if end == L:
                break
    total_nll = torch.stack(nlls).sum()
    return torch.exp(total_nll / len(nlls)).item()

# summary stats
def compute_summary_stats(texts):
    toks = [word_tokenize(t) for t in texts]
    sent_lens = [len(s) for s in toks]
    all_toks = [tok for sent in toks for tok in sent]
    tok_lens = [len(t) for t in all_toks]
    vocab = set(all_toks)
    total = len(all_toks)
    ttr = len(vocab) / total if total else 0.0
    top10 = Counter(all_toks).most_common(10)
    top_unigrams = [
        {"token": w, "count": c, "pct": c / total * 100}
        for w, c in top10
    ]
    return {
        "sent_mean_len": np.mean(sent_lens),
        "sent_std_len": np.std(sent_lens),
        "tok_mean_len": np.mean(tok_lens),
        "tok_std_len": np.std(tok_lens),
        "vocab_size": len(vocab),
        "ttr": ttr,
        "top_unigrams": top_unigrams
    }

# reference-free stats 
def compute_ref_free(texts, gpt2_tok, gpt2_model, xlmr_tok, xlmr_mlm, device):
    bleu = BLEU(effective_order=True)
    joined = "\n".join(texts)
    flesch = textstat.flesch_reading_ease(joined)
    tokens = word_tokenize(joined.lower())
    distinct1 = len(set(tokens)) / len(tokens) if tokens else 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    distinct2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    refs = texts.copy()
    div_scores = []
    for hyp in texts:
        others = [r for r in refs if r != hyp]
        if others:
            div_scores.append(bleu.sentence_score(hyp, others).score / 100.0)
    batch_diversity = 1.0 - np.mean(div_scores) if div_scores else 0.0

    P, R, F1 = bert_score(texts, texts, lang="en")

    enc = gpt2_tok(joined, return_tensors="pt").to(device)
    input_ids = enc.input_ids[0]
    doc_stride = gpt2_model.config.n_positions // 2
    max_len = gpt2_model.config.n_positions
    nlls = []
    total_len = input_ids.size(0)
    with torch.no_grad():
        for start in range(0, total_len, doc_stride):
            end = min(start + max_len, total_len)
            slice_ids = input_ids[start:end].unsqueeze(0)
            target = slice_ids.clone()
            target[:, :- (end - start)] = -100
            out = gpt2_model(slice_ids, labels=target)
            nlls.append(out.loss * (end - start))
            if end == total_len:
                break
    gpt2_ppl = torch.exp(torch.stack(nlls).sum() / total_len).item()
    mlm_ppl = pseudo_perplexity(joined, xlmr_tok, xlmr_mlm, device)

    return {
        "flesch": flesch,
        "distinct1": distinct1,
        "distinct2": distinct2,
        "batch_diversity": batch_diversity,
        "bert_f1": float(F1.mean()),
        "gpt2_ppl": gpt2_ppl,
        "mlm_ppl": mlm_ppl
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()

    base = "/scratch/gpfs/de7281"
    intrinsics_dir = os.path.join(base, "intrinsics", "sentiment")
    csv_dir = "scripts/fine_tune"
    csv_path = os.path.join(csv_dir, args.file)
    variant = os.path.splitext(args.file)[0]
    out_path = os.path.join(intrinsics_dir, f"final_stats_{variant}.csv")
    os.makedirs(intrinsics_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-large")
    xlmr_mlm = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large", ignore_mismatched_sizes=True).to(device).eval()

    df = pd.read_csv(csv_path)
    ace = df.get("ace_text", df.get("flores_passage")).astype(str).tolist()
    en  = df.get("en_text", df.get("question")).astype(str).tolist()

    summary = compute_summary_stats(ace)
    ref_free = compute_ref_free(en, gpt2_tok, gpt2_model, xlmr_tok, xlmr_mlm, device)

    row = {"variant": variant, **summary, **ref_free}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[info] saved stats to {out_path}")
