import os
import random
import torch
import csv
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
from datasets import load_dataset
import textstat
from bert_score import score as bert_score
from nltk import ngrams
from sacrebleu import corpus_bleu
import re
import html
gc = __import__('gc')

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# cleaning utility
def clean_text(text):
    t = text.strip().replace("quot;", "")
    t = html.unescape(t)
    t = re.sub(r"\s+", ' ', t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t

# lexical diversity
def distinct_n(text, n):
    toks = text.split()
    total = max(1, len(toks)-n+1)
    uniq = len(set(ngrams(toks, n)))
    return uniq/total

# batch diversity
def self_bleu(batch):
    scores = []
    for i, hyp in enumerate(batch):
        refs = batch[:i] + batch[i+1:]
        scores.append(corpus_bleu([hyp], [refs]).score/100)
    return scores

# semantic faithfulness

def bert_f1_score(batch):
    refs = gold_refs[:len(batch)]
    P, R, F1 = bert_score(batch, refs, lang='en', rescale_with_baseline=True)
    return F1.tolist()

# perplexity

def sentence_perplexity(text, tokenizer, model):
    enc = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
        ppl = torch.exp(out.loss)
    return ppl.item()

# seed-word check

def contains_seed_words(text, seeds, min_count):
    lc = text.lower()
    return sum(1 for w in seeds if w.lower() in lc) >= min_count

if __name__ == '__main__':
    set_seed()
    # config
    base = '/scratch/gpfs/de7281/sentiment_analysis/variant_5'
    os.makedirs(base, exist_ok=True)
    out_csv = os.path.join(base, 'variant7_raw.csv')
    exists = os.path.isfile(out_csv)
    fout = open(out_csv, 'a', newline='', encoding='utf-8')
    writer = csv.writer(fout)
    if not exists:
        writer.writerow(['sentiment','en_text'])

    HF = os.path.join('/scratch/gpfs/de7281','huggingface')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load LMs
    lm_tok = AutoTokenizer.from_pretrained('gpt2', cache_dir=HF)
    lm_mod = AutoModelForSeq2SeqLM.from_pretrained('gpt2', cache_dir=HF).to(device)
    gen_tok = AutoTokenizer.from_pretrained('google/flan-t5-large', cache_dir=HF)
    gen_mod = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', cache_dir=HF).to(device)
    sent_tok= AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=HF)
    sent_mod= AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=HF).to(device)
    classifier = pipeline('sentiment-analysis', model=sent_mod, tokenizer=sent_tok, device=0 if torch.cuda.is_available() else -1, batch_size=32, truncation=True, max_length=128)

    # load lexicon
    lex_data = load_dataset('google/smol','gatitos__en_ace',split='train')
    lexicon = [e['src'] for e in lex_data]

    # gold refs for BERTScore
    gold_refs = [
        # twenty example reference sentences...
    ]

    # thresholds
    NUM = 1000
    BATCH = 16
    MIN_LEX = 2
    SENT_MIN = 0.8
    FLESCH_MIN=50.0
    D1_MIN=0.5; D2_MIN=0.3; BDIV_MIN=0.3; BERT_MIN=0.85; PPL_MAX=200.0

    for label, guide in {'positive':'uplifting and cheerful','neutral':'calm and balanced','negative':'intense and somber'}.items():
        pbar, count = tqdm(total=NUM, desc=label), 0
        while count < NUM:
            sample = random.sample(lexicon, k=min(len(lexicon),MIN_LEX*3))
            prompt = f"generate a {label} english sentence that is {guide} and includes at least {MIN_LEX} of these words: {', '.join(sample)}."
            inputs=gen_tok(prompt,return_tensors='pt',truncation=True,max_length=512).to(device)
            with torch.no_grad():
                out = gen_mod.generate(**inputs, max_length=64, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=BATCH, no_repeat_ngram_size=3, repetition_penalty=1.3, decoder_start_token_id=gen_tok.pad_token_id)
            cands = [clean_text(s) for s in gen_tok.batch_decode(out,skip_special_tokens=True)]

            # metrics
            flesch = [textstat.flesch_reading_ease(s) for s in cands]
            d1 = [distinct_n(s,1) for s in cands]
            d2 = [distinct_n(s,2) for s in cands]
            bdiv = self_bleu(cands)
            bertf = bert_f1_score(cands)
            ppl_scores = [sentence_perplexity(s,lm_tok,lm_mod) for s in cands]
            res = classifier(cands)
            for i, s in enumerate(cands):
                lbl = {'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}[res[i][0]['label']]
                cond = [
                    lbl==label,
                    res[i][0]['score']>=SENT_MIN,
                    flesch[i]>=FLESCH_MIN,
                    d1[i]>=D1_MIN,
                    d2[i]>=D2_MIN,
                    (1-bdiv[i])>=0.7,
                    bertf[i]>=BERT_MIN,
                    ppl_scores[i]<=PPL_MAX,
                    contains_seed_words(s,lexicon,MIN_LEX)
                ]
                if all(cond):
                    writer.writerow([label,s]); fout.flush(); count+=1; pbar.update(1)
                    if count>=NUM: break
        pbar.close()
    fout.close()
