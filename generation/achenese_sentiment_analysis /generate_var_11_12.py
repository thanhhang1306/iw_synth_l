import os
import random
import torch
import csv
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
import textstat
from bert_score import score as bert_score
from nltk import ngrams
from sacrebleu import corpus_bleu
import re
import html

def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# readability metric
def flesch_score(text):
    return textstat.flesch_reading_ease(text)

# lexical diversity
def distinct_n(text, n):
    toks = text.split()
    total = max(1, len(toks) - n + 1)
    return len(set(ngrams(toks, n))) / total

# batch diversity
def self_bleu(batch):
    scores = []
    for i, hyp in enumerate(batch):
        refs = batch[:i] + batch[i+1:]
        scores.append(corpus_bleu([hyp], [refs]).score / 100)
    return scores

# semantic faithfulness
def bert_f1_score(batch):
    refs = gold_refs[: len(batch)]
    _, _, f1 = bert_score(batch, refs, lang='en', rescale_with_baseline=True)
    return f1.tolist()

# perplexity 
def sentence_perplexity(text, tokenizer, model):
    enc = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=enc['input_ids'])
        return torch.exp(out.loss).item()

# text cleanup
def clean_text(text):
    t = text.strip().replace("quot;", "")
    t = html.unescape(t)
    t = re.sub(r"\s+", ' ', t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t

# seed-word check
def contains_seed_words(text, seeds, min_count):
    lc = text.lower()
    return sum(1 for w in seeds if w.lower() in lc) >= min_count

# chain-of-thought generation + QA filtering

def generate_and_filter(label, guide, BATCH, gen_tok, gen_mod, sent_pipe, lexicon, lm_tok, lm_mod):
    sample = random.sample(lexicon, k=min(len(lexicon), MIN_LEX * 3))
    prompt = (
        f"you are a helpful assistant. think step by step to craft a {label} english sentence that is {guide}"
        f" and includes at least {MIN_LEX} of these words: {', '.join(sample)}. then provide only the final sentence."
    )
    inputs = gen_tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outs = gen_mod.generate(
            **inputs,
            max_length=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=BATCH,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            decoder_start_token_id=gen_tok.pad_token_id
        )
    cands = [clean_text(s) for s in gen_tok.batch_decode(outs, skip_special_tokens=True)]

    flesch = [flesch_score(s) for s in cands]
    d1 = [distinct_n(s, 1) for s in cands]
    d2 = [distinct_n(s, 2) for s in cands]
    bdiv = self_bleu(cands)
    bertf = bert_f1_score(cands)
    ppl = [sentence_perplexity(s, lm_tok, lm_mod) for s in cands]

    survivors = []
    for i, s in enumerate(cands):
        res = sent_pipe([s])[0]
        pred = LABEL_MAP[res['label']]
        score = res['score']
        cond = [
            pred == label,
            score >= SENT_MIN,
            flesch[i] >= FLESCH_MIN,
            d1[i] >= D1_MIN,
            d2[i] >= D2_MIN,
            (1 - bdiv[i]) >= COHERENCE_MIN,
            bertf[i] >= BERT_MIN,
            ppl[i] <= PPL_MAX,
            contains_seed_words(s, lexicon, MIN_LEX)
        ]
        if all(cond):
            survivors.append(s)
    return survivors

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HF = '/scratch/gpfs/de7281/huggingface'

    gen_tok = AutoTokenizer.from_pretrained('google/flan-t5-large', cache_dir=HF)
    gen_mod = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', cache_dir=HF).to(device)
    sent_tok = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=HF)
    sent_mod = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=HF).to(device)
    sent_pipe = pipeline(
        'sentiment-analysis', model=sent_mod, tokenizer=sent_tok,
        device=0 if torch.cuda.is_available() else -1,
        BATCH=32, truncation=True, max_length=128, top_k=1
    )

    # cross-reuse
    lm_tok = AutoTokenizer.from_pretrained(LM_MODEL, cache_dir=HF)
    lm_mod = AutoModelForCausalLM.from_pretrained(LM_MODEL, cache_dir=HF).to(device)

    lex_data = load_dataset('google/smol', 'gatitos__en_ace', split='train')
    lexicon = [e['src'] for e in lex_data]

    # parameters
    NUM = 1000
    BATCH = 16
    MIN_LEX = 2
    SENT_MIN = 0.8
    FLESCH_MIN=51.48
    D1_MIN=0.9
    D2_MIN=1.0
    BDIV_MIN=0.99
    COHERENCE_MIN = 0.7
    BERT_MIN=1.00
    PPL_MAX=148.74
    

    OUT_DIR = '/scratch/gpfs/de7281/sentiment_analysis/variant7_data'
    os.makedirs(OUT_DIR, exist_ok=True)
    CSV_FILE = os.path.join(OUT_DIR, 'variant11_raw.csv')
    exists  = os.path.isfile(CSV_FILE)
    fout    = open(CSV_FILE, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(fout)
    if not exists:
        writer.writerow(['sentiment','en_text'])

    for label, guide in {'positive':'uplifting and cheerful', 'neutral':'calm and balanced', 'negative':'intense and somber'}.items():
        pbar, count = tqdm(total=NUM, desc=label), 0
        while count < NUM:
            keep = generate_and_filter(label, guide, BATCH, gen_tok, gen_mod, sent_pipe, lexicon, lm_tok, lm_mod)
            for s in keep:
                writer.writerow([label, s])
                fout.flush()
                count += 1
                pbar.update(1)
                if count >= NUM:
                    break
        pbar.close()
    fout.close()
