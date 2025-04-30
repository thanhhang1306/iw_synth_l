import os
import random
import time
import math
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer as HF_AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import spacy
from nltk import ngrams
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
import textstat
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score

# seed for reproducibility
def set_seed(seed: int = 100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# diversity-aware buckets
def load_title_buckets(max_per_title: int):
    ds = load_dataset('squad', split='train', cache_dir=HF_CACHE)
    buckets = {}
    for ex in ds:
        title = ex['title'].replace(' ', '_')
        lst = buckets.setdefault(title, [])
        if len(lst) < max_per_title:
            lst.append(ex['context'])
    return buckets

def sample_passage():
    remaining = [t for t, c in bucket_counts.items()
                 if c < TARGET // len(bucket_counts) + 1]
    title = random.choice(remaining)
    return title, random.choice(TITLE_BUCKETS[title])

# NER / noun-chunk span picker
nlp = spacy.load('en_core_web_sm', disable=['textcat'])
def pick_answer_span(text: str) -> str:
    doc = nlp(text[:400])
    ents = [e.text for e in doc.ents if len(e.text.split()) <= 5]
    if ents:
        return random.choice(ents)
    chunks = [nc.text for nc in doc.noun_chunks if len(nc.text.split()) <= 5]
    return random.choice(chunks) if chunks else text.split()[0]

# build distractors using entities + WordNet
embedder = SentenceTransformer('all-mpnet-base-v2', cache_folder=HF_CACHE)
def build_distractors(ans: str, passage: str) -> list[str]:
    ents = [e.text for e in nlp(passage).ents
            if e.text.lower() != ans.lower() and len(e.text.split()) <= 5]
    wn_alts = [l.name().replace('_',' ') for syn in wn.synsets(ans)
               for l in syn.hyponyms()[:3]]
    corpus = list(dict.fromkeys(ents + wn_alts))[:10]
    if not corpus:
        return ['—'] * 3
    sims = util.cos_sim(embedder.encode([ans]), embedder.encode(corpus))[0]
    idxs = sims.argsort(descending=False)[:3]
    return [corpus[i] for i in idxs]

# QA verification pipelines
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qa1 = pipeline(
    'question-answering',
    model='distilbert-base-cased-distilled-squad',
    tokenizer='distilbert-base-cased-distilled-squad',
    device=0 if torch.cuda.is_available() else -1
)
qa2 = pipeline(
    'question-answering',
    model='deepset/roberta-base-squad2',
    tokenizer='deepset/roberta-base-squad2',
    device=0 if torch.cuda.is_available() else -1
)
def qa_preds(passage: str, question: str) -> tuple[str, str]:
    return (
        qa1(question=question, context=passage)['answer'].lower(),
        qa2(question=question, context=passage)['answer'].lower()
    )
def qa_agree(ans: str, p1: str, p2: str) -> bool:
    return ((ans.lower() in p1 or p1 in ans.lower()) and
            (ans.lower() in p2 or p2 in ans.lower()))

# NLI entailment
nli_tokenizer = HF_AutoTokenizer.from_pretrained(
    'facebook/bart-large-mnli', cache_dir=HF_CACHE
)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    'facebook/bart-large-mnli', cache_dir=HF_CACHE
).to(DEVICE)
def nli_entails(passage: str, question: str, ans: str, th: float = 0.5) -> bool:
    hypo = f"the answer to the question '{question}' is '{ans}'."
    enc = nli_tokenizer(
        passage[:512], hypo,
        return_tensors='pt', padding=True, truncation=True
    ).to(DEVICE)
    logits = nli_model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return (probs.argmax().item() == 2 and probs[2].item() >= th)

# QA generation with chain-of-thought prompt
qg_tokenizer = T5Tokenizer.from_pretrained(
    'valhalla/t5-base-qg-hl', cache_dir=HF_CACHE
)
qg_model = AutoModelForSeq2SeqLM.from_pretrained(
    'valhalla/t5-base-qg-hl', cache_dir=HF_CACHE
).to(DEVICE)

# intrinsic naturalness thresholds
TH_FLESCH   = 107.01
TH_DIST1    = 0.90
TH_DIST2    = 1.00
TH_COS      = 0.85
TH_PPL_GPT2 = 10.72
TH_BDIV     = 0.90
TH_BERT_F1  = 1.00
TH_PPL_MLM  = 19.19
BATCH_SIZE  = 3
MAX_REFS    = 2

# metrics functions
def flesch_score(text: str) -> float:
    return textstat.flesch_reading_ease(text)
def distinct_n_score(text: str, n: int) -> float:
    toks = text.split()
    return len(set(ngrams(toks, n))) / max(1, len(toks)-n+1)
def batch_diversity(batch: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    scores = []
    for i, c in enumerate(batch):
        refs = [b for j,b in enumerate(batch) if j != i]
        scores.append(bleu.corpus_score([c], [refs]).score/100)
    return 1 - sum(scores)/len(scores)
def bert_f1(cand: str, refs: list[str]) -> float:
    P, R, F1 = bert_score([cand], refs, lang='en', rescale_with_baseline=True)
    return F1.mean().item()

# load GPT-2 & MLM for perplexity
gpt2_tok = HF_AutoTokenizer.from_pretrained('gpt2', cache_dir=HF_CACHE)
gpt2_mod = AutoModelForSeq2SeqLM.from_pretrained('gpt2', cache_dir=HF_CACHE).to(DEVICE)
mlm_tok  = HF_AutoTokenizer.from_pretrained('roberta-base', cache_dir=HF_CACHE)
mlm_mod  = AutoModelForSequenceClassification.from_pretrained('roberta-base', cache_dir=HF_CACHE).to(DEVICE)

def ppl_gpt2(text: str) -> float:
    ids = gpt2_tok(text, return_tensors='pt').input_ids.to(DEVICE)
    with torch.no_grad():
        loss = gpt2_mod(ids, labels=ids).loss
    return torch.exp(loss).item()

def ppl_mlm(text: str) -> float:
    toks = mlm_tok.tokenize(text)
    ids = mlm_tok.convert_tokens_to_ids(toks)
    total_nll = 0.0
    for i, tid in enumerate(ids):
        masked = ids.copy()
        masked[i] = mlm_tok.mask_token_id
        inp = torch.tensor([masked]).to(DEVICE)
        with torch.no_grad(): logits = mlm_mod(inp).logits
        logp = torch.log_softmax(logits[0,i], dim=-1)
        total_nll += -logp[tid].item()
    return math.exp(total_nll/max(1, len(ids)))

# single MCQ generation + filtering
def generate_mcq(passage: str) -> dict | None:
    ans = pick_answer_span(passage)
    refs = CONTEXT_QUESTIONS.get(passage, [])[:MAX_REFS]
    batch = []
    for _ in range(BATCH_SIZE):
        prompt = (
            "you are a helpful assistant. let's think step by step: "
            f"1) identify the answer span '{ans}' in the passage. "
            "2) formulate a clear multiple-choice question about that span. "
            "3) provide four options including the correct answer. "
            "finally, output only the question and options. "
            f"passage: {passage}"
        )
        ids = qg_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        out = qg_model.generate(**ids, max_length=80, do_sample=True, top_p=0.9, no_repeat_ngram_size=2)
        batch.append(qg_tokenizer.decode(out[0], skip_special_tokens=True).strip())
    # apply naturalness filters
    valid = [q for q in batch if (
        flesch_score(q) >= TH_FLESCH and
        distinct_n_score(q,1) >= TH_DIST1 and
        distinct_n_score(q,2) >= TH_DIST2 and
        util.cos_sim(embedder.encode([passage]), embedder.encode([q]))[0,1].item() >= TH_COS and
        ppl_gpt2(q) <= TH_PPL_GPT2 and
        batch_diversity(batch) >= TH_BDIV and
        (not refs or bert_f1(q, refs) >= TH_BERT_F1) and
        ppl_mlm(q) <= TH_PPL_MLM
    )]
    if not valid:
        return None
    question = valid[0]
    # factuality checks
    if not qa_agree(ans, *qa_preds(passage, question)):
        return None
    if not nli_entails(passage, question, ans):
        return None
    opts = [ans] + build_distractors(ans, passage)
    random.shuffle(opts)
    return {
        'passage': passage,
        'question_number': 0,
        'question': question,
        'mc_answer1': opts[0],
        'mc_answer2': opts[1],
        'mc_answer3': opts[2],
        'mc_answer4': opts[3],
        'correct_answer_num': opts.index(ans) + 1
    }

if __name__ == '__main__':
    set_seed()
    HF_CACHE = '/scratch/gpfs/de7281/huggingface'
    TARGET   = 500
    MAX_BUCKET = 10
    OUTPUT_DIR = os.path.join(HF_CACHE, '..', 'new_data', 'mcq_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # define output CSV
    CSV_FILE = os.path.join(OUTPUT_DIR, 'variant_8_mcq.csv')

    # load buckets and questions
    TITLE_BUCKETS = load_title_buckets(MAX_BUCKET)
    bucket_counts = {t: 0 for t in TITLE_BUCKETS}
    CONTEXT_QUESTIONS = {}
    for ex in load_dataset('squad', split='train', cache_dir=HF_CACHE):
        CONTEXT_QUESTIONS.setdefault(ex['context'], []).append(ex['question'])

    results, loops = [], 0
    start = time.time()
    while len(results) < TARGET and loops < len(TITLE_BUCKETS) * TARGET:
        loops += 1
        title, passage = sample_passage()
        mcq = generate_mcq(passage)
        if mcq:
            bucket_counts[title] += 1
            mcq['question_number'] = len(results) + 1
            mcq['source_title'] = title
            results.append(mcq)
    # save
    pd.DataFrame(results).to_csv(CSV_FILE, index=False)
    set_seed()
    HF_CACHE = '/scratch/gpfs/de7281/huggingface'
    TARGET   = 500
    MAX_BUCKET = 10
    OUTPUT_DIR = os.path.join(HF_CACHE, '..', 'new_data', 'mcq_data')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # load buckets and questions
    TITLE_BUCKETS = load_title_buckets(MAX_BUCKET)
    bucket_counts = {t: 0 for t in TITLE_BUCKETS}
    CONTEXT_QUESTIONS = {}
    for ex in load_dataset('squad', split='train', cache_dir=HF_CACHE):
        CONTEXT_QUESTIONS.setdefault(ex['context'], []).append(ex['question'])
    # define output CSV
    CSV_FILE = os.path.join(OUTPUT_DIR, 'variant9_10_mcq.csv')

    results, loops = [], 0
    start = time.time()
    while len(results) < TARGET and loops < len(TITLE_BUCKETS) * TARGET:
        loops += 1
        title, passage = sample_passage()
        mcq = generate_mcq(passage)
        if mcq:
            bucket_counts[title] += 1
            mcq['question_number'] = len(results) + 1
            mcq['source_title'] = title
            results.append(mcq)
    # save
    pd.DataFrame(results).to_csv(CSV_FILE, index=False)
    print(f'saved {len(results)} mcqs to {CSV_FILE}')
