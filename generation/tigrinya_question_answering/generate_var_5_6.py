import os
import random
import time
import sys
import torch
import pandas as pd
from datasets import load_dataset
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, pipeline
import spacy
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# diversity-aware passage buckets
def load_title_buckets(max_per_title):
    ds = load_dataset('squad', split='train', cache_dir=HF_CACHE)
    buckets = {}
    for ex in ds:
        title = ex['title'].replace(' ', '_')
        lst = buckets.setdefault(title, [])
        if len(lst) < max_per_title:
            lst.append(ex['context'])
    return buckets

# sample uniformly under-served buckets
def sample_passage():
    remaining = [t for t,c in bucket_counts.items() if c < TARGET//len(bucket_counts)+1]
    title = random.choice(remaining)
    return title, random.choice(TITLE_BUCKETS[title])

# pick short answer span
nlp = spacy.load('en_core_web_sm', disable=['textcat'])
def pick_answer_span(text):
    doc = nlp(text[:400])
    ents = [e.text for e in doc.ents if len(e.text.split()) <= 5]
    if ents:
        return random.choice(ents)
    nps = [nc.text for nc in doc.noun_chunks if len(nc.text.split()) <= 5]
    return random.choice(nps) if nps else text.split()[0]

# build distractors via entities + wordnet
embedder = SentenceTransformer('all-mpnet-base-v2', cache_folder=HF_CACHE)
def build_distractors(ans, passage):
    ents = [e.text for e in nlp(passage).ents if e.text.lower() != ans.lower() and len(e.text.split()) <= 5]
    wn_alts = [l.name().replace('_',' ') for syn in wn.synsets(ans) for l in syn.hyponyms()[:3]]
    corpus = list(dict.fromkeys(ents + wn_alts))[:10]
    if not corpus:
        return ['—','—','—']
    sims = util.cos_sim(embedder.encode([ans]), embedder.encode(corpus))[0]
    idxs = sims.argsort(descending=False)[:3]
    return [corpus[i] for i in idxs]

# QA model predictions
qa1_pipe = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad', device=0 if torch.cuda.is_available() else -1)
qa2_pipe = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2', device=0 if torch.cuda.is_available() else -1)
def qa_preds(passage, question):
    return qa1_pipe(question=question, context=passage)['answer'].lower(), qa2_pipe(question=question, context=passage)['answer'].lower()

def qa_agree(ans, p1, p2):
    return (ans.lower() in p1 or p1 in ans.lower()) and (ans.lower() in p2 or p2 in ans.lower())

# NLI entailment check
from transformers import AutoTokenizer, AutoModelForSequenceClassification
nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli', cache_dir=HF_CACHE)
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', cache_dir=HF_CACHE).to(device)
def nli_entails(passage, question, ans, th=0.5):
    hypo = f"the answer to the question '{question}' is '{ans}'."
    enc = nli_tokenizer(passage[:512], hypo, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = nli_model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    # contradiction, neutral, entailment
    label = ['CONTRADICTION','NEUTRAL','ENTAILMENT'][probs.argmax().item()]
    score = probs.max().item()
    return label=='ENTAILMENT' and score>=th

# generate one MCQ
qg_tokenizer = T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=HF_CACHE)
qg_model = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=HF_CACHE).to(device)
def generate_question(passage):
    ans = pick_answer_span(passage)
    prompt = f"generate question: <hl> {ans} <hl> {passage} </s>"
    ids = qg_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    out = qg_model.generate(**ids, max_length=64, do_sample=True, top_p=0.9)
    question = qg_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    options = [ans] + build_distractors(ans, passage)
    random.shuffle(options)
    p1, p2 = qa_preds(passage, question)
    if not qa_agree(ans, p1, p2):
        return None
    if not nli_entails(passage, question, ans):
        return None
    return {'passage': passage, 'question': question, 'mc_answer1': options[0], 'mc_answer2': options[1], 'mc_answer3': options[2], 'mc_answer4': options[3], 'correct_answer_num': options.index(ans)+1}

if __name__ == '__main__':
    set_seed()
    # config
    HF_CACHE = '/scratch/gpfs/de7281/huggingface'
    TARGET   = 500
    MAX_BUCKET = 10
    BASE_DIR = '/scratch/gpfs/de7281/new_data/mcq_data'
    os.makedirs(BASE_DIR, exist_ok=True)
    CSV_FILE = os.path.join(BASE_DIR, 'variant_5_mcq.csv')

    # prepare buckets
    TITLE_BUCKETS = load_title_buckets(MAX_BUCKET)
    bucket_counts = {t:0 for t in TITLE_BUCKETS}

    # generate loop
    res, loops = [], 0
    while len(res) < TARGET and loops < len(TITLE_BUCKETS)*TARGET:
        loops += 1
        _, passage = sample_passage()
        mcq = generate_question(passage)
        if mcq:
            res.append(mcq)
            bucket_counts[_] += 1
    pd.DataFrame(res).to_csv(CSV_FILE, index=False)
    print(f"saved {len(res)} MCQs to {CSV_FILE}")
