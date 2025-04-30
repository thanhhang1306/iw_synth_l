import os
import random
import time
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoTokenizer as HF_AutoTokenizer,
    AutoModelForSequenceClassification
)
import spacy
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# buckets for diversity-aware sampling
def load_title_buckets(max_per_title):
    ds = load_dataset('squad', split='train', cache_dir=HF_CACHE)
    buckets = {}
    for ex in ds:
        title = ex['title'].replace(' ', '_')
        lst = buckets.setdefault(title, [])
        if len(lst) < max_per_title:
            lst.append(ex['context'])
    return buckets

def sample_passage():
    remaining = [t for t,c in bucket_counts.items() if c < TARGET//len(bucket_counts)+1]
    title = random.choice(remaining)
    return title, random.choice(TITLE_BUCKETS[title])

# pick answer span via NER/noun-chunks
nlp = spacy.load('en_core_web_sm', disable=['textcat'])
def pick_answer_span(text):
    doc = nlp(text[:400])
    ents = [e.text for e in doc.ents if len(e.text.split()) <= 5]
    if ents:
        return random.choice(ents)
    nps = [nc.text for nc in doc.noun_chunks if len(nc.text.split()) <= 5]
    return random.choice(nps) if nps else text.split()[0]

# build distractors from entities + WordNet
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

# QA pipelines for verification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qa1_pipe = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad', device=0 if torch.cuda.is_available() else -1)
qa2_pipe = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2', device=0 if torch.cuda.is_available() else -1)
def qa_preds(passage, question):
    return qa1_pipe(question=question, context=passage)['answer'].lower(), qa2_pipe(question=question, context=passage)['answer'].lower()
def qa_agree(ans, p1, p2):
    return (ans.lower() in p1 or p1 in ans.lower()) and (ans.lower() in p2 or p2 in ans.lower())

# NLI entailment check
def load_nli():
    tok = HF_AutoTokenizer.from_pretrained('facebook/bart-large-mnli', cache_dir=HF_CACHE)
    mdl = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', cache_dir=HF_CACHE).to(device)
    return tok, mdl
nli_tokenizer, nli_model = load_nli()
def nli_entails(passage, question, ans, th=0.5):
    hypo = f"the answer to the question '{question}' is '{ans}'."
    enc = nli_tokenizer(passage[:512], hypo, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = nli_model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    label = ['CONTRADICTION','NEUTRAL','ENTAILMENT'][probs.argmax().item()]
    score = probs.max().item()
    return label=='ENTAILMENT' and score>=th

# chain-of-thought MCQ generation
qg_tokenizer = T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=HF_CACHE)
qg_model     = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=HF_CACHE).to(device)
def generate_question(passage):
    ans = pick_answer_span(passage)
    prompt = (
        "you are a helpful assistant. let's think step by step: "
        f"1) identify the answer span '{ans}' in the passage. "
        "2) formulate a clear multiple-choice question about that span. "
        "3) provide four options including the correct answer. "
        "finally, output only the question and options. "
        f"passage: {passage}"
    )
    ids = qg_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    out= qg_model.generate(**ids, max_length=80, do_sample=True, top_p=0.9, no_repeat_ngram_size=2)
    raw = qg_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    # parse into fields
    data={}
    for line in raw.splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            data[k.lower().replace(' ','_')] = v.strip()
    required = ['question','mc_answer1','mc_answer2','mc_answer3','mc_answer4','correct_answer_num']
    if not all(r in data for r in required):
        return None
    # verification
    q = data['question']
    if not qa_agree(ans,*qa_preds(passage,q)):
        return None
    if not nli_entails(passage,q,ans):
        return None
    data['passage']=passage
    return data

if __name__=='__main__':
    set_seed()
    HF_CACHE = '/scratch/gpfs/de7281/huggingface'
    TARGET   = 500
    MAX_BUCKET = 10
    OUT_DIR = '/scratch/gpfs/de7281/new_data/mcq_data'
    os.makedirs(OUT_DIR,exist_ok=True)
    CSV_FILE = os.path.join(OUT_DIR,'variant_9_mcq.csv')

    TITLE_BUCKETS = load_title_buckets(MAX_BUCKET)
    bucket_counts = {t:0 for t in TITLE_BUCKETS}

    res=[]; loops=0
    while len(res)<TARGET and loops < len(TITLE_BUCKETS)*TARGET:
        loops+=1
        title, passage = sample_passage()
        mcq = generate_question(passage)
        if mcq:
            bucket_counts[title]+=1
            mcq['question_number']=len(res)+1
            mcq['source_title']=title
            res.append(mcq)
    pd.DataFrame(res).to_csv(CSV_FILE,index=False)
    print(f'saved {len(res)} MCQs to {CSV_FILE}')
