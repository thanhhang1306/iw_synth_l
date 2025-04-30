import os
import random
import time
import sys
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
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

# pick a short answer span from passage
def pick_answer_span(text):
    doc = nlp(text[:400])
    ents = [e.text for e in doc.ents if len(e.text.split()) <= 5]
    if ents:
        return random.choice(ents)
    nps = [nc.text for nc in doc.noun_chunks if len(nc.text.split()) <= 5]
    return random.choice(nps) if nps else text.split()[0]

# build 3 distractors using named entities and wordnet
def build_distractors(ans, passage):
    ents = [e.text for e in nlp(passage).ents if e.text.lower() != ans.lower() and len(e.text.split()) <= 5]
    wn_alts = [l.name().replace("_", " ") for syn in wn.synsets(ans) for l in syn.hyponyms()[:3]]
    corpus = list(dict.fromkeys(ents + wn_alts))[:10]
    if not corpus:
        return ["—", "—", "—"]
    sims = util.cos_sim(embedder.encode([ans]), embedder.encode(corpus))[0]
    idxs = sims.argsort(descending=False)[:3]
    return [corpus[i] for i in idxs]

# get qa model predictions
def qa_preds(passage, question):
    a1 = qa1_pipe(question=question, context=passage)["answer"].lower()
    a2 = qa2_pipe(question=question, context=passage)["answer"].lower()
    return a1, a2

# require both qa models to agree on answer span
def qa_agree(ans, p1, p2):
    return ans.lower() in p1 or p1 in ans.lower() and ans.lower() in p2 or p2 in ans.lower()

# check nli entailment between passage and answer hypothesis
def nli_entails(passage, question, ans, threshold=0.5):
    hypo = f"the answer to the question '{question}' is '{ans}'."
    enc = nli_tokenizer(passage[:512], hypo, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = nli_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # labels: contradiction, neutral, entailment
    label = ['CONTRADICTION','NEUTRAL','ENTAILMENT'][probs.argmax().item()]
    score = probs.max().item()
    return label == 'ENTAILMENT' and score >= threshold, label, score

# generate one mcq with dual verification
def generate_mcq(passage):
    ans = pick_answer_span(passage)
    prompt = f"generate question: <hl>{ans}<hl>{passage}</s>"
    ids = qg_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    out = qg_model.generate(**ids, max_length=64, do_sample=True, top_p=0.9)
    question = qg_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    options = [ans] + build_distractors(ans, passage)
    random.shuffle(options)
    print(f"debug: question='{question}' answer='{ans}' options={options}")
    p1, p2 = qa_preds(passage, question)
    print(f"debug: qa1={p1} qa2={p2}")
    if not qa_agree(ans, p1, p2):
        print("reject: qa disagreement")
        return None
    ok, label, score = nli_entails(passage, question, ans)
    print(f"debug: nli={label} score={score:.2f}")
    if not ok:
        print("reject: nli entailment")
        return None
    return {
        'passage': passage,
        'question': question,
        'mc_answer1': options[0],
        'mc_answer2': options[1],
        'mc_answer3': options[2],
        'mc_answer4': options[3],
        'correct_answer_num': options.index(ans) + 1
    }

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'
    # load models and tools
    qg_model = AutoModelForSeq2SeqLM.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=cache).to(device)
    qg_tokenizer = T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl', cache_dir=cache)
    qa1_pipe = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased-distilled-squad', device=0 if torch.cuda.is_available() else -1)
    qa2_pipe = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2', device=0 if torch.cuda.is_available() else -1)
    nli_model = pipeline('text-classification', model='facebook/bart-large-mnli', tokenizer='facebook/bart-large-mnli', device=0 if torch.cuda.is_available() else -1)
    # sentence transformer and spacy
    embedder = SentenceTransformer('all-mpnet-base-v2', cache_folder=cache)
    nlp = spacy.load('en_core_web_sm', disable=['textcat'])

    # load passages
    passages = list({ex['context'] for ex in load_dataset('squad', split='train', cache_dir=cache)})

    # prepare output
    OT = '/scratch/gpfs/de7281/new_data/var_2'
    os.makedirs(OUT_DIR, exist_ok=True)
    CSV_FILE = os.path.join(OUT_DIR, 'variant_2_mcq.csv')
    df_res = []
    target = 500

    # generation loop
    while len(df_res) < target:
        mcq = generate_mcq(random.choice(passages))
        if mcq:
            mcq['question_number'] = len(df_res) + 1
            df_res.append(mcq)
    # save to csv
    pd.DataFrame(df_res)[['passage','question_number','question','mc_answer1','mc_answer2','mc_answer3','mc_answer4','correct_answer_num']].to_csv(CSV_FILE, index=False)
    print(f"saved {len(df_res)} mcqs to {CSV_FILE}")
