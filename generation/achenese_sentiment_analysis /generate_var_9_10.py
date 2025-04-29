import os
import random
import torch
import csv
import re
import html
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
from datasets import load_dataset

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# clean text utility
def clean_text(text):
    t = text.strip().replace("quot;", "")
    t = html.unescape(t)
    t = re.sub(r"\s+", ' ', t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t

# check seed words utility
def contains_seed_words(text, seeds, min_count):
    lc = text.lower()
    return sum(1 for w in seeds if w.lower() in lc) >= min_count

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'

    # load models
    gen_tok   = AutoTokenizer.from_pretrained('google/flan-t5-large', cache_dir=cache)
    gen_mod   = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', cache_dir=cache).to(device)
    sent_tok  = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache)
    sent_mod  = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache).to(device)
    classifier= pipeline(
        'sentiment-analysis',
        model=sent_mod,
        tokenizer=sent_tok,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32,
        truncation=True,
        max_length=128,
        top_k=1
    )

    # load lexicon
    lex_data = load_dataset('google/smol','gatitos__en_ace',split='train')
    lexicon = [e['src'] for e in lex_data]

    # prepare output file
    out_dir = '/scratch/gpfs/de7281/sentiment_analysis/variant9_data'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'variant9_raw.csv')
    exists  = os.path.isfile(out_csv)
    fout    = open(out_csv, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(fout)
    if not exists:
        writer.writerow(['sentiment','en_text'])

    # generation settings
    settings     = {
        'positive': 'uplifting and cheerful',
        'neutral':  'calm and balanced',
        'negative': 'intense and somber'
    }
    NUM_EXAMPLES = 500
    BATCH_SIZE    = 16
    MIN_LEX      = 2

    # generate loop with chain-of-thought
    for label, guide in settings.items():
        pbar, count = tqdm(total=NUM_EXAMPLES, desc=label), 0
        while count < NUM_EXAMPLES:
            sample = random.sample(lexicon, k=min(len(lexicon), MIN_LEX*3))
            prompt = (
                f"you are a helpful assistant. let's think step by step about crafting a {label} english sentence that is {guide}"  
                f" and naturally includes at least {MIN_LEX} of these words: {', '.join(sample)}. then provide only the final sentence."
            )
            inputs = gen_tok(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outs = gen_mod.generate(
                    **inputs,
                    max_length=64,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=BATCH_SIZE,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    decoder_start_token_id=gen_tok.pad_token_id
                )
            cands = gen_tok.batch_decode(outs, skip_special_tokens=True)
            cands = [clean_text(s) for s in cands]

            for text in cands:
                res  = classifier([text])[0]
                pred = {'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}[res['label']]
                if pred == label and res['score'] >= 0.8 and contains_seed_words(text, lexicon, MIN_LEX):
                    writer.writerow([label, text])
                    fout.flush()
                    count += 1
                    pbar.update(1)
                    if count >= NUM_EXAMPLES:
                        break
        pbar.close()
    fout.close()
