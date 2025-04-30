# File: generate_variant5.py
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
import re
import html

# seed for reproducibility
def set_seed(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# clean text utility
def clean_text(text):
    text = text.strip().replace("quot;", "")
    text = html.unescape(text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text

# check seed words utility
def contains_seed_words(text, seeds, min_count):
    lc = text.lower()
    return sum(1 for w in seeds if w.lower() in lc) >= min_count

# batch generation with lexicon seeding
def generate_batch(label, guide, batch_size, lexicon, min_lex):
    sample = random.sample(lexicon, k=min(len(lexicon), min_lex * 3))
    prompt = (
        f"generate a {label} english sentence that is {guide} "
        f"and includes at least {min_lex} of these words: {', '.join(sample)}."
    )
    inputs = gen_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
    outs = gen_model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=batch_size,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        decoder_start_token_id=gen_tokenizer.pad_token_id
    )
    texts = gen_tokenizer.batch_decode(outs, skip_special_tokens=True)
    return [clean_text(t) for t in texts]

if __name__ == '__main__':
    set_seed()
    # load models and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'
    gen_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large', cache_dir=cache)
    gen_model     = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large', cache_dir=cache).to(device)
    sent_tokenizer= AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache)
    sent_model    = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache).to(device)
    classifier    = pipeline(
        'sentiment-analysis',
        model=sent_model,
        tokenizer=sent_tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32,
        truncation=True,
        max_length=128,
        top_k=1
    )
    lex_data      = load_dataset('google/smol', 'gatitos__en_ace', split='train')
    lexicon       = [e['src'] for e in lex_data]

    # prepare output file
    OUT_DIR = '/scratch/gpfs/de7281/sentiment_analysis/variant5_data'
    os.makedirs(OUT_DIR, exist_ok=True)
    CSV_FILE = os.path.join(OUT_DIR, 'variant5_raw.csv')
    exists  = os.path.isfile(CSV_FILE)
    fout    = open(CSV_FILE, 'a', newline='', encoding='utf-8')
    writer  = csv.writer(fout)
    if not exists:
        writer.writerow(['sentiment', 'en_text'])

    # generation settings
    settings     = {
        'positive': 'uplifting and cheerful',
        'neutral':  'calm and balanced',
        'negative': 'intense and somber'
    }
    num_examples = 500
    batch_size   = 16
    min_lex      = 2

    # generate loop
    for label, guide in settings.items():
        pbar, count = tqdm(total=num_examples, desc=label), 0
        while count < num_examples:
            batch_texts = generate_batch(label, guide, batch_size, lexicon, min_lex)
            for text in batch_texts:
                res  = classifier([text])[0]
                pred = {'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}[res['label']]
                if pred == label and res['score'] >= 0.8 and contains_seed_words(text, lexicon, min_lex):
                    writer.writerow([label, text])
                    fout.flush()
                    count += 1
                    pbar.update(1)
                    if count >= num_examples:
                        break
        pbar.close()
    fout.close()