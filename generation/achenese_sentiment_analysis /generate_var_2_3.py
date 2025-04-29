import random
import os
import gc
import csv
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)

# seed for reproducibility
random.seed(100)
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

# generation with flan-t5 and twitter-roberta

def generate_sentiment():
    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_cache = "/scratch/gpfs/de7281/huggingface"
    tokenizer_gen = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", cache_dir=hf_cache
    )
    model_gen = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large", cache_dir=hf_cache
    ).to(device)
    tokenizer_sent = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment", cache_dir=hf_cache
    )
    model_sent = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment", cache_dir=hf_cache
    ).to(device)
    classifier = pipeline(
        "sentiment-analysis",
        model=model_sent,
        tokenizer=tokenizer_sent,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=16,
        truncation=True,
        top_k=1
    )
    # prepare output file
    base_dir = "/scratch/gpfs/de7281/sentiment_analysis/variant2_data"
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "variant2_raw_for_lex.csv")
    exists = os.path.isfile(csv_path)
    csvfile = open(csv_path, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    if not exists:
        writer.writerow(['sentiment', 'en_text'])
        csvfile.flush()
    # generate and filter
    settings = {
        'positive': 'uplifting and cheerful',
        'neutral':  'neither positive nor negative, but calm and balanced',
        'negative': 'intense and somber'
    }
    num_examples = 500
    for label, guide in settings.items():
        # generate sentences for each label
        pbar = tqdm(total=num_examples, desc=label)
        count = 0
        while count < num_examples:
            prompt = f"generate a {label} english sentence that is {guide}."
            inputs = tokenizer_gen(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outs = model_gen.generate(
                    **inputs,
                    max_length=64,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=16,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    decoder_start_token_id=tokenizer_gen.pad_token_id
                )
            candidates = tokenizer_gen.batch_decode(
                outs, skip_special_tokens=True
            )
            for text in candidates:
                res = classifier([text])[0]
                pred = {'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}[res['label']]
                if pred == label and res['score'] >= 0.8:
                    writer.writerow([label, text])
                    csvfile.flush()
                    count += 1
                    pbar.update(1)
                    if count >= num_examples:
                        break
        pbar.close()
    csvfile.close()
    
if __name__ == '__main__':
    generate_sentiment()