import random
import torch
import asyncio
import csv
import os
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from googletrans import Translator

# seed for reproducibility
random.seed(100)
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

# async translation helper
async def translate_async(text, src='ace', dest='en'):
    # google translate async
    tr = Translator()
    res = await tr.translate(text, src=src, dest=dest)
    return res.text.strip()

def translate(text):
    # run async translation
    return asyncio.run(translate_async(text))

# translate and verify sentiment

def translate_and_verify(input_path, output_path):
    # load classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'
    tokenizer = AutoTokenizer.from_pretrained(
        'cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        'cardiffnlp/twitter-roberta-base-sentiment', cache_dir=cache
    ).to(device)
    classifier = pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32,
        truncation=True,
        max_length=128,
        top_k=1
    )
    label_map = {'LABEL_0':'negative','LABEL_1':'neutral','LABEL_2':'positive'}

    processed = set()
    header = True
    if os.path.isfile(output_path):
        with open(output_path, newline='', encoding='utf-8') as prev:
            for row in csv.DictReader(prev): processed.add(row['ace_text'])
        header = False

    with open(input_path, newline='', encoding='utf-8') as inf, \
         open(output_path, 'a' if not header else 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=['sentiment','ace_text','en_text'])
        if header: writer.writeheader()
        for row in tqdm(reader, desc='translate & verify'):
            ace = row['ace_text']
            if ace in processed: continue
            en  = translate(ace)
            res = classifier([en])[0]
            pred= label_map[res['label']]
            if pred==row['sentiment'] and res['score']>=0.80:
                writer.writerow({'sentiment':row['sentiment'],'ace_text':ace,'en_text':en})
                processed.add(ace)
    print(f'final data at {output_path}')

if __name__=='__main__':
    base = '/scratch/gpfs/de7281/variant4_data'
    inp = os.path.join(base, 'variant4_raw.csv')
    out = os.path.join(base, 'variant4_final.csv')
    translate_and_verify(inp, out)
