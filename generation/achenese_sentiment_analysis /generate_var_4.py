import random
import torch
import csv
import os
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# seed for reproducibility
random.seed(100)
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

# generate acehnese sentences

def generate_acehnese(sentiment, guide, num=1):
    # load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'
    model_id = 'LazarusNLP/indo-t5-base-v2-nusax'
    tokenizer = MT5Tokenizer.from_pretrained(model_id, cache_dir=cache, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache).to(device)
    # prompt and sample
    prompt = f"generate a {sentiment} acehnese sentence. {guide}"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num
    )
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outs]

# main script
if __name__ == '__main__':
    settings = {
        'positive': 'the mood is uplifting and cheerful',
        'neutral':  'the mood is calm and balanced',
        'negative': 'the mood is intense and somber'
    }
    count_per_label = 500
    base = '/scratch/gpfs/de7281/variant4_data'
    os.makedirs(base, exist_ok=True)
    outpath = os.path.join(base, 'variant4_raw.csv')
    exists = os.path.isfile(outpath)
    outfile = open(outpath, 'a', newline='', encoding='utf-8')
    writer = csv.writer(outfile)
    if not exists:
        writer.writerow(['sentiment','ace_text'])
        outfile.flush()
    for label, guide in settings.items():
        pbar = tqdm(total=count_per_label, desc=label)
        count = 0
        while count < count_per_label:
            texts = generate_acehnese(label, guide, 1)
            if not texts:
                continue
            writer.writerow([label, texts[0]])
            outfile.flush()
            count += 1
            pbar.update(1)
        pbar.close()
    outfile.close()
