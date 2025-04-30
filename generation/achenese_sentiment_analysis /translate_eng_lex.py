import os
import csv
import re
from tqdm import tqdm
from datasets import load_dataset

# token-by-token lexicon lookup

def translate_to_ace(sentence, lexicon):
    # split tokens and replace
    pattern = r"(\b\w+\b|[^\w\s])"
    tokens = re.findall(pattern, sentence)
    out = []
    for t in tokens:
        key = t.lower()
        out.append(lexicon[key][0] if key in lexicon else t)
    text = " ".join(out)
    return re.sub(r'\s+([,.!?;:])', r'\1', text)

if __name__ == '__main__':
    OUT_DIR = "/scratch/gpfs/de7281/sentiment_analysis/variant2_data"
    INP_FILE = os.path.join(OUT_DIR, "variant_9_raw_for_lex.csv")
    OUT_FILE = os.path.join(OUT_DIR, "variant_9_final.csv")
    # load lexicon
    ds = load_dataset("google/smol", "gatitos__en_ace", split="train")
    lex = {e['src'].lower(): e['trgs'] for e in ds}
    # translate rows
    with open(INP_FILE, newline='', encoding='utf-8') as inf, \
         open(OUT_FILE, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=['sentiment','en_text','ace_text'])
        writer.writeheader()
        for row in tqdm(reader, desc="translate with lexicon"):
            ace = translate_to_ace(row['en_text'], lex)
            writer.writerow({
                'sentiment': row['sentiment'],
                'en_text':   row['en_text'],
                'ace_text':  ace
            })