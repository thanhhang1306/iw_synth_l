import os
import csv
import asyncio
from tqdm import tqdm
from googletrans import Translator

# async translation helper

async def translate_async(text, src='en', dest='ace'):
    # google translate async
    tr = Translator()
    res = await tr.translate(text, src=src, dest=dest)
    return res.text.strip()

def translate(text):
    # run async translation
    return asyncio.run(translate_async(text))

if __name__ == '__main__':
    base = "/scratch/gpfs/de7281/sentiment_analysis/variant3_data"
    inp = os.path.join(base, "variant2_raw_for_lex.csv")
    out = os.path.join(base, "variant3_final.csv")
    with open(inp, newline='', encoding='utf-8') as inf, \
         open(out, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=['sentiment','en_text','ace_text'])
        writer.writeheader()
        for row in tqdm(reader, desc="translate with mt"):
            ace = translate(row['en_text'])
            writer.writerow({
                'sentiment': row['sentiment'],
                'en_text':   row['en_text'],
                'ace_text':  ace
            })
