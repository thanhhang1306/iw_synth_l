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
    OUT_DIR = "/scratch/gpfs/de7281/sentiment_analysis/variant3_data"
    INP_FILE = os.path.join(OUT_DIR, "variant_2_raw_for_mt.csv")
    OUT_FILE = os.path.join(OUT_DIR, "variant_2_final.csv")
    with open(INP_FILE, newline='', encoding='utf-8') as inf, \
         open(OUT_FILE, 'w', newline='', encoding='utf-8') as outf:
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
