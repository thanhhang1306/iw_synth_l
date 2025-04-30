import os
import json
import random
import asyncio
from googletrans import Translator
from datasets import load_dataset

# async helper 
async def translate_async(text: str, src: str = 'en', dest: str = 'ti') -> str:
    translator = Translator()
    result = await translator.translate(text, src=src, dest=dest)
    return result.text.strip()

# sync wrapper
def translate(text: str, src: str = 'en', dest: str = 'ti') -> str:
    return asyncio.run(translate_async(text, src, dest))


# main
def main():
    OUT_FILE     = "passages_tir.jsonl"
    MAX_PASSAGES = 2000
    SEED         = 100

    os.makedirs(os.path.dirname(OUT_FILE) or '.', exist_ok=True)
    ds = load_dataset("squad", split="train")
    contexts = list({ex['context'] for ex in ds})

    random.seed(SEED)
    random.shuffle(contexts)
    contexts = contexts[:MAX_PASSAGES]

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for i, en_passage in enumerate(contexts, 1):
            ti_passage = translate(en_passage, src='en', dest='ti')
            fout.write(json.dumps({"en": en_passage, "ti": ti_passage}, ensure_ascii=False) + "")
            if i % 100 == 0:
    print(f"Done: translations saved to {OUT_FILE}")

if __name__ == "__main__":
    main()