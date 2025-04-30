import asyncio
import random
import pandas as pd
from datasets import load_dataset
from googletrans import Translator
from tqdm import tqdm

# seed for reproducibility
random.seed(100)

train = load_dataset("indonlp/NusaX-senti", "ace", split="train")
df = train.to_pandas()[["text", "label"]]

async def translate_async(text, src='en', dest='ace'):
    # async translation via google translate
    translator = Translator()
    result = await translator.translate(text, src=src, dest=dest)
    return result.text.strip()


def translate_run(text, src, dest):
    # run async translation synchronously
    return asyncio.run(translate_async(text, src, dest))


def translate(text, src='en', dest='ace'):
    # translate text via google translate
    return translate_run(text, src, dest)


def back_translate(text):
    # back translation via ace->en->ace
    try:
        en = translate(text, 'ace', 'en')
        ace = translate(en, 'en', 'ace')
    except Exception:
        en = text
        ace = text
    return en, ace


tqdm.pandas(desc="back translating")
df[['en_text', 'ace_text']] = df['text'].progress_apply(
    lambda x: pd.Series(back_translate(x))
)

# rename and map labels
mapper = {0: 'negative', 1: 'neutral', 2: 'positive'}
df.rename(columns={'label': 'sentiment'}, inplace=True)
df['sentiment'] = df['sentiment'].map(mapper)

final = df[['sentiment', 'en_text', 'ace_text']]

# save results
def main():
    # write to csv
    final.to_csv("variant_1.csv", index=False)
    print("saved 500 examples to variant_1.csv")


if __name__ == '__main__':
    main()
