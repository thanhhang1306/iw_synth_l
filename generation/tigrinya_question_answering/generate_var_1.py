import asyncio
import random
import pandas as pd
from datasets import load_dataset
from googletrans import Translator
from tqdm import tqdm

# seed for reproducibility
random.seed(100)

# async translation helper
def translate_async(text, src='en', dest='ti'):
    translator = Translator()
    res = await translator.translate(text, src=src, dest=dest)
    return res.text.strip()

def translate_run(text, src, dest):
    return asyncio.run(translate_async(text, src, dest))

def translate(text, src='en', dest='ti'):
    return translate_run(text, src, dest)

# back translation for a single string
def back_translate(text):
    try:
        en = translate(text, src='ti', dest='en')
        ti = translate(en, src='en', dest='ti')
    except Exception:
        en, ti = text, text
    return en, ti

# process one QA row
text_columns = ['passage','question','mc_answer1','mc_answer2','mc_answer3','mc_answer4']

def process_row(row):
    data = {}
    for col in text_columns:
        en, ti = back_translate(row[col])
        data[f'en_{col}'] = en
        data[f'translated_{col}'] = ti
    data['question_number'] = row['question_number']
    data['correct_answer_num'] = row['correct_answer_num']
    return pd.Series(data)

def main():
    ds = load_dataset("facebook/belebele", "tir_Ethi", split="test")
    df = ds.to_pandas()
    df = df[['passage','question_number','question','mc_answer1','mc_answer2','mc_answer3','mc_answer4','correct_answer_num']]

    tqdm.pandas(desc="back translating QA")
    translated = df.progress_apply(process_row, axis=1)

    final = pd.concat([
        df[['question_number','correct_answer_num']],
        translated
    ], axis=1)

    CSV_FILE = "variant_1_qa.csv"
    final.to_csv(CSV_FILE, index=False)
    print(f"saved 900 mcqs to {CSV_FILE}")

if __name__ == '__main__':
    main()