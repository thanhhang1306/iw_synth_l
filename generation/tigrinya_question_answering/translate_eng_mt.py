import os
import csv
import asyncio
from tqdm import tqdm
from googletrans import Translator

# async translation helper
async def translate_async(text: str, src: str = 'en', dest: str = 'ti') -> str:
    tr = Translator()
    res = await tr.translate(text, src=src, dest=dest)
    return res.text.strip()

def translate(text: str, src: str = 'en', dest: str = 'ti') -> str:
    return asyncio.run(translate_async(text, src=src, dest=dest))

if __name__ == '__main__':
    # directories and filepaths
    BASE_DIR     = "/scratch/gpfs/de7281"
    INPUT_FILE   = os.path.join(BASE_DIR, "new_data", "mcq_data", "variant_12_mcq.csv")
    OUTPUT_FILE  = os.path.join(BASE_DIR, "new_data", "mcq_data", "variant_12_mcq_google.csv")

    # ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # define CSV headers: same as input plus translated_* columns
    fieldnames = [
        "passage","question_number","question",
        "mc_answer1","mc_answer2","mc_answer3","mc_answer4",
        "correct_answer_num",
        "translated_passage","translated_question",
        "translated_mc_answer1","translated_mc_answer2",
        "translated_mc_answer3","translated_mc_answer4"
    ]

    # open input and output CSVs
    with open(INPUT_FILE, newline='', encoding='utf-8') as inf, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outf:

        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="translating with MT"):
            # translate each field
            tp = translate(row["passage"])
            tq = translate(row["question"])
            a1 = translate(row["mc_answer1"])
            a2 = translate(row["mc_answer2"])
            a3 = translate(row["mc_answer3"])
            a4 = translate(row["mc_answer4"])

            # write augmented row
            writer.writerow({
                "passage":                 row["passage"],
                "question_number":         row["question_number"],
                "question":                row["question"],
                "mc_answer1":              row["mc_answer1"],
                "mc_answer2":              row["mc_answer2"],
                "mc_answer3":              row["mc_answer3"],
                "mc_answer4":              row["mc_answer4"],
                "correct_answer_num":      row["correct_answer_num"],
                "translated_passage":       tp,
                "translated_question":      tq,
                "translated_mc_answer1":    a1,
                "translated_mc_answer2":    a2,
                "translated_mc_answer3":    a3,
                "translated_mc_answer4":    a4,
            })

