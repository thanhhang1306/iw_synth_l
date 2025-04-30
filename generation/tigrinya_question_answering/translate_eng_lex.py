import os, csv, re
from tqdm import tqdm
from datasets import load_dataset

# token-by-token lexicon lookup
def translate_to_ti(sentence: str, lexicon: dict) -> str:
    pattern = r"(\b\w+\b|[^\w\s])"
    tokens = re.findall(pattern, sentence)
    out = []
    for t in tokens:
        key = t.lower()
        mapped = lexicon.get(key, [t])[0]
        out.append(mapped)
    # fix spacing before punctuation
    return re.sub(r"\s+([,.!?;:])", r"\1", " ".join(out))

if __name__ == '__main__':
    # file paths
    OUT_DIR   = '/scratch/gpfs/de7281/new_data/mcq_data'
    IN_FILE  = os.path.join(OUT_DIR, 'variant_10_mcq.csv')
    OUT_FILE = os.path.join(OUT_DIR, 'variant_10_mcq_lex.csv')

    # load bilingual lexicon
    ds = load_dataset('google/smol', 'gatitos__en_ti', split='train')
    eng_to_ti = {e['src'].lower(): e['trgs'] for e in ds}

    # prepare output
    with open(IN_FILE, newline='', encoding='utf-8') as inf, \
         open(OUT_FILE, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames + [
            'translated_passage', 'translated_question',
            'translated_mc_answer1', 'translated_mc_answer2',
            'translated_mc_answer3', 'translated_mc_answer4'
        ]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc='lexicon translate'):
            row['translated_passage'] = translate_to_ti(row['passage'], eng_to_ti)
            row['translated_question'] = translate_to_ti(row['question'], eng_to_ti)
            for i in range(1, 5):
                row[f'translated_mc_answer{i}'] = translate_to_ti(row[f'mc_answer{i}'], eng_to_ti)
            writer.writerow(row)
