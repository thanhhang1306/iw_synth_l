import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# pseudo-perplexity filtering for acehnese

if __name__ == '__main__':
    base = '/scripts/fine_tune'
    inp = os.path.join(base, 'variant_7_final.csv')
    out = os.path.join(base, 'variant_7_final_check.csv')
    THRESH = 100.0
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = '/scratch/gpfs/de7281/huggingface'
    tok = AutoTokenizer.from_pretrained('xlm-roberta-large', cache_dir=cache)
    model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large', cache_dir=cache).to(device)
    model.eval()

    def pseudo_perplexity(txt):
        tokens = tok(txt, return_tensors='pt').input_ids.to(device)
        n = tokens.size(1)
        loss = 0.0
        with torch.no_grad():
            for i in range(1,n-1):
                mask = tokens.clone()
                mask[0,i] = tok.mask_token_id
                logits = model(mask).logits[0,i]
                prob = torch.softmax(logits, dim=-1)[tokens[0,i]]
                loss += -torch.log(prob + 1e-12)
        return float(torch.exp(loss/(n-2)))

    # process rows
    with open(inp,newline='',encoding='utf-8') as fin, open(out,'w',newline='',encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fnames = reader.fieldnames + ['acehnese_pppl']
        writer = csv.DictWriter(fout, fieldnames=fnames)
        writer.writeheader()
        for idx,row in enumerate(reader,1):
            txt = row.get('ace_text','').strip()
            if not txt:
                continue
            ppl = pseudo_perplexity(txt)
            if ppl <= THRESH:
                row['acehnese_pppl'] = f"{ppl:.2f}"
                writer.writerow(row)
    print(f"filtered output to {out}")
