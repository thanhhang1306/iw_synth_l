import pandas as pd

# file to merge 
files = [
    '/Users/hangpham/Downloads/IW/della/data/var_7_1.csv',
    '/Users/hangpham/Downloads/IW/della/data/var_7_2.csv',
]

# read
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# counts 
counts = df['sentiment'].value_counts()
missing = [s for s in ['positive','neutral','negative'] if counts.get(s,0)==0]
if missing:
    raise ValueError(f"missing rows for {missing}.")

print("Counts before sampling:", counts.to_dict())

target_size = 500
final_parts = []
for s in ['positive','neutral','negative']:
    subset = df[df['sentiment']==s]
    if len(subset) >= target_size:
        sampled = subset.sample(target_size, random_state=100, replace=False)
    else:
        sampled = subset.sample(target_size, random_state=100, replace=True)
    final_parts.append(sampled)

# save
OUT_DIR = pd.concat(final_parts, ignore_index=True)
CSV_FILE = '/Users/hangpham/Downloads/IW/della/data/var_7_final_time.csv'
df_final.to_csv(CSV_FILE, index=False)

print("final counts:", df_final['sentiment'].value_counts().to_dict())
print(f"saved balanced file to {CSV_FILE}")
