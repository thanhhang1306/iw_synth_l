import os
import csv
import json
import random
import numpy as np
import pandas as pd
import torch
import nltk
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)
from scipy.special import softmax
from wordcloud import WordCloud
from nltk import word_tokenize, ngrams
from collections import Counter
from matplotlib import font_manager

nltk.download('punkt')  # download required nltk resources

# configuration
BASE_DIR = "/scratch/gpfs/de7281"
HF_CACHE = os.path.join(BASE_DIR, "huggingface")
os.makedirs(HF_CACHE, exist_ok=True)

SYNTHETIC_METHOD = "variant_12_final"
SYNTHETIC_FILE   = "scripts/fine_tune_2/variant_12_final.csv"

PLOT_DIR         = os.path.join(BASE_DIR, "final_qa/hang_plots", SYNTHETIC_METHOD)
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "final_qa/hang_results", SYNTHETIC_METHOD)
MODEL_SAVE_DIR   = os.path.join(BASE_DIR, "final_qa/hang_models", SYNTHETIC_METHOD)
for d in (PLOT_DIR, TRAIN_OUTPUT_DIR, MODEL_SAVE_DIR):
    os.makedirs(d, exist_ok=True)

ETHIOPIC_FONT = os.path.join("scripts", "fonts/NotoSansEthiopic-Regular.ttf")
font_manager.fontManager.addfont(ETHIOPIC_FONT)
mpl.rcParams['font.family'] = 'Noto Sans Ethiopic'

# data analysis helper functions

def compute_text_length(text):
    if not isinstance(text, str):
        return 0
    return len(word_tokenize(text))

def compute_type_token_ratio(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = word_tokenize(text)
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def plot_histogram(data, column, title, xlabel, ylabel, bins=30, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_word_cloud(text, title, save_path=None):
    wc = WordCloud(width=800, height=400, background_color='white',
                   font_path=ETHIOPIC_FONT).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_ngrams_analysis(text, top_k=10, n=2, save_path=None):
    tokens = word_tokenize(text.lower())
    ngram_freq = Counter(ngrams(tokens, n))
    common = ngram_freq.most_common(top_k)
    if common:
        labels, counts = zip(*common)
        labels = [' '.join(gram) for gram in labels]
    else:
        labels, counts = [], []
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(labels))
    plt.title(f"Top {top_k} {'bigrams' if n==2 else 'trigrams'}")
    plt.xlabel('frequency')
    plt.ylabel('n-grams')
    if save_path:
        plt.savefig(save_path)
    plt.close()

# load, combine, and eda
def load_and_combine_data(real_split="test", real_size=250, syn_size=None):
    real_ds = load_dataset("facebook/belebele", "tir_Ethi", split=real_split, cache_dir=HF_CACHE)
    cols = ["flores_passage","question","mc_answer1","mc_answer2","mc_answer3","mc_answer4","correct_answer_num"]
    real_df = real_ds.to_pandas()[cols]
    real_df = real_df.loc[:, ~real_df.columns.duplicated()].reset_index(drop=True)
    real_df = real_df.sample(n=real_size, random_state=100)

    syn_df = pd.read_csv(SYNTHETIC_FILE, dtype=str)
    syn_df = syn_df.rename(columns={
        "translated_passage":"flores_passage",
        "translated_question":"question",
        "translated_mc_answer1":"mc_answer1",
        "translated_mc_answer2":"mc_answer2",
        "translated_mc_answer3":"mc_answer3",
        "translated_mc_answer4":"mc_answer4",
    })
    syn_df = syn_df.loc[:, ~syn_df.columns.duplicated()]
    syn_df = syn_df[cols]
    if syn_size:
        syn_df = syn_df.sample(n=syn_size, random_state=100)

    df = pd.concat([real_df, syn_df], ignore_index=True)
    full_real = load_dataset("facebook/belebele","tir_Ethi",split="train",cache_dir=HF_CACHE).to_pandas()[cols]
    leftover = full_real.drop(real_df.index, errors='ignore').reset_index(drop=True)
    val_df  = leftover.sample(n=int(0.2*len(leftover)), random_state=100)
    test_df = leftover.drop(val_df.index).sample(n=int(0.3*len(leftover)), random_state=100)
    val_df['correct_answer_num'] = val_df['correct_answer_num'].astype(int)
    test_df['correct_answer_num'] = test_df['correct_answer_num'].astype(int)
    val_df.to_csv(os.path.join(TRAIN_OUTPUT_DIR,"val_real.csv"), index=False)
    test_df.to_csv(os.path.join(TRAIN_OUTPUT_DIR,"test_real.csv"), index=False)

    df = df.sample(frac=1, random_state=100).reset_index(drop=True)
    df['correct_answer_num'] = pd.to_numeric(df['correct_answer_num'], errors='coerce').dropna().astype(int)
    return df

# eda qa dataset
def eda_qa_dataset(df, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    def dst(fname): return os.path.join(plot_dir, fname)

    df['passage_length'] = df['flores_passage'].apply(compute_text_length)
    df['question_length'] = df['question'].apply(compute_text_length)
    df['question_ttr']    = df['question'].apply(compute_type_token_ratio)
    for i in range(1, 5):
        df[f'mc_answer{i}_length'] = df[f'mc_answer{i}'].apply(compute_text_length)

    print("=== passage length stats ===", df['passage_length'].describe(), sep='\n')
    print("=== question length stats ===", df['question_length'].describe(), sep='\n')
    print("=== question ttr stats ===", df['question_ttr'].describe(), sep='\n')

    plot_histogram(df, 'passage_length', 'passage length distribution', 'words', 'freq', save_path=dst('passage_len.png'))
    plot_histogram(df, 'question_length', 'question length distribution', 'words', 'freq', save_path=dst('question_len.png'))
    plot_histogram(df, 'question_ttr', 'question ttr distribution', 'ttr', 'freq', save_path=dst('question_ttr.png'))
    for i in range(1, 5):
        plot_histogram(df, f'mc_answer{i}_length', f'answer {i} length distribution', 'words', 'freq', save_path=dst(f'ans{i}_len.png'))

    plt.figure(figsize=(8, 6))
    sns.countplot(x='correct_answer_num', data=df)
    plt.title('correct answer label distribution')
    plt.savefig(dst('correct_label_dist.png'))
    plt.close()

    generate_word_cloud(" ".join(df['flores_passage'].dropna()), 'passage wordcloud', save_path=dst('wc_passages.png'))
    generate_word_cloud(" ".join(df['question'].dropna()), 'question wordcloud', save_path=dst('wc_questions.png'))
    for i in range(1, 5):
        generate_word_cloud(" ".join(df[f'mc_answer{i}'].dropna()), f'answer {i} wordcloud', save_path=dst(f'wc_ans{i}.png'))

    all_q = " ".join(df['question'].dropna()).lower()
    plot_ngrams_analysis(all_q, top_k=10, n=2, save_path=dst('bigrams.png'))
    plot_ngrams_analysis(all_q, top_k=10, n=3, save_path=dst('trigrams.png'))

# preprocessing function
 def preprocess_function(examples, tokenizer):
    first = [[ctx]*4 for ctx in examples['flores_passage']]
    seconds = [[f"{q} {a}" for a in (a1,a2,a3,a4)] for q,a1,a2,a3,a4 in zip(examples['question'], examples['mc_answer1'],examples['mc_answer2'], examples['mc_answer3'],examples['mc_answer4'])]
    flat1 = sum(first, [])
    flat2 = sum(seconds, [])
    tok = tokenizer(flat1, flat2, truncation=True, padding='max_length', max_length=256)
    grouped = {k:[v[i:i+4] for i in range(0,len(v),4)] for k,v in tok.items()}
    grouped['labels'] = [(lbl-1) for lbl in examples['correct_answer_num']]
    return grouped

# ranking metric helpers
def mean_reciprocal_rank(logits, labels):
    ranks = []
    for logit, true in zip(logits, labels):
        order = np.argsort(-logit)
        rank = np.where(order == true)[0][0] + 1
        ranks.append(1.0/rank)
    return float(np.mean(ranks))

def mean_average_precision(logits, labels):
    ap = []
    for logit, true in zip(logits, labels):
        order = np.argsort(-logit)
        pos = np.where(order == true)[0][0] + 1
        ap.append(1.0/pos)
    return float(np.mean(ap))

# compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    wp, wr, wf, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    top2 = np.argsort(logits, axis=1)[:, -2:]
    top3 = np.argsort(logits, axis=1)[:, -3:]
    top2_acc = float(np.mean([labels[i] in top2[i] for i in range(len(labels))]))
    top3_acc = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))
    mrr = mean_reciprocal_rank(logits, labels)
    map_score = mean_average_precision(logits, labels)
    try:
        probs = softmax(logits, axis=-1)
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr')
    except:
        roc_auc = None
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "top2_accuracy": top2_acc,
        "top3_accuracy": top3_acc,
        "mrr": mrr,
        "map": map_score,
        "roc_auc": roc_auc,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist()
    }

# logging callback
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_logs = []
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history:
            self.epoch_logs.append(state.log_history[-1])
        return control
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        out_path = os.path.join(args.output_dir, "epoch_logs.json")
        with open(out_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=2)
        return control

# single-experiment runner
def run_experiment(name, real_n, syn_n):
    exp_out   = os.path.join(TRAIN_OUTPUT_DIR, name)
    exp_plot  = os.path.join(PLOT_DIR, name)
    exp_model = os.path.join(MODEL_SAVE_DIR, name)
    for d in (exp_out, exp_plot, exp_model): os.makedirs(d, exist_ok=True)
    df = load_and_combine_data(real_size=real_n, syn_size=syn_n)
    eda_qa_dataset(df, plot_dir=exp_plot)
    df_train = df.copy()
    df_val   = pd.read_csv(os.path.join(TRAIN_OUTPUT_DIR,"val_real.csv"))
    df_test  = pd.read_csv(os.path.join(TRAIN_OUTPUT_DIR,"test_real.csv"))
    ds_train = Dataset.from_pandas(df_train)
    ds_val   = Dataset.from_pandas(df_val)
    ds_test  = Dataset.from_pandas(df_test)
    tokenizer = AutoTokenizer.from_pretrained('castorini/afriberta_large', use_fast=False, cache_dir=HF_CACHE)
    model     = AutoModelForMultipleChoice.from_pretrained('castorini/afriberta_large', cache_dir=HF_CACHE)
    train_ds = ds_train.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=ds_train.column_names)
    val_ds   = ds_val.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=ds_val.column_names)
    test_ds  = ds_test.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=ds_test.column_names)
    for d in (train_ds, val_ds, test_ds): d.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    args = TrainingArguments(output_dir=exp_out, eval_strategy='epoch', save_strategy='epoch', learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=8, num_train_epochs=10, load_best_model_at_end=True, metric_for_best_model='weighted_f1')
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback()])
    trainer.train()
    eval_res = trainer.evaluate(test_ds)
    preds_output = trainer.predict(test_ds)
    logits, labels = preds_output.predictions, preds_output.label_ids
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    with open(os.path.join(exp_out, "classification_report.json"), "w") as f: json.dump(report, f, indent=2)
    print(f"saved per-class classification report to {os.path.join(TRAIN_OUTPUT_DIR, name, 'classification_report.json')}")
    trainer.save_model(exp_model)
    tokenizer.save_pretrained(exp_model)
    with open(os.path.join(exp_model, 'eval.json'), 'w') as f: json.dump(eval_res, f, indent=2)

# batch-run all experiments
def run_all():
    for name,(r,s) in EXPERIMENTS.items():
        print(f"\n=== experiment {name} ===")
        run_experiment(name, real_n=r, syn_n=s)

EXPERIMENTS = {
    'real_only':       (250, 0),
    'low_aug':         (250, 125),
    'moderate_aug':    (250, 250),
    'high_aug':        (250, 500),
    'synthetic_only':  (0, 500),
}

if __name__ == '__main__':
    run_all()
