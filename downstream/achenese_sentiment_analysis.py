from transformers import TrainerCallback, TrainerControl, TrainerState
import os
import csv
import json
import random
import numpy as np
import pandas as pd
import torch
import nltk
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from nltk import ngrams, word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.special import softmax
from transformers import EarlyStoppingCallback
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# set random seed for reproducibility
torch.manual_seed(100)
np.random.seed(100)
random.seed(100)

SYNTHETIC_METHOD = "variant_12_final"
SYNTHETIC_FILE = "scripts/fine_tune/variant_12_final.csv"

# define output directories for plots, training results, and models
BASE_DIR = "/scratch/gpfs/de7281"
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
nltk.data.path.append(NLTK_DATA_DIR)

HF_CACHE = os.getenv("HF_CACHE", os.path.join(BASE_DIR, "huggingface"))
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_DIR, "hf_datasets")
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

dirs = {
    "plots": os.path.join(BASE_DIR, "final_sentiment", "hang_plots", "ace", SYNTHETIC_METHOD),
    "results": os.path.join(BASE_DIR, "final_sentiment", "hang_results", "ace", SYNTHETIC_METHOD),
    "models": os.path.join(BASE_DIR, "final_sentiment", "hang_models", "ace", SYNTHETIC_METHOD)
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

PLOT_DIR = dirs["plots"]
TRAINING_OUTPUT_DIR = dirs["results"]
MODEL_SAVE_DIR = dirs["models"]

# helper function: standardize sentiment labels
def standardize_label(label):
    if isinstance(label, str):
        label_lower = label.lower().strip()
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        return mapping.get(label_lower, label)
    return label

# data analysis functions
def download_nltk_resources():
    """
    attempt to download nltk resources, but warn and continue on failure.
    """
    for res in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            try:
                nltk.download(res, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception as e:
                print(f"warning: could not download nltk resource '{res}': {e}")


def load_and_process_sentiment_data():
    # load synthetic data
    synthetic_df = pd.read_csv(SYNTHETIC_FILE)
    synthetic_df = synthetic_df.rename(columns={"sentiment": "label", "ace_text": "text"})
    synthetic_df = synthetic_df[["label", "text"]]
    synthetic_df["label"] = synthetic_df["label"].apply(standardize_label)

    # load real data
    real_data = load_dataset("indonlp/NusaX-senti", "ace", split="train")
    real_df = real_data.to_pandas()[["text", "label"]]
    real_df = real_df.sample(n=500, random_state=100)

    return real_df, synthetic_df


def display_dataset_info_sentiment(df):
    print("=== combined data preview ===")
    print(df.head(10))
    print("\n=== data info ===")
    print(df.info())
    print("\n=== class distribution ===")
    print(df['label'].value_counts())


def tokenize_and_display_frequency(df):
    all_text = " ".join(df["text"].dropna().tolist())
    tokens = word_tokenize(all_text.lower())
    fdist = FreqDist(tokens)
    print("\nmost common words:")
    print(fdist.most_common(30))
    return tokens


def plot_class_distribution(df, label_names, save=False, fname="analysis_class_distribution.png", plot_dir=None):
    df['label_name'] = df['label'].map(label_names)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='label_name', palette='viridis')
    plt.title('class distribution of sentiment labels')
    plt.xlabel('sentiment')
    plt.ylabel('number of samples')
    if save:
        out_dir = plot_dir or PLOT_DIR
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
    else:
        plt.show()


def generate_word_cloud(data, sentiment_label, label_names, save=False, fname_prefix="analysis_wordcloud_", plot_dir=None):
    subset = data[data['label'] == sentiment_label]
    text = " ".join(subset['text'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'word cloud for {label_names[sentiment_label]} sentiment')
    if save:
        out_dir = plot_dir or PLOT_DIR
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{fname_prefix}{label_names[sentiment_label]}.png"
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
    else:
        plt.show()


def plot_ngrams(data, sentiment_label, label_names, n=2, top_k=10, save=False, fname_prefix="analysis_ngrams_", plot_dir=None):
    subset = data[data['label'] == sentiment_label]
    text = " ".join(subset['text'].dropna().tolist()).lower()
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens, n))
    ngram_freq = Counter(n_grams)
    common_ngrams = ngram_freq.most_common(top_k)
    if common_ngrams:
        ngram_labels, counts = zip(*common_ngrams)
        ngram_labels = [" ".join(ngram) for ngram in ngram_labels]
    else:
        ngram_labels, counts = [], []
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(ngram_labels), palette='viridis', dodge=False)
    ngram_type = "bigrams" if n == 2 else "trigrams"
    plt.title(f'top {top_k} {ngram_type} for {label_names[sentiment_label]} sentiment')
    plt.xlabel('frequency')
    plt.ylabel('n-grams')
    if save:
        out_dir = plot_dir or PLOT_DIR
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{fname_prefix}{label_names[sentiment_label]}_{ngram_type}.png"
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
    else:
        plt.show()


def run_data_analysis_sentiment():
    download_nltk_resources()
    real_df, synthetic_df = load_and_process_sentiment_data()

    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=100).reset_index(drop=True)
    print("combined dataset shape:", combined_df.shape)

    display_dataset_info_sentiment(combined_df)
    tokenize_and_display_frequency(combined_df)

    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    plot_class_distribution(combined_df, label_names, save=True, fname="analysis_class_distribution.png", plot_dir=PLOT_DIR)
    for label in label_names:
        print(f"\n--- analysis for {label_names[label]} sentiment ---")
        generate_word_cloud(combined_df, label, label_names, save=True, fname_prefix="analysis_wordcloud_", plot_dir=PLOT_DIR)
        plot_ngrams(combined_df, label, label_names, n=2, top_k=10, save=True, fname_prefix="analysis_bigram_", plot_dir=PLOT_DIR)
        plot_ngrams(combined_df, label, label_names, n=3, top_k=10, save=True, fname_prefix="analysis_trigram_", plot_dir=PLOT_DIR)

# prepare experiment splits for sentiment analysis
def prepare_experiment_splits_sentiment():
    """
    prepares six training splits for sentiment experiments.
    """
    real_df, synthetic_df = load_and_process_sentiment_data()

    real_core = real_df.sample(n=500, random_state=100)
    splits = {
        "real_only":      (500,    0),
        "low_aug":        (500,  250),
        "moderate_aug":   (500,  500),
        "high_aug":       (500, 1000),
        "full_aug":       (500, 1500),
        "synthetic_only": (  0, 1500),
    }

    for name, (n_real, n_syn) in splits.items():
        train_real = real_core.sample(n=n_real, random_state=100) if n_real > 0 else pd.DataFrame(columns=real_core.columns)
        train_synth = synthetic_df.sample(n=n_syn, random_state=100)
        train_df = pd.concat([train_real, train_synth], ignore_index=True).sample(frac=1, random_state=100).reset_index(drop=True)
        train_df.to_csv(os.path.join(TRAINING_OUTPUT_DIR, f"train_{name}.csv"), index=False)
        print(f"saved split '{name}' with {len(train_real)} real + {len(train_synth)} synthetic examples.")

    val = load_dataset("indonlp/NusaX-senti", "ace", split="validation")
    test = load_dataset("indonlp/NusaX-senti", "ace", split="test")
    val.to_pandas()[["text", "label"]].to_csv(os.path.join(TRAINING_OUTPUT_DIR, "val_real.csv"), index=False)
    test.to_pandas()[["text", "label"]].to_csv(os.path.join(TRAINING_OUTPUT_DIR, "test_real.csv"), index=False)
    print("saved real-only validation and test splits.")

# preprocessing and label encoding for sentiment ml
def ml_preprocess_function(examples, tokenizer):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    if "label" in examples:
        tokenized["labels"] = examples["label"]
    return tokenized


def encode_labels_fn(example, label2id):
    example["labels"] = label2id[example["label"]]
    return example

# updated compute metrics function for sentiment
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    wp, wr, wf, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)
    try:
        probs = softmax(logits, axis=-1)
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except:
        roc_auc = None
    kappa = cohen_kappa_score(labels, preds)
    return {
        "accuracy": accuracy,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf,
        "roc_auc": roc_auc,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist()
    }

# custom callback for logging (sentiment)
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_logs = []

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history:
            last_log = state.log_history[-1]
            self.epoch_logs.append(last_log)
            print(f"\n=== epoch {state.epoch:.0f} summary ===")
            for k, v in last_log.items():
                print(f"{k}: {v}")
            print("=" * 30)
        return control

# define experiments dictionary for sentiment
EXPERIMENTS = {
    "real_only":      "train_real_only.csv",
    "low_aug":        "train_low_aug.csv",
    "moderate_aug":   "train_moderate_aug.csv",
    "high_aug":       "train_high_aug.csv",
    "full_aug":       "train_full_aug.csv",
    "synthetic_only": "train_synthetic_only.csv",
}

# experiment runner for sentiment
def run_experiment_sentiment(exp_name, train_csv):
    exp_plot_dir = os.path.join(PLOT_DIR, exp_name)
    exp_training_output_dir = os.path.join(TRAINING_OUTPUT_DIR, exp_name)
    exp_model_save_dir = os.path.join(MODEL_SAVE_DIR, exp_name)
    os.makedirs(exp_plot_dir, exist_ok=True)
    os.makedirs(exp_training_output_dir, exist_ok=True)
    os.makedirs(exp_model_save_dir, exist_ok=True)

    print(f"\n--- starting sentiment experiment: {exp_name} ---")
    print("training file:", train_csv)

    train_df = pd.read_csv(os.path.join(TRAINING_OUTPUT_DIR, train_csv))
    print("selected training split:", train_csv, "with", len(train_df), "samples.")

    val_df = pd.read_csv(os.path.join(TRAINING_OUTPUT_DIR, "val_real.csv"))
    test_df = pd.read_csv(os.path.join(TRAINING_OUTPUT_DIR, "test_real.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)

    def tokenize_fn(examples):
        return ml_preprocess_function(examples, tokenizer)

    remove_columns = ["text", "label"]
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=exp_training_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        save_only_model=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LoggingCallback()]
    )

    trainer.train()
    final_dir = os.path.join(exp_model_save_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    eval_results = trainer.evaluate(test_dataset)
    print("=== evaluation results on test set for", exp_name, "===")
    print(eval_results)

    results_path = os.path.join(exp_training_output_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    print(f"evaluation results saved to '{results_path}'")

    predictions_output = trainer.predict(test_dataset)
    logits, labels = predictions_output.predictions, predictions_output.label_ids
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, target_names=["negative", "neutral", "positive"])
    report_path = os.path.join(exp_training_output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"classification report saved to '{report_path}'")

    cm = confusion_matrix(labels, predictions)
    print("=== confusion matrix on test set for", exp_name, "===")
    print(cm)

    trainer.save_model(exp_model_save_dir)
    tokenizer.save_pretrained(exp_model_save_dir)
    print(f"model and tokenizer saved to '{exp_model_save_dir}'")

# main function
def run_all_experiments_sentiment():
    run_data_analysis_sentiment()
    prepare_experiment_splits_sentiment()
    for exp_name, train_csv in EXPERIMENTS.items():
        run_experiment_sentiment(exp_name, train_csv)

if __name__ == '__main__':
    print("=== running data analysis pipeline for sentiment ===")
    run_data_analysis_sentiment()
    print("\n=== running sentiment ml model training and evaluation pipeline ===")
    run_all_experiments_sentiment()
