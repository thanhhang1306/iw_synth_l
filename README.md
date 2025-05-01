# SYNTH-L: Synthetic Data Generation for Low-Resource Natural Language Processing
A repository storing the codebase for SYNTH-L.

## Acknowledgments
I acknowledge the use of OpenAI’s ChatGPT (Basic version) for occasional assistance in refining the syntactical structure and phrasing of the code. All core algorithmic design, implementation, and conceptual development were performed independently by me; ChatGPT’s role was limited to code polishing.

## Notes

All experiments were carried out on the Princeton Della high-performance computing cluster.  Due to the extensive size of the model repository (over 900 GB encompassing 130 distinct fine-tuned models) individual weight files and detailed configuration information are available upon request.  Please contact hang.pham@princeton.edu to obtain access.

---

## Repository Structure

- [`README.md`](README.md)
- [`data/`](data/)
  - [`achinese_sentiment_analysis/`](data/achinese_sentiment_analysis/)
    - [`variant_1_data.csv`](data/achinese_sentiment_analysis/variant_1_data.csv)
    - [`variant_2_data.csv`](data/achinese_sentiment_analysis/variant_2_data.csv)
    - …  
    - [`variant_12_data.csv`](data/achinese_sentiment_analysis/variant_12_data.csv)
  - [`tigrinya_question_answering/`](data/tigrinya_question_answering/)
    - [`variant_1_data.csv`](data/tigrinya_question_answering/variant_1_data.csv)
    - [`variant_2_data.csv`](data/tigrinya_question_answering/variant_2_data.csv)
    - …  
    - [`variant_12_data.csv`](data/tigrinya_question_answering/variant_12_data.csv)

- [`generation/`](generation/)
  - [`achinese_sentiment_analysis/`](generation/achinese_sentiment_analysis/)
  - [`tigrinya_question_answering/`](generation/tigrinya_question_answering/)

- [`intrinsics/`](intrinsics/)
  - [`achinese_sentiment_analysis/`](intrinsics/achinese_sentiment_analysis/)
  - [`tigrinya_question_answering/`](intrinsics/tigrinya_question_answering/)

- [`extrinsics/`](extrinsics/)
  - [`achinese_sentiment_analysis/`](extrinsics/achinese_sentiment_analysis/)
  - [`tigrinya_question_answering/`](extrinsics/tigrinya_question_answering/)

- [`downstream/`](downstream/)
  - [`achinese_sentiment_analysis.py`](downstream/achinese_sentiment_analysis.py)
  - [`tirginya_question_answering.py`](downstream/tirginya_question_answering.py)
  - [`results/`](downstream/results/)
  - [`plots/`](downstream/plots/)


---

## generation/ Overview

### Achenese Sentiment Analysis

- `generate_var_1.py`: Variant 1 (back-translation baseline)
- `generate_var_2_3.py`: Variants 2 & 3 (zero-shot LLM generation + label verification)
- `generate_var_4.py`: Variant 4 (in-language LLM generation)
- `generate_var_5_6.py`: Variants 5 & 6 (LLM + lexicon or MT, no QA or CoT)
- `generate_var_7_8.py`: Variants 7 & 8 (add QA filter)
- `generate_var_9_10.py`: Variants 9 & 10 (add CoT prompting)
- `generate_var_11_12.py`: Variants 11 & 12 (full Lex/MT + QA + CoT)
- `filter_achenese.py`: quality assurance via psuedo-perplexity for examples in Achenese
- `translate_eng_lex.py`: lexicon-based English→Acehnese substitution
- `translate_eng_mt.py`: machine translation English→Acehnese via Google Translate
- `translate_var_4.py`: back-translate variant 4 outputs to English for verification
- `merge_files.py`: merge generated data files into a single file

### Tigrinya Question Answering

- `extract_tir_passage.py`: translate SQuAD passages to Tigrinya for QA seeding
- `filter_tigrinya.py`: quality assurance for QA examples in Tigrinya 
- `generate_var_1.py` … `generate_var_11_12.py`: same variant numbering as above, adapted for QA
- `translate_eng_lex.py`: lexicon-based English→Tigrinya
- `translate_eng_mt.py`: machine translation English→Tigrinya
- `translate_var_4.py`: back-translate variant 4 QA examples
- `passage_tir.jsonl` / `passage_tir.fixed.jsonl`: raw and cleaned passages for prompting

---

## intrinsics/ Overview

### Achenese Sentiment Analysis Intrinsics

- `intrinsics_sentiment.py`: compute and aggregate statistics per variant
- `final_stats.csv`: consolidated intrinsic metrics
- `final_stats_variant_1.csv` … `final_stats_variant_12.csv`: intrinsic metrics per variant 

### Tigrinya QA Intrinsics

- `intrinsics_question_answering.py`: compute and aggregate statistics per variant
- `final_stats.csv`: consolidated intrinsic metrics
- `final_stats_variant_1.csv` … `final_stats_variant_12.csv`: intrinsic metrics per variant 

---

## extrinsics/ Overview

### Achenese Sentiment Analysis Extrinsics

- `general_stats.py`: compute overall model performance stats and plots
- `overall_metrics.csv`: aggregated metrics per variant and experiment
- `perclass_metrics.csv`: per-class performance metrics
- `paper_data.py`: code to generate graphs for the paper 
- `plots/`: exploratory visuals
- `plots_by_variant/`: metrics plots organized by variant
- `plots_by_class/`: per-class heatmaps
- `plots_by_cm/`: confusion matrix visualizations
- `reports/`: summary tables and logs

### Tigrinya QA Extrinsics

- `general_stats.py`: compute overall model performance stats and plots
- `overall_metrics.csv`: aggregated metrics per variant and experiment
- `perclass_metrics.csv`: per-class performance metrics
- `paper_data.py`: code to generate graphs for the paper 
- `plots/`: exploratory visuals
- `plots_by_variant/`: metrics plots organized by variant
- `plots_by_class/`: per-class heatmaps
- `plots_by_cm/`: confusion matrix visualizations
- `reports/`: summary tables and logs

---

## downstream/ Overview

- `achinese_sentiment_analysis.py`: sentiment analysis fine-tuning / downstream task
- `tirginya_question_answering.py`: QA fine-tuning / downstream task
- `results/`: files containing results per variant & regime combination
- `plots/`: files containing exploratory plots per variant & regime combination

---

Please refer to the paper for full methodological details, hyperparameters, and evaluation results.
