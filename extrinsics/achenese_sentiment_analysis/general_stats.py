import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# configuration
METRICS = ["weighted_f1", "macro_f1", "accuracy", "roc_auc", "cohen_kappa"]
TOP_N = 12
PLOT_DIR = "plots"
CSV_OVERALL = "overall_metrics.csv"
EXPERIMENTS = ["real_only", "low_aug", "moderate_aug", "high_aug", "full_aug", "synthetic_only"]

# ensure directories exist
os.makedirs("reports", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# redirect stdout to report file
with open("reports/summary.txt", "w") as log_file:
    orig_stdout = sys.stdout
    sys.stdout = log_file

    # load data
    overall_df = pd.read_csv(CSV_OVERALL)
    perclass_df = pd.read_csv("perclass_metrics.csv")

    # map variants to names
    variant_mapping = {
        1: "Back-translation baseline",
        2: "Base LLM + Lexicon",
        3: "Base LLM + MT",
        4: "In-language LLM (mT5-LR)",
        5: "Seeds + LLM + Lex",
        6: "Seeds + LLM + MT",
        7: "V5 + QA",
        8: "V6 + QA",
        9: "V5 + CoT",
        10: "V6 + CoT",
        11: "V7 + CoT (full Lex path)",
        12: "V8 + CoT (full MT path)",
    }
    overall_df['variant_name'] = overall_df['variant'].map(variant_mapping)
    perclass_df['variant_name'] = perclass_df['variant'].map(variant_mapping)
    VARIANTS = sorted(overall_df['variant'].unique())

    # build leaderboards
    leaderboards = {}
    for metric in METRICS:
        if metric not in overall_df.columns:
            print(f"'{metric}' not found in overall_metrics.csv, skipped")
            continue
        topn = overall_df.sort_values(metric, ascending=False).head(TOP_N).reset_index(drop=True)
        leaderboards[metric] = topn
        print(f"\n=== top {TOP_N} runs by {metric} ===")
        print(topn[['variant', 'experiment', metric]])
        topn.to_csv(f"top_{TOP_N}_{metric}.csv", index=False)
        print(f"saved top_{TOP_N}_{metric}.csv")

    # plot combined top metrics
    for metric, tbl in leaderboards.items():
        if tbl.empty:
            continue
        top = (
            tbl.sort_values(metric, ascending=False)
               .head(TOP_N)
               .assign(label=lambda df: df['variant'].astype(str) + "_" + df['experiment'])
               .reset_index(drop=True)
        )
        plt.figure(figsize=(15, 5))
        ax = sns.barplot(
            data=top,
            x='label',
            y=metric,
            palette=sns.color_palette('Set2', n_colors=TOP_N),
            dodge=False,
            legend=False
        )
        ax.set_title(f"top {TOP_N} runs by {metric.replace('_',' ').title()}")
        ax.set_xlabel('variant_experiment')
        ax.set_ylabel(metric.replace('_',' ').title())
        y0, y1 = top[metric].min(), top[metric].max()
        for bar in ax.patches:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (y1 - y0) * 0.005,
                f"{h:.3f}",
                ha='center', va='bottom', fontsize=8
            )
        plt.xticks(rotation=45, ha='right')
        ax.set_ylim(y0 * 0.995, y1 * 1.005)
        plt.subplots_adjust(bottom=0.30, left=0.15, right=0.95, top=0.90)
        out_path = os.path.join(PLOT_DIR, f"top_{TOP_N}_{metric}_combined.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"saved {out_path}")

    # per-experiment analysis
    for exp in EXPERIMENTS:
        sub = overall_df[overall_df['experiment'] == exp]
        if sub.empty:
            print(f"\nno data for experiment '{exp}', skipping")
            continue
        for metric in METRICS:
            if metric not in sub.columns:
                print(f"metric '{metric}' not in dataframe, skipped")
                continue
            sorted_sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)
            print(f"\n=== experiment '{exp}' — variants ranked by {metric} ===")
            print(sorted_sub[['variant', metric]].to_string(index=False))
            plt.figure(figsize=(10, 4))
            ax = sns.barplot(
                data=sorted_sub,
                x='variant',
                y=metric,
                palette=sns.color_palette('Set2', n_colors=len(sorted_sub)),
                dodge=False,
                legend=False
            )
            ax.set_title(f"{exp.replace('_',' ').title()} — {metric.replace('_',' ').title()}")
            ax.set_xlabel('variant')
            ax.set_ylabel(metric.replace('_',' ').title())
            y0, y1 = sorted_sub[metric].min(), sorted_sub[metric].max()
            for bar in ax.patches:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (y1 - y0) * 0.005,
                    f"{h:.3f}",
                    ha='center', va='bottom', fontsize=8
                )
            plt.xticks(rotation=45, ha='right')
            ax.set_ylim(y0 * 0.995, y1 * 1.005)
            plt.subplots_adjust(bottom=0.25, left=0.15, right=0.95, top=0.85)
            out_path = os.path.join(PLOT_DIR, f"{exp}_{metric}.png")
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"saved plot to {out_path}")

    # per-variant analysis
    for var in VARIANTS:
        sub = overall_df[overall_df['variant'] == var]
        if sub.empty:
            print(f"\nno rows for variant {var}, skipping")
            continue
        for metric in METRICS:
            if metric not in sub.columns:
                print(f"metric '{metric}' missing, skipped")
                continue
            ranked = (
                sub.sort_values(metric, ascending=False)
                   .reset_index(drop=True)[['experiment', metric]]
            )
            print(f"\n=== variant {var} — experiments ranked by {metric} ===")
            print(ranked.to_string(index=False))
        melt = sub.melt(
            id_vars='experiment',
            value_vars=METRICS,
            var_name='metric',
            value_name='value'
        )
        plt.figure(figsize=(15, 5))
        ax = sns.barplot(
            data=melt,
            x='experiment', y='value', hue='metric',
            hue_order=METRICS,
            palette='Set2', dodge=True
        )
        ax.set_title(f"variant {var} — all metrics across experiments")
        ax.set_xlabel('experiment')
        ax.set_ylabel('score')
        plt.xticks(rotation=30, ha='right')
        for p in ax.patches:
            h = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2,
                h + 0.002,
                f"{h:.3f}", ha='center', va='bottom', fontsize=7
            )
        ymin, ymax = melt['value'].min(), melt['value'].max()
        ax.set_ylim(ymin * 0.99, ymax * 1.01)
        plt.legend(title='metric', bbox_to_anchor=(1.02,1), loc='upper left')
        plt.subplots_adjust(bottom=0.25, left=0.08, right=0.82, top=0.88)
        fn = os.path.join(PLOT_DIR, f"variant_{var}_metrics.png")
        plt.savefig(fn, dpi=200)
        plt.close()
        print(f"saved plot {fn}")

        # per-class f1 heatmaps
        os.makedirs("plots_by_class", exist_ok=True)
        for cls in [1, 2, 3, 4]:
            sel = perclass_df[perclass_df['class'] == cls].copy()
            if sel.empty:
                print(f"\nno data for class {cls}, skipping heatmap")
                continue
            pivot = sel.pivot_table(index='variant', columns='experiment', values='f1')
            print(f"\n=== class {cls}-class f1 ===")
            print(pivot.to_string(float_format='%.3f'))
            plt.figure(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title(f"class {cls}-class f1")
            plt.savefig(f"plots_by_class/class_{cls}_f1_heatmap.png", dpi=200)
            plt.close()

        # confusion matrix gallery
        best = overall_df.loc[overall_df.groupby('variant')['macro_f1'].idxmax()]
        os.makedirs("plots_by_cm", exist_ok=True)
        for _, row in best.iterrows():
            cm = np.array(eval(row['conf_mat']))
            print(f"\n=== variant {row.variant} ({row.experiment}) confusion matrix ===")
            print(cm)
            plt.figure()
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"class {i}" for i in [1,2,3,4]],
                yticklabels=[f"class {i}" for i in [1,2,3,4]]
            )
            plt.title(f"variant {row.variant} — {row.experiment}")
            plt.savefig(f"plots_by_cm/variant_{row.variant}.png", dpi=200)
            plt.close()

        # precision vs recall scatter
        os.makedirs("plots", exist_ok=True)
        for cls in [1, 2, 3, 4]:
            pts = perclass_df[perclass_df['class'] == cls]
            print(f"\n=== class {cls}-class precision vs recall ===")
            print(
                pts[['variant', 'experiment', 'precision', 'recall']]
                   .sort_values(['precision', 'recall'], ascending=False)
                   .to_string(index=False, float_format='%.3f')
            )
            plt.figure(figsize=(6, 5))
            sns.scatterplot(
                data=pts,
                x='recall', y='precision',
                hue='experiment', style='variant',
                s=100, alpha=0.8
            )
            plt.title(f"class {cls} precision vs recall")
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            out_fn = f"plots/prec_rec_class_{cls}.png"
            plt.savefig(out_fn, dpi=200)
            plt.close()
            print(f"saved scatter {out_fn}")

    # restore stdout and close report
    sys.stdout = orig_stdout

# save variant summary
best = overall_df.loc[overall_df.groupby('variant')['weighted_f1'].idxmax()]
worst = overall_df.loc[overall_df.groupby('variant')['weighted_f1'].idxmin()]
summary = (
    pd.concat([best.assign(rank='best'), worst.assign(rank='worst')])
      .sort_values(['variant', 'rank'])
      .loc[:, ['variant_name','experiment','rank','weighted_f1','macro_f1','accuracy','roc_auc','cohen_kappa']]
)
summary.to_csv('reports/variant_summary.csv', index=False)
print(summary.to_markdown(index=False))
