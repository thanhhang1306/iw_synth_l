import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# configuration
METRICS = [
    'weighted_f1', 'macro_f1', 'accuracy', 'roc_auc', 'cohen_kappa',
    'top2_accuracy', 'top3_accuracy', 'mrr', 'map'
]
TOP_N = 10
IN_FILE = 'overall_metrics.csv'
VAR_MAP = {
    1: 'Back-translation baseline', 2: 'Base LLM + Lexicon', 3: 'Base LLM + MT',
    4: 'In-language LLM (mT5-LR)', 5: 'Seeds + LLM + Lex', 6: 'Seeds + LLM + MT',
    7: 'V5 + QA', 8: 'V6 + QA', 9: 'V5 + CoT', 10: 'V6 + CoT',
    11: 'V7 + CoT (full Lex path)', 12: 'V8 + CoT (full MT path)'
}
OUT_DIR = 'paper_plots'
TABLE_DIR = 'reports'
SECTION2_DIR = os.path.join(OUT_DIR,'section2')
SECTION4_DIR = os.path.join(OUT_DIR,'section4')
for d in [OUT_DIR, TABLE_DIR, SECTION2_DIR, SECTION4_DIR]:
    os.makedirs(d, exist_ok=True)

# load and annotate data
df = pd.read_csv(IN_FILE)
df['variant_name'] = df['variant'].map(VAR_MAP)

# plot top-n best and worst for each metric
fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 3*len(METRICS)))
plt.subplots_adjust(hspace=0.6, wspace=0.4)
for i, metric in enumerate(METRICS):
    # best runs
    best = df.nlargest(TOP_N, metric)
    labels_best = best['variant_name'] + '\n' + best['experiment']
    axes[i,0].bar(labels_best, best[metric], color='tab:blue')
    axes[i,0].set_title(f"{metric.replace('_',' ').title()} — top {TOP_N} best")
    axes[i,0].tick_params(axis='x', rotation=45, labelsize=8)
    y0, y1 = best[metric].min(), best[metric].max()
    axes[i,0].set_ylim(y0*0.995, y1*1.005)
    axes[i,0].set_ylabel(metric.replace('_',' ').title())
    # worst runs
    worst = df.nsmallest(TOP_N, metric)
    labels_worst = worst['variant_name'] + '\n' + worst['experiment']
    axes[i,1].bar(labels_worst, worst[metric], color='tab:red')
    axes[i,1].set_title(f"{metric.replace('_',' ').title()} — top {TOP_N} worst")
    axes[i,1].tick_params(axis='x', rotation=45, labelsize=8)
    y0_w, y1_w = worst[metric].min(), worst[metric].max()
    axes[i,1].set_ylim(y0_w*0.995, y1_w*1.005)
# save figure
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR,'figure_top10_best_worst.png'), dpi=300)
plt.close(fig)
print(f"figure saved to {OUT_DIR}/figure_top10_best_worst.png")

# best and worst weighted_f1 per variant
best_idx = df.groupby('variant')['weighted_f1'].idxmax()
worst_idx = df.groupby('variant')['weighted_f1'].idxmin()
tbl_best = df.loc[best_idx].assign(rank='best')
tbl_worst = df.loc[worst_idx].assign(rank='worst')
table1 = (
    pd.concat([tbl_best, tbl_worst])
      .sort_values(['variant','rank'])
      [['variant','variant_name','rank','experiment'] + METRICS]
)
table1.to_csv(os.path.join(TABLE_DIR,'table1_best_and_worst.csv'), index=False)
print(table1.to_markdown(index=False))

# mean performance by variant in 2x3 grid
mean_df = (
    df.groupby(['variant','variant_name'], as_index=False)[METRICS]
      .mean().sort_values('variant')
)
fig, axes = plt.subplots(4, 3, figsize=(14,14))
axes = axes.flatten()
palette = sns.color_palette('Set3', n_colors=len(METRICS))
for i, metric in enumerate(METRICS):
    if i >= len(axes)-1: break
    ax = axes[i]
    sns.barplot(
        data=mean_df, x='variant', y=metric,
        color=palette[i], width=0.6, ax=ax
    )
    ax.margins(x=0.1)
    for p in ax.patches:
        h = p.get_height()
        ax.text(p.get_x()+p.get_width()/2, h+0.002, f"{h:.3f}",
                ha='center', va='bottom', fontsize=8)
    ax.set_title(metric.replace('_',' ').title(), fontsize=12)
    ax.set_xlabel('')
    ax.set_xticks(mean_df['variant']-1)
    ax.set_xticklabels([f"({v})" for v in mean_df['variant']], rotation=0, fontsize=9)
# remove unused subplot
fig.delaxes(axes[-1])
axes[0].set_ylabel('Mean Score', fontsize=12)
fig.suptitle('Mean Performance by Variant', fontsize=16, y=1.02)
plt.tight_layout()
out2a = os.path.join(SECTION2_DIR,'figure2a_2x3_grid.png')
fig.savefig(out2a, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"saved figure2a → {out2a}")

# performance trajectories by variant
experiments = ['real_only','low_aug','moderate_aug','high_aug','synthetic_only']
variants = sorted(df['variant'].unique())
rows = (len(variants)+3)//4
fig, axes = plt.subplots(rows,4, figsize=(16,4*rows))
axes = axes.flatten()
for ax, var in zip(axes, variants):
    sub = df[df['variant']==var].copy()
    sub['experiment'] = pd.Categorical(sub['experiment'], categories=experiments, ordered=True)
    sub = sub.sort_values('experiment')
    for metric in METRICS:
        sns.lineplot(data=sub, x='experiment', y=metric, marker='o', ax=ax, label=metric)
    ax.set_title(f"({var}) {VAR_MAP[var]}", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    ax.legend(fontsize=6, loc='upper right')
# remove empty
for ax in axes[len(variants):]: fig.delaxes(ax)
plt.tight_layout()
out2b = os.path.join(SECTION2_DIR,'figure2b_line_multiples.png')
fig.savefig(out2b, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"saved figure2b → {out2b}")

# weighted f1 distribution by augmentation
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='experiment', y='weighted_f1', order=experiments, palette='Set3')
sns.stripplot(data=df, x='experiment', y='weighted_f1', order=experiments,
              color='gray', size=4, jitter=True, alpha=0.6)
plt.xlabel('Augmentation Regime')
plt.ylabel('Weighted F1 Score')
plt.title('Distribution of Weighted F1 Across Variants')
plt.xticks(rotation=30)
plt.tight_layout()
out2c = os.path.join(SECTION2_DIR,'figure2c_weighted_f1_boxplot.png')
plt.savefig(out2c, dpi=300)
plt.close()
print(f"saved figure2c → {out2c}")

# per-class f1 heatmaps and confusion & precision-recall
df_per = pd.read_csv('perclass_metrics.csv')
df_per['variant_name'] = df_per['variant'].map(VAR_MAP)
# per-class f1 heatmaps
for cls in [1,2,3,4]:
    pivot = (df_per[df_per['class']==cls]
             .pivot(index='variant_name',columns='experiment',values='f1')
             .reindex(columns=experiments))
    plt.figure(figsize=(8,10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label':'F1'})
    plt.title(f'per-class f1 heatmap class {cls}')
    plt.xlabel('augmentation')
    plt.ylabel('variant')
    plt.tight_layout()
    fn = os.path.join(SECTION4_DIR,f'heatmap_f1_{cls}.png')
    plt.savefig(fn, dpi=300)
    plt.close()
    print(f"saved {fn}")
    
# confusion matrix gallery
best = df.loc[df.groupby('variant')['weighted_f1'].idxmax()]
rows = (len(best)+3)//4
fig, axes = plt.subplots(rows,4, figsize=(4*4,4*rows))
axes = axes.flatten()
class_labels = [1,2,3,4]
for ax, (_,row) in zip(axes, best.iterrows()):
    cm = json.loads(row['conf_mat'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f"{row['variant_name']} @ {row['experiment']}")
for ax in axes[len(best):]: fig.delaxes(ax)
plt.tight_layout()
fn_cm = os.path.join(SECTION4_DIR,'confusion_matrices_gallery.png')
plt.savefig(fn_cm, dpi=300)
plt.close()
print(f"saved {fn_cm}")

# precision-recall scatter plots
fig, axes = plt.subplots(2,2, figsize=(15,10))
axes = axes.flatten()
for ax, cls in zip(axes, [1,2,3,4]):
    pts = df_per[df_per['class']==cls]
    sns.scatterplot(data=pts, x='recall', y='precision', hue='variant',
                    style='experiment', ax=ax, palette='Set3', s=100)
    ax.set_title(f'precision vs recall class {cls}')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
plt.tight_layout()
fn_pr = os.path.join(SECTION4_DIR,'precision_recall_tradeoffs.png')
plt.savefig(fn_pr, dpi=300)
plt.close()
print(f"saved {fn_pr}")
