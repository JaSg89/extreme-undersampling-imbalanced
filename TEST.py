import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
import re


df_feus = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/FEUS_depurada/ALL_MODELS_FEUS_summary.csv')
df_meus = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/MEUS/ALL_MODELS_MEUS_summary.csv')
df_smote = pd.read_csv('C:/Users/Diana/Desktop/data_balance/Recall_scenario2/SMOTE/ALL_MODELS_SMOTE_summary.csv')
df_nearmiss = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/NearMiss/ALL_MODELS_NearMiss_summary.csv')
df_rus = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/RUS/ALL_MODELS_RUS_summary.csv')
df_enn = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/ENN/ALL_MODELS_ENN_summary.csv')
df_tl = pd.read_csv('C:/Users/Diana/Desktop/data_balance_IEEE/TL/ALL_MODELS_TL_summary.csv')


recall_data = pd.DataFrame({
    'FEUS': df_feus['recall_NN_FEUS'].head(30),
    'MEUS': df_meus['recall_NN_MEUS'].head(30),
    'SMOTE': df_smote['recall_NN_SMOTE'].head(30),
    'NearMiss': df_nearmiss['recall_NN_NearMiss'].head(30),
    'RUS': df_rus['recall_NN_RUS'].head(30),
    'ENN': df_enn['recall_NN_ENN'].head(30),
    'TL': df_tl['recall_NN_TL'].head(30)
})

precision_data = pd.DataFrame({
    'FEUS': df_feus['precision_NN_FEUS'].head(30),
    'MEUS': df_meus['precision_NN_MEUS'].head(30),
    'SMOTE': df_smote['precision_NN_SMOTE'].head(30),
    'NearMiss': df_nearmiss['precision_NN_NearMiss'].head(30),
    'RUS': df_rus['precision_NN_RUS'].head(30),
    'ENN': df_enn['precision_NN_ENN'].head(30),
    'TL': df_tl['precision_NN_TL'].head(30)
})

# --- Verificación ---
print(f"Filas en Recall Data: {len(recall_data)}")
print(f"Filas en SMOTE: {len(smote_recall)}")


stat, p_value = friedmanchisquare(*[recall_data[col] for col in recall_data.columns])

print(f'Estadístico de Friedman: {stat:.4f}')
print(f'P-valor: {p_value:.4f}')

if p_value < 0.05:
    print("Rechazamos la hipótesis nula: Hay una diferencia estadísticamente significativa entre las técnicas.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia de una diferencia significativa.")


stat, p_value = friedmanchisquare(*[precision_data[col] for col in recall_data.columns])

print(f'Estadístico de Friedman: {stat:.4f}')
print(f'P-valor: {p_value:.4f}')

if p_value < 0.05:
    print("Rechazamos la hipótesis nula: Hay una diferencia estadísticamente significativa entre las técnicas.")
else:
    print("No podemos rechazar la hipótesis nula: No hay evidencia de una diferencia significativa.")


# --- Test de Friedman (Este ya funcionaba bien) ---
stat, p_value = friedmanchisquare(*[recall_data[col] for col in recall_data.columns])
print(f'Estadístico de Friedman: {stat:.4f}')
print(f'P-valor: {p_value:.4f}')


print("\nRealizando Test Post-Hoc de Nemenyi...")
posthoc_df = sp.posthoc_nemenyi_friedman(recall_data)

print("\nTabla de P-valores del Test Post-Hoc de Nemenyi:")
print(posthoc_df)

# --- Visualización (Sin cambios) ---
plt.figure(figsize=(8, 6))
sns.heatmap(posthoc_df, annot=True, cmap='coolwarm', fmt=".3f", cbar_kws={'label': 'p-value'})
plt.title('Mapa de Calor de P-valores para Comparaciones por Pares (Nemenyi)')
plt.show()


# --- Test de Friedman (Este ya funcionaba bien) ---
stat, p_value = friedmanchisquare(*[precision_data[col] for col in precision_data.columns])
print(f'Estadístico de Friedman: {stat:.4f}')
print(f'P-valor: {p_value:.4f}')


print("\nRealizando Test Post-Hoc de Nemenyi...")
posthoc_df = sp.posthoc_nemenyi_friedman(precision_data)

print("\nTabla de P-valores del Test Post-Hoc de Nemenyi:")
print(posthoc_df)

# --- Visualización (Sin cambios) ---
plt.figure(figsize=(8, 6))
sns.heatmap(posthoc_df, annot=True, cmap='coolwarm', fmt=".3f", cbar_kws={'label': 'p-value'})
plt.title('Mapa de Calor de P-valores para Comparaciones por Pares (Nemenyi)')
plt.show()

from sklearn.metrics import precision_recall_curve

# ─────────────────────────────────────────────────────────────────────────────
# 1) Paired-scatter plot: Precision vs. Recall per run
# ─────────────────────────────────────────────────────────────────────────────
sns.set_palette("colorblind")
plt.figure(figsize=(8, 8))
for tech in recall_data.columns:
    plt.scatter(
        recall_data[tech],
        precision_data[tech],
        alpha=0.7,
        label=tech
    )

lims = [0, 1]
plt.plot(lims, lims, '--', color='gray', linewidth=1)
plt.xlim(lims)
plt.ylim(lims)

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision vs. Recall Across 30 Runs', fontsize=16)
plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2) 95 % Bootstrap CIs for Mean Recall & Precision
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(arr, n_boot=2000, ci=95):
    """Percentile bootstrap on the mean."""
    boot_means = np.random.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return arr.mean(), lower, upper

ci_records = []
for tech in recall_data.columns:
    mu_r, lo_r, hi_r = bootstrap_ci(recall_data[tech].values)
    mu_p, lo_p, hi_p = bootstrap_ci(precision_data[tech].values)
    ci_records.append({
        'Technique'          : tech,
        'Mean Recall'        : mu_r,
        'Recall CI Lower'    : lo_r,
        'Recall CI Upper'    : hi_r,
        'Mean Precision'     : mu_p,
        'Precision CI Lower' : lo_p,
        'Precision CI Upper' : hi_p
    })

ci_df = pd.DataFrame(ci_records).set_index('Technique')
print(ci_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Recall barplot
ax1.bar(
    ci_df.index,
    ci_df['Mean Recall'],
    yerr=[ci_df['Mean Recall'] - ci_df['Recall CI Lower'],
          ci_df['Recall CI Upper'] - ci_df['Mean Recall']],
    capsize=4
)
ax1.set_title('Mean Recall ±95% CI')
ax1.set_ylabel('Recall')
ax1.tick_params(axis='x', rotation=45)

# Precision barplot
ax2.bar(
    ci_df.index,
    ci_df['Mean Precision'],
    yerr=[ci_df['Mean Precision'] - ci_df['Precision CI Lower'],
          ci_df['Precision CI Upper'] - ci_df['Mean Precision']],
    capsize=4
)
ax2.set_title('Mean Precision ±95% CI')
ax2.set_ylabel('Precision')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


import seaborn as sns

# 1) Define the techniques in the exact order of your columns
techniques = recall_data.columns.tolist()

# 2) Create a color‐blind–safe palette of the same length
palette = sns.color_palette("colorblind", n_colors=len(techniques))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6), dpi=300)

for tech, color in zip(techniques, palette):
    ax.scatter(
        recall_data[tech],
        precision_data[tech],
        label=tech,
        alpha=0.7,
        s=50,
        color=color,
        edgecolor='k',
        linewidth=0.3
    )

# 45° reference line (no legend entry)
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, label='_')

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel('Recall',    fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('Precision vs. Recall Across 30 Runs', fontsize=16)
ax.legend(title='Technique', fontsize=12, title_fontsize=12, loc='lower left')
plt.tight_layout()

# Save for your paper
fig.savefig('paired_scatter.png', dpi=300, bbox_inches='tight')
fig.savefig('paired_scatter.pdf', bbox_inches='tight')
plt.show()



from matplotlib.ticker import FormatStrFormatter

# ─────────────────────────────────────────────────────────────────────────────
# 3) Boxplots for Recall and Precision across the 30 runs
# ─────────────────────────────────────────────────────────────────────────────

# Pasar a formato largo
recall_long = recall_data.melt(var_name='Technique', value_name='Recall')
precision_long = precision_data.melt(var_name='Technique', value_name='Precision')

# Orden fijo de las técnicas
order = ['FEUS', 'MEUS', 'SMOTE', 'NearMiss', 'RUS', 'ENN', 'TL']

# Paleta fija: mismo color por técnica en ambos gráficos
palette = {
    'FEUS': '#1f77b4',
    'MEUS': '#ff7f0e',
    'SMOTE': '#2ca02c',
    'NearMiss': '#d62728',
    'RUS': '#9467bd',
    'ENN': '#8c564b',
    'TL': '#e377c2'
}

# Tamaño exacto: 4027x1320 píxeles
dpi = 300
fig_width = 4027 / dpi
fig_height = 1320 / dpi

fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)

# Misma cantidad de ticks en Y, pero no necesariamente mismo rango
n_ticks = 6

# Límites para Recall
recall_min = recall_long['Recall'].min()
recall_max = recall_long['Recall'].max()
recall_pad = 0.02
recall_ymin = max(0, recall_min - recall_pad)
recall_ymax = min(1, recall_max + recall_pad)
recall_ticks = np.linspace(recall_ymin, recall_ymax, n_ticks)

# Límites para Precision
precision_min = precision_long['Precision'].min()
precision_max = precision_long['Precision'].max()
precision_pad = 0.02
precision_ymin = max(0, precision_min - precision_pad)
precision_ymax = min(1, precision_max + precision_pad)
precision_ticks = np.linspace(precision_ymin, precision_ymax, n_ticks)

# -------------------- Recall --------------------
sns.boxplot(
    data=recall_long,
    x='Technique',
    y='Recall',
    order=order,
    palette=palette,
    ax=axes[0],
    showfliers=False
)


axes[0].set_title('')
axes[0].set_xlabel('Resampling Technique', fontsize=12)
axes[0].set_ylabel('Recall', fontsize=12)
axes[0].set_ylim(recall_ymin, recall_ymax)
axes[0].set_yticks(recall_ticks)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[0].tick_params(axis='x', labelsize=9, rotation=0)
axes[0].tick_params(axis='y', labelsize=10)

# -------------------- Precision --------------------
sns.boxplot(
    data=precision_long,
    x='Technique',
    y='Precision',
    order=order,
    palette=palette,
    ax=axes[1],
    showfliers=False
)



axes[1].set_title('')
axes[1].set_xlabel('Resampling Technique', fontsize=12)
axes[1].set_ylabel('Precission', fontsize=12)
axes[1].set_ylim(precision_ymin, precision_ymax)
axes[1].set_yticks(precision_ticks)
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].tick_params(axis='x', labelsize=9, rotation=0)
axes[1].tick_params(axis='y', labelsize=10)

plt.tight_layout()

# Guardar
plt.savefig('boxplots_recall_precision.png', dpi=dpi, bbox_inches=None)
plt.savefig('boxplots_recall_precision.pdf', dpi=dpi, bbox_inches=None)

plt.show()


