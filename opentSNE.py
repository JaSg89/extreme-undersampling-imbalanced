import os
import logging
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from openTSNE import TSNE
from sklearn.neighbors import NearestNeighbors


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

# --- PARÁMETROS GLOBALES ---
CSV_PATH        = "C:/Users/Diana/Desktop/Credit_card_data/creditcard.csv"
TARGET_COLUMN   = "Class"
RANDOM_SEED     = 42
OUTPUT_DIR      = "C:/Users/Diana/Desktop/data_balance/FIGURAS/"


# ------------------------------------ MEUS (Majority-class Extreme UnderSampling) -------------------------------------


# --- PARÁMETROS ESPECÍFICOS PARA MEUS ---
TSNE_BALANCED   = os.path.join(OUTPUT_DIR, "tSNE_MEUS.png")

# Paso 2: Función MEUS (matching 1-a-1 real con Mahalanobis)
def apply_meus(X: pd.DataFrame, y: pd.Series, seed: int):
    data = X.copy()
    data[y.name] = y

    df_min = data[data[y.name] == 1].drop(columns=[y.name])
    df_maj = data[data[y.name] == 0].drop(columns=[y.name])
    if df_min.empty or df_maj.empty:
        logging.warning("Clases insuficientes; retorno original.")
        return X, y

    # 2.1 Calcular Sigma⁻¹
    cov     = np.cov(X.values, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    # 2.2 Ajustar K al tamaño de la mayoría
    K = min(150, len(df_maj))
    nn = NearestNeighbors(n_neighbors=K,
                          metric='mahalanobis',
                          metric_params={'VI': inv_cov})
    nn.fit(df_maj.values)

    # 2.3 Buscar vecinos para cada minoría
    _, neigh_idx = nn.kneighbors(df_min.values)

    # 2.4 Claiming 1-a-1
    claimed  = set()
    selected = []
    for i in range(len(df_min)):
        for j in neigh_idx[i]:
            idx = df_maj.index[j]
            if idx not in claimed:
                claimed.add(idx)
                selected.append(idx)
                break

    df_maj_sel = df_maj.loc[selected]

    # NUEVO: guardar eliminados
    removed_indices = list(set(df_maj.index) - set(selected))

    # 2.5 Reconstruir y barajar
    X_res = pd.concat([df_min, df_maj_sel], axis=0)
    y_res = pd.Series([1]*len(df_min) + [0]*len(df_maj_sel),
                      name=y.name, index=X_res.index)

    np.random.seed(seed)
    perm    = np.random.permutation(X_res.index)
    X_res   = X_res.loc[perm].reset_index(drop=True)
    y_res   = y_res.loc[perm].reset_index(drop=True)

    logging.info(f"MEUS → minoría: {y_res.sum()}, mayoría: {len(y_res)-y_res.sum()}")
    return X_res, y_res, selected, removed_indices


# Paso 3: Función de visualización t-SNE (solo balanced, sin título)
def plot_tsne_balanced(X: pd.DataFrame, y: pd.Series, out_path: str, X_all, y_all, selected_indices, removed_indices):
    if X.empty or y.empty:
        logging.error("No hay datos para plotear.")
        return

    # 3.1 Cálculo t-SNE
    start = time.time()
    tsne  = TSNE(n_components=2,
                 perplexity=25,
                 #n_iter=1500,
                 random_state=RANDOM_SEED,
                 verbose=1)
    #coords = tsne.fit_transform(X)

    # FIT SOLO CON BALANCEADOS
    embedding = tsne.fit(X.values)
    coords_selected = embedding

    # PROYECTAR ELIMINADOS (solo mayoría eliminada)
    X_removed = X_all.loc[removed_indices]
    coords_removed = embedding.transform(X_removed.values)


    # ROTACIÓN
    coords_selected = np.column_stack((coords_selected[:, 1], -coords_selected[:, 0]))
    coords_removed  = np.column_stack((coords_removed[:, 1],  -coords_removed[:, 0]))

    # MIRROR 
    coords_selected[:, 0] *= -1
    coords_removed[:, 0]  *= -1

    logging.info(f"t-SNE completado en {time.time() - start:.2f}s")

    # 3.2 Dibujar scatter
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap   = {0: '#0072B2', 1:'#D55E00' }
    #labels = {0: 'Majority Class (Non-Fraud)', 1: 'Minority Class (Fraud)'}
    labels = {0: 'Majority Class', 1: 'Minority Class'}


    # 🔴 eliminados
    ax.scatter(coords_removed[:, 0],
               coords_removed[:, 1],
               c='silver',
               alpha=0.35,
               s=10,
               label='Removed')


    for cls in [0, 1]:
        mask = (y.values == cls)
        ax.scatter(coords_selected[mask, 0],
                   coords_selected[mask, 1],
                   c=cmap[cls],
                   label=labels[cls],
                   alpha=0.8,
                   edgecolors='w',
                   s=50)

    # 3.3 Etiquetas de ejes en inglés y con fuente grande
    ax.set_xlabel('First t-SNE Dimension', fontsize=21)
    ax.set_ylabel('Second t-SNE Dimension', fontsize=21)

    # pocos ticks abajo del 0
    ax.set_ylim(-50, 50)
    ax.set_yticks([50, 37.5, 25, 12.5, 0, -12.5, -25, -37.5, -50])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    ax.tick_params(axis='both', which='major', labelsize=17)

    # leyenda limpia
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='lower left',
              fontsize=10,
              title="Classes",
              title_fontsize=10,
              frameon=True,
              facecolor='white',
              framealpha=1.0)

    ax.grid(True, linestyle='--', alpha=0.6)

    # 3.4 Guardar figura
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    logging.info(f"Figura guardada en: {out_path}")

    plt.show()
    plt.close(fig)


# Paso 4: Bloque principal
if __name__ == "__main__":
    # 4.1 Carga y split
    df         = pd.read_csv(CSV_PATH)
    X, y       = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # 4.2 Escalado
    scaler         = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # 4.3 Submuestreo MEUS
    logging.info("Aplicando MEUS para balancear clases…")
    X_bal, y_bal, selected_indices, removed_indices = apply_meus(X_train_scaled, y_train, seed=RANDOM_SEED)

    # 4.4 Visualizar SOLO el conjunto balanceado
    logging.info("Generando t-SNE de datos balanceados…")
    plot_tsne_balanced(
        X_bal,
        y_bal,
        TSNE_BALANCED,
        X_train_scaled,
        y_train,
        selected_indices,
        removed_indices
    )

    logging.info("Fin del proceso.")


# ------------------------------------ FEUS (Furthest-point Extreme UnderSampling) ------------------------------------

# --- PARÁMETROS ESPECÍFICOS PARA FEUS ---
N_SAMPLES_FEUS  = 5000 # Número de muestras a seleccionar con FEUS
TSNE_FEUS_PATH  = os.path.join(OUTPUT_DIR, "tSNE_FEUS.png")


# --- FUNCIÓN DE SUBMUESTREO FEUS ---
def apply_feus(X: pd.DataFrame, y: pd.Series, n_samples_to_keep: int):
    """
    Aplica FEUS (Furthest Under-Sampling) basado en la distancia de Mahalanobis.
    Selecciona las `n_samples_to_keep` muestras más lejanas del centroide del conjunto de datos.
    """
    if X.empty:
        logging.warning("El DataFrame de entrada está vacío. Devolviendo original.")
        return X, y

    logging.info("Calculando la distancia de Mahalanobis para cada punto...")
    mean_vector = X.mean().values
    try:
        inv_cov_matrix = np.linalg.pinv(np.cov(X.values, rowvar=False))
    except Exception as e:
        logging.error(f"Error al calcular la matriz de covarianza inversa: {e}. Abortando.")
        return X, y

    diff_values = X.values - mean_vector
    distances_sq = np.einsum('ij,ij->i', diff_values.dot(inv_cov_matrix), diff_values)
    distances = np.sqrt(np.maximum(distances_sq, 0))

    data_with_dist = X.copy()
    data_with_dist['mahalanobis_dist'] = distances
    
    n_to_keep = min(n_samples_to_keep, len(X))
    selected_indices = data_with_dist.nlargest(n_to_keep, 'mahalanobis_dist').index

    # NUEVO (guardar eliminados)
    all_indices = set(X.index)
    selected_set = set(selected_indices)
    removed_indices = list(all_indices - selected_set)

    X_sub = X.loc[selected_indices].reset_index(drop=True)
    y_sub = y.loc[selected_indices].reset_index(drop=True)

    logging.info(f"FEUS → Original: {len(X)}, Seleccionado: {len(X_sub)}")
    logging.info(f"Distribución de clases en el subconjunto: {dict(y_sub.value_counts())}")
    
    return X_sub, y_sub, removed_indices, selected_indices


# --- FUNCIÓN DE VISUALIZACIÓN T-SNE (CON CORRECCIÓN) ---
def plot_tsne(X: pd.DataFrame, y: pd.Series, out_path: str, X_all, y_all, selected_indices, removed_indices):
    """
    Calcula y grafica la proyección t-SNE de los datos.
    """
    if X.empty or y.empty:
        logging.error("No hay datos para plotear.")
        return

    start_time = time.time()
    tsne = TSNE(n_components=2,
                perplexity=30,
                #n_iter=1500,
                random_state=RANDOM_SEED,
                verbose=True)
    #coords = tsne.fit_transform(X)

    # 1. FIT SOLO CON SELECCIONADOS (como autor)
    embedding = tsne.fit(X.values)

    coords_selected = embedding

    # 2. PROYECTAR ELIMINADOS (clave)
    X_removed = X_all.loc[removed_indices]
    coords_removed = embedding.transform(X_removed.values)

    # Rotación 90° en sentido horario
    coords_selected = np.column_stack((coords_selected[:, 1], -coords_selected[:, 0]))
    coords_removed  = np.column_stack((coords_removed[:, 1],  -coords_removed[:, 0]))

    logging.info(f"t-SNE completado en {time.time() - start_time:.2f}s")

    # ----- PLOTEO -----
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap   = {0: '#0072B2', 1:'#D55E00'} 
    #labels = {0: 'Majority Class (Non-Fraud)', 1: 'Minority Class (Fraud)'}
    labels = {0: 'Majority Class', 1: 'Minority Class'}

    # 🔴 ELIMINADOS (fondo)
    ax.scatter(coords_removed[:, 0],
               coords_removed[:, 1],
               c='silver',
               alpha=0.35,
               s=10,
               label='Removed')

    # 🔵 SELECCIONADOS
    for cls in [0, 1]:
        mask = (y.values == cls)
        if np.any(mask): 
            ax.scatter(coords_selected[mask, 0],
                       coords_selected[mask, 1],
                       c=cmap[cls],
                       label=labels[cls],
                       alpha=0.8,
                       edgecolors='w',
                       s=50)

    ax.set_xlabel('First t-SNE Dimension', fontsize=21)
    ax.set_ylabel('Second t-SNE Dimension', fontsize=21)

    # cantidad ticks arriba/abajo del 0
    ax.set_ylim(-100, 100)
    ax.set_yticks([100.0, 75.0, 50.0, 25.0, 0, -25.0, -50.0, -75.0, -100.0])

    ax.tick_params(axis='both', which='major', labelsize=17)

    # LEYENDA (sin duplicados)
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='lower left',
              fontsize=10,
              title="Classes",
              title_fontsize=10,
              frameon=True,
              facecolor='white',
              framealpha=1.0)
    
    ax.grid(True, linestyle='--', alpha=0.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    logging.info(f"Figura guardada en: {out_path}")

    plt.show()
    plt.close(fig)


# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    X, y = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    logging.info("Aplicando FEUS para submuestrear el conjunto de entrenamiento...")
    X_subsampled, y_subsampled, removed_indices, selected_indices = apply_feus(
        X_train_scaled,
        y_train,
        n_samples_to_keep=N_SAMPLES_FEUS
    )

    #  Alineación
    X_subsampled = X_subsampled.reset_index(drop=True)
    y_subsampled = y_subsampled.reset_index(drop=True)

    logging.info("Generando t-SNE de los datos submuestreados con FEUS...")
    plot_tsne(
        X_subsampled,
        y_subsampled,
        TSNE_FEUS_PATH,
        X_train_scaled,
        y_train,
        selected_indices,
        removed_indices
    )

    logging.info("Fin del proceso.")




# ------------------------------------ COMBINACIÓN DE FIGURAS MEUS + FEUS -------------------------------------


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_meus = "C:/Users/Diana/Desktop/data_balance/FIGURAS/tSNE_MEUS.png"
img_feus = "C:/Users/Diana/Desktop/data_balance/FIGURAS/tSNE_FEUS.png"

meus = mpimg.imread(img_meus)
feus = mpimg.imread(img_feus)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- MEUS ---
axes[0].imshow(meus)
axes[0].axis('off')

# 🔑 etiqueta (a)
#axes[0].text(
#    0.12, 0.1, "a)",
#    transform=axes[0].transAxes,
#    fontsize=16,
#    fontweight='bold',
#    va='bottom',
#    ha='left'
#)

# --- FEUS ---
axes[1].imshow(feus)
axes[1].axis('off')

# 🔑 etiqueta (b)
#axes[1].text(
#    0.12, 0.1, "b)",
#    transform=axes[1].transAxes,
#    fontsize=16,
#    fontweight='bold',
#    va='bottom',
#    ha='left'
#)

# 🔑 ejes comunes
#fig.supxlabel("First t-SNE Dimension", fontsize=16)
#fig.supylabel("Second t-SNE Dimension", fontsize=16)

plt.tight_layout()

# guardar
output_path = "C:/Users/Diana/Desktop/data_balance/FIGURAS/tSNE_COMBINADO.png"
plt.savefig(output_path, dpi=500, bbox_inches='tight')

plt.show()
