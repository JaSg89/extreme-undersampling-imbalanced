import os
import sys
import time
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import scipy.stats as st

# =========================================================
# Intel Extension for Scikit-learn (intelex / sklearnex)
# =========================================================
USE_INTEL_SKLEARN = True
INTEL_PATCH_ACTIVE = False

os.environ["SKLEARNEX_VERBOSE"] = "INFO"   # verificar fallback/aceleración

if USE_INTEL_SKLEARN:
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        INTEL_PATCH_ACTIVE = True
        print("Intel Extension for Scikit-learn activada correctamente.")
    except Exception as e:
        print(f"No se pudo activar Intel Extension for Scikit-learn: {e}")
        print("Se continuará con scikit-learn original.")

# Imports para Scikit-Learn (Comunes, LR, SVM, y RF)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsClassifier

# Imports para TensorFlow/Keras (NN)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Importación para el modelo XGBoost
import xgboost as xgb

# --- CONFIGURACIÓN GLOBAL ---
# Base 1
CSV_PATH = "C:/Users/Diana/Desktop/Credit_card_data/creditcard.csv"
BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance/MEUS"

# Base 2
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_ready_depurada.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE/MEUS"

# Base 3 = base 2 balanceada
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_balanced.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE_balanced/MEUS"


warnings.filterwarnings("ignore")
N_SIMULATIONS = 30 
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
TARGET_COLUMN_NAME = 'Class'

# Parámetros óptimos para cada modelo
OPTIMAL_NN_PARAMS = {'learning_rate': 0.001, 'dropout_rate': 0.05, 'batch_size': 8, 'epochs': 50}

OPTIMAL_LR_PARAMS = {'penalty': 'l2', 'C': 0.05, 'solver': 'saga', 'max_iter': 2000}
OPTIMAL_SVM_PARAMS = {'C': 0.05, 'kernel': 'rbf', 'gamma': 'scale', 'shrinking': True, 'max_iter':2000} 

OPTIMAL_XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 50, 
                      'learning_rate': 0.001, 'max_depth': 20, 'subsample': 0.05, 'colsample_bytree': 0.4, 
                      'gamma': 1, 'min_child_weight': 0.6, 'reg_alpha': 0.9, 'use_label_encoder': False}

OPTIMAL_RF_PARAMS = {'n_estimators': 1000, 'criterion': 'gini', 'max_depth': 1, 'min_samples_split': 2,
                      'min_samples_leaf': 45, 'max_leaf_nodes': None, 'bootstrap': True, 'max_samples': None, 'oob_score': True, 
                      'ccp_alpha': 0.01, 'warm_start': True, 'verbose': 0}

# --- FUNCIONES DE LOGGING Y CARGA DE DATOS ---
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler); handler.close()
    logging.basicConfig(level=logging.INFO, format=LOG_FMT,
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])

def load_data(csv_path):
    logging.info(f"Cargando datos desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en la ruta: {csv_path}")
        sys.exit("Error: El archivo de datos no fue encontrado. Verifica la variable CSV_PATH.")

# Implementación robusta de MEUS 
def apply_meus_on_train_data(X_train_scaled: pd.DataFrame, y_train: pd.Series, random_seed: int) -> (pd.DataFrame, pd.Series):
    """
    Aplica la técnica de submuestreo MEUS (Matching based on Euclidean-Mahalanobis Under-Sampling)
    con un ajuste robusto para el número de vecinos K.
    """
    logging.info("Aplicando técnica MEUS con matching 1-a-1 real sobre el conjunto de entrenamiento...")

    start_time = time.perf_counter()

    data_with_target = X_train_scaled.copy()
    data_with_target[y_train.name] = y_train

    df_minority = data_with_target[data_with_target[y_train.name] == 1].drop(columns=[y_train.name])
    df_majority = data_with_target[data_with_target[y_train.name] == 0].drop(columns=[y_train.name])

    if df_minority.empty or df_majority.empty:
        logging.warning("No hay suficientes muestras de una o ambas clases para realizar el matching. Devolviendo dataset original.")
        return X_train_scaled, y_train

    logging.info(f"Muestras de clase minoritaria: {len(df_minority)}, Muestras de clase mayoritaria: {len(df_majority)}")
    
    logging.info("Calculando la matriz de covarianza inversa global del conjunto de entrenamiento escalado...")
    try:
        cov_matrix = np.cov(X_train_scaled.values, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    except Exception as e:
        logging.error(f"Error al calcular la matriz de covarianza inversa: {e}. Abortando MEUS.")
        return X_train_scaled, y_train


    MAX_K_NEIGHBORS = 100
    K = min(MAX_K_NEIGHBORS, len(df_majority))
    
    if K < MAX_K_NEIGHBORS:
        logging.warning(f"El número de muestras mayoritarias ({len(df_majority)}) es menor que el K máximo ({MAX_K_NEIGHBORS}). "
                        f"Se ajustará K a {K} para esta simulación.")

    logging.info(f"Configurando NearestNeighbors con métrica Mahalanobis para buscar los {K} candidatos más cercanos...")

    nn = NearestNeighbors(
        n_neighbors=K,  
        metric='mahalanobis', 
        metric_params={'VI': inv_cov_matrix},
        algorithm='auto'
    )


    nn.fit(df_majority.values)
    
    logging.info(f"Buscando los {K} vecinos más cercanos en la clase mayoritaria para cada muestra de la clase minoritaria...")
    distances, neighbor_indices = nn.kneighbors(df_minority.values)

    logging.info("Realizando el matching 1-a-1 para asegurar unicidad y balance...")
    claimed_majority_indices = set()
    final_selected_indices = []

    for i in range(len(df_minority)):
        found_pair = False
        # Iteramos sobre los K candidatos para este punto minoritario
        for j in range(K): # Usamos K dinámico
            candidate_positional_idx = neighbor_indices[i, j]
            candidate_label = df_majority.index[candidate_positional_idx]
            
            if candidate_label not in claimed_majority_indices:
                claimed_majority_indices.add(candidate_label)
                final_selected_indices.append(candidate_label)
                found_pair = True
                break
        
        if not found_pair:
            logging.warning(f"No se pudo encontrar un par único para la muestra minoritaria en la posición {i} "
                            f"dentro de los {K} candidatos. El conjunto final podría estar desbalanceado.")

    logging.info(f"Se seleccionaron {len(final_selected_indices)} muestras únicas de la clase mayoritaria.")
    
    df_majority_selected = df_majority.loc[final_selected_indices]
    X_res = pd.concat([df_minority, df_majority_selected], axis=0)
    y_res = pd.Series([1] * len(df_minority) + [0] * len(df_majority_selected), name=y_train.name, index=X_res.index)
    
    np.random.seed(random_seed)
    perm = np.random.permutation(len(X_res))
    
    X_res = X_res.iloc[perm]
    y_res = y_res.iloc[perm]
    
    logging.info(f"MEUS: Tamaño del conjunto de entrenamiento original: {len(X_train_scaled)}, nuevo tamaño: {len(X_res)}")
    logging.info(f"Distribución de clases después de MEUS: {dict(y_res.value_counts())}")

    resampling_time = time.perf_counter() - start_time
    logging.info(f"Tiempo total para aplicar MEUS: {resampling_time:.4f} segundos.")
    
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True), resampling_time


# =========================================================
# PREPARACIÓN DE DATOS — VERSIÓN ANTERIOR, NO UTILIZADA
# =========================================================
# NOTA:
# Esta función se conserva únicamente como referencia histórica.
# Ya no se utiliza en el experimento actual, porque correspondía a un
# esquema hold-out con train_test_split.
#
# El diseño experimental actual usa repeated stratified 5-fold cross-validation.
# En consecuencia, la partición train/validation, el escalamiento y el remuestreo
# se realizan dentro de cada fold mediante la función run_cv(), evitando leakage.
# =========================================================
#def prepare_data(df_original, target_col_name, random_seed):
#    logging.info(">> Iniciando preparación de datos (Split -> Scale -> Balance)")
#    X_original, y_original = df_original.drop(target_col_name, axis=1), df_original[target_col_name]
#    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, stratify=y_original, random_state=random_seed)
#    scaler = MinMaxScaler()
#    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
#    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
#    X_train_final, y_train_final, resampling_time = apply_meus_on_train_data(X_train_scaled, y_train, random_seed)
#    return X_train_final, X_test_scaled, y_train_final, y_test, scaler, resampling_time

# --- FUNCIONES ESPECÍFICAS DE CADA MODELO ---

def train_nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Red Neuronal (NN)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo NN."); return None


    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.4, stratify=y_train, random_state=random_seed)
    
    model = Sequential([Dense(16, input_shape=(X_train_new.shape[1],), activation='softplus'),
                        Dropout(OPTIMAL_NN_PARAMS['dropout_rate']),
                        Dense(16, activation='softplus'),
                        Dropout(OPTIMAL_NN_PARAMS['dropout_rate']),   
                        Dense(8, activation='softplus'),     
                        Dense(2, activation='softmax')])
    
    model.compile(optimizer=Adam(learning_rate=OPTIMAL_NN_PARAMS['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)]
    
    start_time = time.perf_counter()
    model.fit(X_train_new, y_train_new, epochs=OPTIMAL_NN_PARAMS['epochs'], batch_size=OPTIMAL_NN_PARAMS['batch_size'],
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento NN completado en {train_time:.4f} segundos.")
    #model_path = os.path.join(output_dir, f"{exp_name}_nn_model.h5")
    #model.save(model_path)
    #logging.info(f"Modelo NN (Keras) guardado en: {model_path}")
    return model, train_time

def evaluate_nn_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo NN...")
    #model = load_model(model_path, compile=False)
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # Probabilidad de la clase 1 (Fraude)
    y_prob_positive = y_prob[:, 1]

    # Calcular Métricas de Curvas
    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación NN:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"NN -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return recall, precision, roc_auc, pr_auc, f1, accuracy

def train_lr_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Regresión Logística (LR)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo LR."); return None
    model = LogisticRegression(**OPTIMAL_LR_PARAMS, random_state=random_seed)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento LR completado en {train_time:.4f} segundos.")
    #model_path = os.path.join(output_dir, f"{exp_name}_lr_model.joblib")
    #joblib.dump(model, model_path)
    #logging.info(f"Modelo LR (joblib) guardado en: {model_path}")
    return model, train_time

def evaluate_lr_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo LR...")
    #model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Probabilidades para la clase positiva
    y_prob_positive = model.predict_proba(X_test)[:, 1]

    # Calcular ROC-AUC y PR-AUC
    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación LR:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"LR -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return recall, precision, roc_auc, pr_auc, f1, accuracy

def train_svm_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo Support Vector Machine (SVM)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo SVM."); return None
    model = SVC(**OPTIMAL_SVM_PARAMS, random_state=random_seed, probability=True)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento SVM completado en {train_time:.4f} segundos.")
    #model_path = os.path.join(output_dir, f"{exp_name}_svm_model.joblib")
    #joblib.dump(model, model_path)
    #logging.info(f"Modelo SVM (joblib) guardado en: {model_path}")
    return model, train_time

def evaluate_svm_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo SVM...")
    #model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Probabilidades para la clase positiva
    y_prob_positive = model.predict_proba(X_test)[:, 1]

    # Calcular ROC-AUC y PR-AUC
    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación SVM:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"SVM -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return recall, precision, roc_auc, pr_auc, f1, accuracy

def train_xgb_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo XGBoost...")
    if X_train.empty or y_train.empty:
        logging.error("X_train o y_train están vacíos. No se puede entrenar el modelo XGBoost.")
        return None
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.5, stratify=y_train, random_state=random_seed
    )
    current_xgb_params = OPTIMAL_XGB_PARAMS.copy()
    current_xgb_params['seed'] = random_seed
    model = xgb.XGBClassifier(**current_xgb_params, early_stopping_rounds=50)
    start_time = time.perf_counter()
    model.fit(
        X_train_new, y_train_new,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento XGBoost completado en {train_time:.4f} segundos.")
    logging.info(f"Mejor iteración de XGBoost (Early Stopping): {model.best_iteration}")
    #model_path = os.path.join(output_dir, f"{exp_name}_xgb_model.joblib")
    #joblib.dump(model, model_path)
    #logging.info(f"Modelo XGBoost (joblib) guardado en: {model_path}")
    return model, train_time

def evaluate_xgb_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo XGBoost...")
    #model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Probabilidades para la clase positiva
    y_prob_positive = model.predict_proba(X_test)[:, 1]

    # Calcular ROC-AUC y PR-AUC
    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación XGBoost:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"XGBoost -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")   
    return recall, precision, roc_auc, pr_auc, f1, accuracy

def train_rf_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo Random Forest (RF)...")
    if X_train.empty:
        logging.error("X_train está vacío. No se puede entrenar el modelo RF.")
        return None
    
    current_rf_params = OPTIMAL_RF_PARAMS.copy()
    current_rf_params['random_state'] = random_seed
    
    model = RandomForestClassifier(**current_rf_params)
    
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento RF completado en {train_time:.4f} segundos.")
    
    #model_path = os.path.join(output_dir, f"{exp_name}_rf_model.joblib")
    #joblib.dump(model, model_path)
    #logging.info(f" Modelo RF (joblib) guardado en: {model_path}")
    return model, train_time

def evaluate_rf_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo RF...")
    #model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    # Probabilidades para la clase positiva
    y_prob_positive = model.predict_proba(X_test)[:, 1]

    # Calcular ROC-AUC y PR-AUC
    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación RF:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"RF -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")    
    return recall, precision, roc_auc, pr_auc, f1, accuracy


def train_1nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo 1NN...")
    if X_train.empty: return None, 0.0

    model = KNeighborsClassifier(n_neighbors=1)
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento 1NN completado en {train_time:.4f} segundos.")
    #model_path = os.path.join(output_dir, f"{exp_name}_1nn_model.joblib")
    #joblib.dump(model, model_path)
    return model, train_time

def evaluate_1nn_model(model, X_test, y_test):
    logging.info(">>> Iniciando evaluación del modelo 1NN...")
    #model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob_positive = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob_positive)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob_positive)
    pr_auc = auc(recall_pts, precision_pts)

    report = classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"], digits=4, zero_division=0, output_dict=True)
    logging.info(f"Reporte de Clasificación 1NN:\n{classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude'], digits=4, zero_division=0)}")
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"1NN -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return recall, precision, roc_auc, pr_auc, f1, accuracy


# =========================================================
# VALIDACIÓN CRUZADA CON MEUS
# =========================================================
# Ejecuta Stratified 5-fold cross-validation sin leakage.
# En cada fold, el escalamiento se ajusta solo con el training fold,
# MEUS se aplica exclusivamente sobre ese training fold, y los modelos
# se evalúan sobre el validation fold original.
#
# Las métricas se promedian sobre los 5 folds. 
# =========================================================
def run_cv(df, target_col, model_runners, random_seed, output_dir, time_records):

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    metrics_sum = {}


    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        logging.info(f"\n===== Fold {fold+1}/5 =====")

        # SPLIT
        X_train_raw = X.iloc[train_idx]
        X_val_raw   = X.iloc[val_idx]
        y_train     = y.iloc[train_idx]
        y_val       = y.iloc[val_idx]

        # SCALING (sin leakage)
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_raw),
            columns=X_train_raw.columns,
            index=X_train_raw.index
        )

        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_raw),
            columns=X_val_raw.columns,
            index=X_val_raw.index
        )

        # CLAVE: MEUS SOLO EN TRAIN
        X_res, y_res, r_time = apply_meus_on_train_data(
            X_train_scaled, y_train, random_seed
        )


        # MODELOS 
        for model_name, (train_func, eval_func) in model_runners.items():

            model, train_time = train_func(
                X_res.copy(), y_res.copy(),
                output_dir, f"cv_{model_name}_fold{fold}", random_seed
            )

            # Guardar tiempo crudo por fold y modelo.
            # Unidad base:
            # # tiempo total por fold = tiempo de MEUS + tiempo de entrenamiento del modelo.
            time_records.append({
                'tecnica': 'MEUS',
                'simulation': random_seed,
                'fold': fold + 1,
                'model': model_name,
                'tiempo_resampling_segundos': r_time,
                'tiempo_training_segundos': train_time,
                'tiempo_total_segundos': r_time + train_time
                })

            if model is not None:
                rec, prec, roc, pr, f1, acc = eval_func(
                    model, X_val_scaled, y_val
                )

                metrics_sum[f'recall_{model_name}'] = metrics_sum.get(f'recall_{model_name}', 0) + rec
                metrics_sum[f'precision_{model_name}'] = metrics_sum.get(f'precision_{model_name}', 0) + prec
                metrics_sum[f'roc_auc_{model_name}'] = metrics_sum.get(f'roc_auc_{model_name}', 0) + roc
                metrics_sum[f'pr_auc_{model_name}'] = metrics_sum.get(f'pr_auc_{model_name}', 0) + pr
                metrics_sum[f'f1_{model_name}'] = metrics_sum.get(f'f1_{model_name}', 0) + f1
                metrics_sum[f'accuracy_{model_name}'] = metrics_sum.get(f'accuracy_{model_name}', 0) + acc

    # PROMEDIO FINAL 
    results = {}

    for k, v in metrics_sum.items():
        results[k] = v / 5.0

    return results


# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
global_start_time = time.perf_counter()

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    all_results = []
    time_records = []
    df_original = load_data(CSV_PATH)

    for i in range(1, N_SIMULATIONS + 1):
        sim_start_time = time.perf_counter()
        exp_name_sim = f"MEUS_Compare_Sim_{i}"
        output_dir_sim = os.path.join(BASE_OUTPUT, exp_name_sim)
        log_file_sim = os.path.join(output_dir_sim, f"run_{exp_name_sim}.log")
        setup_logging(log_file_sim)

        logging.info(f"Intel Extension for Scikit-learn activa: {INTEL_PATCH_ACTIVE}")
        
        print(f"\n{'='*25} INICIANDO SIMULACIÓN {i}/{N_SIMULATIONS} {'='*25}")
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando Simulación Comparativa: {i}/{N_SIMULATIONS} ===")
        
        current_seed = i 
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        logging.info(f"Semillas (Numpy, TF, Split, Sklearn, XGB, RF) establecidas en: {current_seed}")
        
        # =================================================
        # Línea del flujo anterior con train_test_split.
        # =================================================
        # X_train, X_test, y_train, y_test, _, resampling_time = prepare_data(df_original, TARGET_COLUMN_NAME, random_seed=current_seed)
        
        # Procesamiento de todos los modelos
        model_runners = {
            "NN": (train_nn_model, evaluate_nn_model),
            "LR": (train_lr_model, evaluate_lr_model),
            "SVM": (train_svm_model, evaluate_svm_model),
            "XGB": (train_xgb_model, evaluate_xgb_model),
            "RF": (train_rf_model, evaluate_rf_model),
            "1NN": (train_1nn_model, evaluate_1nn_model)
        }
        

        cv_results = run_cv(df_original, TARGET_COLUMN_NAME, model_runners, current_seed, output_dir_sim, time_records)

        sim_results = {'Simulacion': i}

        for model_name in model_runners.keys():
            logging.info(f"\n--- Procesando Modelo: {model_name} con datos de MEUS ---")
            # Metricas de clasificacion para FEUS
            sim_results[f'recall_{model_name}_MEUS'] = cv_results.get(f'recall_{model_name}')
            sim_results[f'roc_auc_{model_name}_MEUS'] = cv_results.get(f'roc_auc_{model_name}')
            sim_results[f'pr_auc_{model_name}_MEUS'] = cv_results.get(f'pr_auc_{model_name}')
            sim_results[f'precision_{model_name}_MEUS'] = cv_results.get(f'precision_{model_name}')
            sim_results[f'f1_{model_name}_MEUS'] = cv_results.get(f'f1_{model_name}')
            sim_results[f'accuracy_{model_name}_MEUS'] = cv_results.get(f'accuracy_{model_name}')


        sim_total_time = time.perf_counter() - sim_start_time
        sim_results['total_simulation_time'] = sim_total_time

        logging.info(f"Tiempo total simulación {i}: {sim_total_time:.4f} segundos")

        all_results.append(sim_results)
        
        print(f"--- Simulación {i} completada. ---")
        logging.info(f"=== Fin Simulación {i} ===\n")

    print(f"\n{'='*25} TODAS LAS SIMULACIONES COMPLETADAS {'='*25}")

    global_total_time = time.perf_counter() - global_start_time
    print(f"\n Tiempo TOTAL de ejecución del script: {global_total_time:.4f} segundos")


    # --- REPORTE FINAL COMPLETO ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        #print("\nResumen completo de métricas:")
        #print(results_df.to_string(index=False))

        def compute_ci(series, confidence=0.95):
            mean = series.mean()
            sem = st.sem(series)
            h = sem * st.t.ppf((1 + confidence) / 2., len(series)-1)
            return mean, mean - h, mean + h

        # Separación de columnas
        metric_keywords = ['precision', 'recall', 'roc_auc', 'pr_auc', 'f1', 'accuracy']
        time_keywords = ['time']

        stats_summary = "Estadísticas de Robustez (MEUS + 5-Fold CV)\n"
        stats_summary += "="*60 + "\n"
        stats_summary += f"Total de Simulaciones: {len(results_df)}\n\n"

        # MÉTRICAS DE CLASIFICACIÓN
        stats_summary += "### MÉTRICAS DE CLASIFICACIÓN ###\n\n"
        for col in results_df.columns:
            if any(k in col for k in metric_keywords):
                mean, low, high = compute_ci(results_df[col])
                stats_summary += f"--- {col} ---\n"
                stats_summary += f"   - Media:           {mean:.4f}\n"
                stats_summary += f"   - IC 95%:          [{low:.4f}, {high:.4f}]\n"
                stats_summary += f"   - Desv. Estándar: {results_df[col].std():.4f}\n"
                stats_summary += f"   - Mínimo:         {results_df[col].min():.4f}\n"
                stats_summary += f"   - Máximo:         {results_df[col].max():.4f}\n\n"

        # TIEMPOS
        stats_summary += "\n### TIEMPOS ###\n"
        stats_summary += "Tiempo considerado por fold: tiempo de MEUS + tiempo de entrenamiento del modelo.\n"
        stats_summary += "Calculo: primero se promedia el tiempo total de los 5 folds dentro de cada simulacion; "
        stats_summary += "luego se calcula la media y desviacion estandar entre simulaciones.\n"
        stats_summary += "Unidad: segundos por fold.\n\n"
        
        if time_records:
            time_df = pd.DataFrame(time_records)

            tiempos_por_simulacion = (
                time_df
                .groupby(["tecnica", "model", "simulation"], as_index=False)
                .agg(
                    n_folds=("fold", "nunique"),
                    media_fold_simulacion_segundos=("tiempo_total_segundos", "mean")
                )
            )

            tiempos_por_simulacion = (
                tiempos_por_simulacion
                .loc[tiempos_por_simulacion["n_folds"] == 5]
                .copy()
            )
            
            resumen_tiempos = (
                tiempos_por_simulacion
                .groupby(["tecnica", "model"], as_index=False)
                .agg(
                    n_simulaciones=("simulation", "nunique"),
                    media_segundos=("media_fold_simulacion_segundos", "mean"),
                    desv_segundos=("media_fold_simulacion_segundos", "std"),
                    min_segundos=("media_fold_simulacion_segundos", "min"),
                    max_segundos=("media_fold_simulacion_segundos", "max")
                )
            )

            orden_modelos = ["LR", "SVM", "NN", "RF", "XGB", "1NN"]
            resumen_tiempos["model"] = pd.Categorical(
                resumen_tiempos["model"],
                categories=orden_modelos,
                ordered=True
            )

            resumen_tiempos = resumen_tiempos.sort_values("model")
            
            for _, row in resumen_tiempos.iterrows():
                stats_summary += f"--- total_time_{row['model']}_MEUS ---\n"
                stats_summary += f"   - Media:           {row['media_segundos']:.4f} \n"
                stats_summary += f"   - Desv. Estandar: {row['desv_segundos']:.4f} \n"
                stats_summary += f"   - Minimo:         {row['min_segundos']:.4f} \n"
                stats_summary += f"   - Maximo:         {row['max_segundos']:.4f} \n\n"


        #print("\n" + stats_summary)

        # Guardar CSV completo
        summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_MEUS_summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Resultados guardados en: {summary_csv_path}")

        # Guardar resumen estadístico
        stats_summary_path = os.path.join(BASE_OUTPUT, 'Statistics_summary_MEUS.txt')
        with open(stats_summary_path, 'w') as f:
            f.write(stats_summary)
        print(f"Estadisticas guardadas en: {stats_summary_path}")