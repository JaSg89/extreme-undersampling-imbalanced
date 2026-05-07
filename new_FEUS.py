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
BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance/FEUS"


# Base 2
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_ready_depurada.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE/FEUS"

# Base 3 = base 2 balanceada
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_balanced.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE_balanced/FEUS"


warnings.filterwarnings("ignore")

N_SIMULATIONS = 30  # Número de simulaciones a realizar

# --- Configuración de logging --- 
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
TARGET_COLUMN_NAME = 'Class'

# --- PARÁMETROS DE FEUS ---
N_SAMPLES_FEUS = 5000 


# Parámetros óptimos para cada modelo (mantenidos del script original para consistencia)
OPTIMAL_NN_PARAMS = {'learning_rate': 0.001, 'dropout_rate': 0.05, 'batch_size': 16, 'epochs': 150}
OPTIMAL_LR_PARAMS = {'penalty': 'l2', 'C': 50, 'solver': 'liblinear', 'max_iter': 2000, 'fit_intercept': True, 'intercept_scaling': 1.0}
OPTIMAL_SVM_PARAMS = {'C': 150, 'kernel': 'rbf', 'gamma': 'scale', 'shrinking': True, 'max_iter':2000} 

OPTIMAL_XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 250,
                       'learning_rate': 0.1, 'max_depth': 1, 'subsample': 0.9, 'colsample_bytree': 0.1, 'gamma': 1, 
                       'min_child_weight': 0.6, 'reg_alpha': 0.9, 'use_label_encoder': False}

OPTIMAL_RF_PARAMS = {'n_estimators': 1000, 'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2,
                      'min_samples_leaf': 10, 'max_leaf_nodes': None, 'bootstrap': True, 'max_samples': None, 
                      'oob_score': True, 'ccp_alpha': 0.001, 'warm_start': False, 'verbose': 0}


# =========================================================
#          FUNCIONES DE LOGGING Y CARGA DE DATOS
# =========================================================

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


# =========================================================
# <<< INICIO DE LA MODIFICACIÓN: Implementación de FEUS >>>
# =========================================================

def feus_apply_logic(data_features_input: pd.DataFrame, n_samples_to_keep: int, operation_context: str) -> pd.Index:
    """
    Lógica central de FEUS: calcula la distancia de Mahalanobis y selecciona las muestras más lejanas.
    """
    if data_features_input.empty:
        logging.warning(f"FEUS {operation_context}: DataFrame de entrada vacío.")
        return pd.Index([])
    
    # Se usa un scaler interno para el cálculo de Mahalanobis, sin afectar los datos originales.
    scaler_internal = MinMaxScaler()
    X_scaled_for_mahalanobis = pd.DataFrame(scaler_internal.fit_transform(data_features_input),
                                            columns=data_features_input.columns,
                                            index=data_features_input.index)
    
    if X_scaled_for_mahalanobis.shape[0] < 2:
        logging.warning(f"FEUS {operation_context}: No hay suficientes muestras para calcular la covarianza.")
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    mean_vector = X_scaled_for_mahalanobis.mean().values
    try:
        inv_cov_matrix = np.linalg.pinv(np.cov(X_scaled_for_mahalanobis.values, rowvar=False))
    except Exception as e:
        logging.error(f"Error al calcular la matriz de covarianza inversa en FEUS: {e}")
        # Fallback: devolver las primeras N muestras si el cálculo falla
        return data_features_input.head(min(n_samples_to_keep, len(data_features_input))).index

    diff_values = X_scaled_for_mahalanobis.values - mean_vector
    # Usar np.einsum para un cálculo eficiente del producto punto diagonal
    distances_sq = np.einsum('ij,ij->i', diff_values.dot(inv_cov_matrix), diff_values)
    distances = np.sqrt(np.maximum(distances_sq, 0)) # Asegurar que no haya valores negativos
    
    data_features_with_dist = data_features_input.copy()
    data_features_with_dist['Mahalanobis_Distance_Internal'] = distances
    
    selected_indices = data_features_with_dist.nlargest(n_samples_to_keep, 'Mahalanobis_Distance_Internal').index
    return selected_indices

def apply_feus_on_train_data(X_train_scaled: pd.DataFrame, y_train: pd.Series, n_samples_to_keep: int) -> (pd.DataFrame, pd.Series):
    """
    Aplica la técnica de submuestreo FEUS (Furthest Euclidean Under-Sampling)
    al conjunto de entrenamiento completo.
    """
    logging.info(f"Aplicando técnica FEUS para seleccionar las {n_samples_to_keep} muestras más lejanas...")

    start_time = time.perf_counter()
    
    # Asegurarse de que no intentamos seleccionar más muestras de las que hay
    if n_samples_to_keep > len(X_train_scaled):
        logging.warning(f"N_SAMPLES_FEUS ({n_samples_to_keep}) es mayor que el tamaño del set de entrenamiento ({len(X_train_scaled)}). "
                        "Se devolverá el set de entrenamiento completo.")
        return X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)

    selected_indices = feus_apply_logic(X_train_scaled, n_samples_to_keep, "TrainSet")
    
    X_res_scaled = X_train_scaled.loc[selected_indices].reset_index(drop=True)
    y_res = y_train.loc[selected_indices].reset_index(drop=True)

    resampling_time = time.perf_counter() - start_time
    
    logging.info(f"Tiempo FEUS: {resampling_time:.4f} segundos.")
    logging.info(f"FEUS: Tamaño del conjunto de entrenamiento original: {len(X_train_scaled)}, nuevo tamaño: {len(X_res_scaled)}")
    logging.info(f"Distribución de clases después de FEUS: {dict(y_res.value_counts())}")

    return X_res_scaled, y_res, resampling_time


# =========================================================
#   PREPARACIÓN DE DATOS — VERSIÓN ANTERIOR, NO UTILIZADA
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
#
#def prepare_data(df_original, target_col_name, random_seed):
#    logging.info(">> Iniciando preparación de datos (Split -> Scale -> Balance con FEUS)")
#    X_original, y_original = df_original.drop(target_col_name, axis=1), df_original[target_col_name]
#    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, stratify=y_original, random_state=random_seed)
#    scaler = MinMaxScaler()
#    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
#    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
#    
#    # CAMBIO: Se llama a la función de FEUS en lugar de MEUS
#    X_train_final, y_train_final, resampling_time = apply_feus_on_train_data(X_train_scaled, y_train, N_SAMPLES_FEUS)
#    
#    return X_train_final, X_test_scaled, y_train_final, y_test, scaler, resampling_time



# =========================================================
#           FUNCIONES ESPECÍFICAS DE CADA MODELO
# =========================================================

def train_nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Red Neuronal (NN)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo NN."); return None
    
    # Usar una división para validación si hay suficientes datos y clases
    stratify_val = y_train if len(y_train.unique()) > 1 else None
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=stratify_val, random_state=random_seed)

    model = Sequential([Dense(32, input_shape=(X_train_new.shape[1],), activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']),
                        Dense(16, activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']), Dense(2, activation='softmax')])
    
    model.compile(optimizer=Adam(learning_rate=OPTIMAL_NN_PARAMS['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=0)]
    
    start_time = time.perf_counter()
    model.fit(X_train_new, y_train_new, epochs=OPTIMAL_NN_PARAMS['epochs'], batch_size=OPTIMAL_NN_PARAMS['batch_size'],
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
    
    train_time = time.perf_counter() - start_time
    logging.info(f"Entrenamiento NN completado en {train_time:.4f} segundos.")
    #model_path = os.path.join(output_dir, f"{exp_name}_nn_model.h5")
    #model.save(model_path)
    #logging.info(f" Modelo NN (Keras) guardado en: {model_path}")
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
    # Probabilidad de la clase 1 (Fraude)
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
        
    stratify_val = y_train if len(y_train.unique()) > 1 else None
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=stratify_val, random_state=random_seed
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
    logging.info(f"XGB -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
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
    #logging.info(f"Modelo RF (joblib) guardado en: {model_path}")
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
    #logging.info(f"Modelo 1NN (joblib) guardado en: {model_path}")

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
    recall = report.get('macro avg', {}).get('recall', 0.0)
    precision = report.get('macro avg', {}).get('precision', 0.0)
    f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    accuracy = report.get('accuracy', 0.0)
    logging.info(f"1NN -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return recall, precision, roc_auc, pr_auc, f1, accuracy


# =========================================================
#               VALIDACIÓN CRUZADA CON FEUS
# =========================================================
# Ejecuta Stratified 5-fold cross-validation sin leakage.
# En cada fold, el escalamiento se ajusta solo con el training fold,
# FEUS se aplica exclusivamente sobre ese training fold, y los modelos
# se evalúan sobre el validation fold original.
#
# Las métricas se promedian sobre los 5 folds. 
# =========================================================
def run_cv(df, target_col, model_runners, random_seed, output_dir, feus_distribution_records, time_records):

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    metrics_sum = {}
    time_sum = {}
    total_resampling_time = 0.0


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

        # FEUS SOLO EN TRAIN
        X_res, y_res, r_time = apply_feus_on_train_data(
            X_train_scaled, y_train, N_SAMPLES_FEUS
        )

        total_resampling_time += r_time
        counts = y_res.value_counts().to_dict()

        n_class_0 = int(counts.get(0, 0))
        n_class_1 = int(counts.get(1, 0))
        n_total = len(y_res)

        feus_distribution_records.append({
            'feus_total': n_total,
            'feus_n_class_0': n_class_0,
            'feus_n_class_1': n_class_1,
            'feus_prop_class_0': n_class_0 / n_total if n_total > 0 else np.nan,
            'feus_prop_class_1': n_class_1 / n_total if n_total > 0 else np.nan
            })

        # MODELOS
        for model_name, (train_func, eval_func) in model_runners.items():

            model, train_time = train_func(
                X_res.copy(), y_res.copy(),
                output_dir, f"cv_{model_name}_fold{fold}", random_seed
            )

            time_sum[model_name] = time_sum.get(model_name, 0.0) + train_time

            # Guardar tiempo crudo por fold y modelo.
            # Esta es la unidad base para el resumen final de tiempos:
            # tiempo total por fold = tiempo de FEUS + tiempo de entrenamiento del modelo.
            time_records.append({
                'tecnica': 'FEUS',
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


# =========================================================
#            BLOQUE PRINCIPAL DE EJECUCIÓN
# =========================================================

global_start_time = time.perf_counter()

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    all_results = []
    time_records = []
    feus_distribution_records = []
    df_original = load_data(CSV_PATH)

    for i in range(1, N_SIMULATIONS + 1):
        
        sim_start_time = time.perf_counter()

        exp_name_sim = f"FEUS_Compare_Sim_{i}"
        output_dir_sim = os.path.join(BASE_OUTPUT, exp_name_sim)
        log_file_sim = os.path.join(output_dir_sim, f"run_{exp_name_sim}.log")
        setup_logging(log_file_sim)

        logging.info(f"Intel Extension for Scikit-learn activa: {INTEL_PATCH_ACTIVE}")
        
        print(f"\n{'='*25} INICIANDO SIMULACIÓN CON FEUS {i}/{N_SIMULATIONS} {'='*25}")
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando Simulación Comparativa (FEUS): {i}/{N_SIMULATIONS} ===")
        
        current_seed = i 
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        logging.info(f"Semillas (Numpy, TF, Split, Sklearn, XGB, RF) establecidas en: {current_seed}")
                
                
        # =================================================
        # Línea del flujo anterior con train_test_split.
        # =================================================
        #X_train, X_test, y_train, y_test, _, resampling_time = prepare_data(df_original, TARGET_COLUMN_NAME, random_seed=current_seed)


        model_runners = {
            "NN": (train_nn_model, evaluate_nn_model),
            "LR": (train_lr_model, evaluate_lr_model),
            "SVM": (train_svm_model, evaluate_svm_model),
            "XGB": (train_xgb_model, evaluate_xgb_model),
            "RF": (train_rf_model, evaluate_rf_model),
            "1NN": (train_1nn_model, evaluate_1nn_model)
        }
        
        cv_results = run_cv(df_original, TARGET_COLUMN_NAME, model_runners, current_seed, output_dir_sim, feus_distribution_records, time_records)

 
        sim_results = {'Simulacion': i}

        for model_name in model_runners.keys():
            logging.info(f"\n--- Procesando Modelo: {model_name} con datos de FEUS ---")
            # Metricas de clasificacion para FEUS
            sim_results[f'recall_{model_name}_FEUS'] = cv_results.get(f'recall_{model_name}')
            sim_results[f'roc_auc_{model_name}_FEUS'] = cv_results.get(f'roc_auc_{model_name}')
            sim_results[f'pr_auc_{model_name}_FEUS'] = cv_results.get(f'pr_auc_{model_name}')
            sim_results[f'precision_{model_name}_FEUS'] = cv_results.get(f'precision_{model_name}')
            sim_results[f'f1_{model_name}_FEUS'] = cv_results.get(f'f1_{model_name}')
            sim_results[f'accuracy_{model_name}_FEUS'] = cv_results.get(f'accuracy_{model_name}')

        sim_total_time = time.perf_counter() - sim_start_time
        sim_results['total_simulation_time'] = sim_total_time

        logging.info(f"Tiempo total simulación {i}: {sim_total_time:.4f} segundos")

        all_results.append(sim_results)
        
        
        print(f"--- Simulación {i} con FEUS completada. ---")
        logging.info(f"=== Fin Simulación {i} ===\n")

    print(f"\n{'='*25} TODAS LAS SIMULACIONES CON FEUS COMPLETADAS {'='*25}")
    
    global_total_time = time.perf_counter() - global_start_time
    print(f"\n Tiempo TOTAL de ejecución del script: {global_total_time:.4f} segundos")
 

    # =========================================================
    #                 REPORTE FINAL COMPLETO
    # =========================================================
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

        stats_summary = "Estadísticas de Robustez (FEUS + 5-Fold CV)\n"
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
        # Unidad base: por cada simulación, fold y modelo.
        # Tiempo total por fold = tiempo FEUS del fold + tiempo de entrenamiento del modelo en ese fold.
        # Resumen final: primero se promedia entre folds dentro de cada simulación;
        # luego se calcula media, desviación estándar, mínimo y máximo entre simulaciones.
        stats_summary += "\n### TIEMPOS ###\n"
        stats_summary += "Tiempo considerado por fold: FEUS + entrenamiento del modelo.\n"
        stats_summary += "Primero se promedia entre folds dentro de cada simulación; luego se resume entre simulaciones.\n"
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

            resumen_tiempos = (
                tiempos_por_simulacion
                .groupby(["tecnica", "model"], as_index=False)
                .agg(
                    n_simulaciones=("simulation", "nunique"),
                    media_segundos=("media_fold_simulacion_segundos", "mean"),
                    desv_estandar_segundos=("media_fold_simulacion_segundos", "std"),
                    minimo_segundos=("media_fold_simulacion_segundos", "min"),
                    maximo_segundos=("media_fold_simulacion_segundos", "max")
                )
            )

            model_order = {model_name: idx for idx, model_name in enumerate(model_runners.keys())}
            resumen_tiempos["orden_modelo"] = resumen_tiempos["model"].map(model_order)
            resumen_tiempos = resumen_tiempos.sort_values(["tecnica", "orden_modelo"]).drop(columns=["orden_modelo"])

            for _, row in resumen_tiempos.iterrows():
                stats_summary += f"--- {row['tecnica']} - {row['model']} ---\n"
                stats_summary += f"   - Media:           {row['media_segundos']:.4f} \n"
                stats_summary += f"   - Desv. Estándar: {row['desv_estandar_segundos']:.4f} \n"
                stats_summary += f"   - Mínimo:         {row['minimo_segundos']:.4f} \n"
                stats_summary += f"   - Máximo:         {row['maximo_segundos']:.4f} \n\n"
        else:
            stats_summary += "No se registraron tiempos por fold y modelo.\n\n"


        # DISTRIBUCIÓN FINAL DE CLASES POST-FEUS
        stats_summary += "\n### DISTRIBUCIÓN FINAL DE CLASES POST-FEUS ###\n"
        stats_summary += "Media y desviación estándar calculadas directamente sobre los folds remuestreados.\n\n"

        if feus_distribution_records:
            feus_dist_df = pd.DataFrame(feus_distribution_records)

            feus_distribution_vars = [
                ("feus_total", "Total post-FEUS"),
                ("feus_n_class_0", "Clase 0 post-FEUS"),
                ("feus_n_class_1", "Clase 1 post-FEUS"),
                ("feus_prop_class_0", "Proporción clase 0 post-FEUS"),
                ("feus_prop_class_1", "Proporción clase 1 post-FEUS"),
            ]

            for col, label in feus_distribution_vars:
                mean_value = feus_dist_df[col].mean()
                std_value = feus_dist_df[col].std(ddof=1)

                stats_summary += f"{label}: {mean_value:.4f} ± {std_value:.4f}\n"

                stats_summary += "\n"
        else:
            stats_summary += "No se registró la distribución de clases post-FEUS.\n\n"

        #print("\n" + stats_summary)

        # Guardar CSV completo
        summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_FEUS_summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Resultados guardados en: {summary_csv_path}")

        # Guardar resumen estadístico
        stats_summary_path = os.path.join(BASE_OUTPUT, 'Statistics_summary_FEUS.txt')
        with open(stats_summary_path, 'w') as f:
            f.write(stats_summary)
        print(f"Estadísticas guardadas en: {stats_summary_path}")