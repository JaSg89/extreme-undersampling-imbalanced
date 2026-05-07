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

# Import para Imbalanced-learn (NearMiss)
from imblearn.under_sampling import NearMiss

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
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance/NearMiss" 
BASE_OUTPUT = "C:/Users/Diana/Desktop/CODIGOS_FINALES/ECCD/NearMiss" 


# Base 2
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_ready_depurada.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE/NearMiss"

# Base 3 = base 2 balanceada
#CSV_PATH = "C:/Users/Diana/Desktop/ieee-fraud-detection/ieee_balanced.csv"
#BASE_OUTPUT = "C:/Users/Diana/Desktop/data_balance_IEEE_balanced/NearMiss"

warnings.filterwarnings("ignore")
N_SIMULATIONS = 30
LOG_FMT = "%(asctime)s %(levelname)-8s %(message)s"
TARGET_COLUMN_NAME = 'Class'


# =========================================================
#                       NEAR MISS
# =========================================================

NEAR_MISS_PARAMS = {
    'version': 1,
    'n_neighbors': 3,
    'sampling_strategy': 'majority'
}

# Parámetros óptimos para cada modelo (los mismos que en el script de MEUS)
OPTIMAL_NN_PARAMS = {'learning_rate': 0.0001, 'dropout_rate': 0.5, 'batch_size': 16, 'epochs': 150}
OPTIMAL_LR_PARAMS = {'penalty': 'l2', 'C': 0.05, 'solver': 'liblinear', 'max_iter': 2000}
OPTIMAL_SVM_PARAMS = {'C': 0.05, 'kernel': 'linear', 'gamma': 'scale', 'shrinking': True, 'max_iter':2000} 

OPTIMAL_XGB_PARAMS = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 100, 'learning_rate': 0.001,
                      'max_depth': 1, 'subsample': 0.05, 'colsample_bytree': 0.4, 'gamma': 1, 'min_child_weight': 0.6,
                      'reg_alpha': 0.9}

OPTIMAL_RF_PARAMS = {'n_estimators': 500, 'criterion': 'entropy', 'max_depth': 1, 'min_samples_split': 2,
                     'min_samples_leaf': 2, 'max_leaf_nodes': None, 'bootstrap': True, 'max_samples': 0.25,
                      'oob_score': True, 'ccp_alpha': 0.001,  'verbose': 0}



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
#           <<< Implementación de NearMiss >>>           
# =========================================================

def apply_nearmiss_on_train_data(X_train_scaled: pd.DataFrame, y_train: pd.Series, random_seed: int) -> (pd.DataFrame, pd.Series):
    """
    Aplica la técnica de submuestreo NearMiss sobre el conjunto de entrenamiento escalado.
    """
    logging.info("Aplicando técnica NearMiss sobre el conjunto de entrenamiento...")
    logging.info(f"Parámetros de NearMiss: {NEAR_MISS_PARAMS}")
    
    if X_train_scaled.empty or y_train.empty:
        logging.warning("NearMiss: DataFrame de entrenamiento vacío, no se puede aplicar resampling.")
        return X_train_scaled, y_train

    # imblearn devuelve arrays de numpy, es necesario reconvertirlos a DataFrame/Series
    try:
        start_time = time.perf_counter()

        # Lógica de NearMiss
        sampler = NearMiss(**NEAR_MISS_PARAMS)
        X_res_np, y_res_np = sampler.fit_resample(X_train_scaled, y_train)
        
        # Medición del tiempo de ejecución de NearMiss
        resampling_time = time.perf_counter() - start_time
        logging.info(f"Tiempo NearMiss: {resampling_time:.4f} segundos")

        # Reconvertir a pandas manteniendo nombres de columnas
        X_res = pd.DataFrame(X_res_np, columns=X_train_scaled.columns)
        y_res = pd.Series(y_res_np, name=y_train.name)

        # Mezclar los datos para evitar cualquier orden inherente del submuestreo
        # Aunque NearMiss no necesariamente ordena, es una buena práctica
        shuffled_indices = np.array(X_res.index)
        np.random.seed(random_seed)
        np.random.shuffle(shuffled_indices)
        
        X_res = X_res.loc[shuffled_indices].reset_index(drop=True)
        y_res = y_res.loc[shuffled_indices].reset_index(drop=True)

        logging.info(f"NearMiss: Tamaño del conjunto de entrenamiento original: {len(X_train_scaled)}, nuevo tamaño: {len(X_res)}")
        logging.info(f"Distribución de clases después de NearMiss: {dict(y_res.value_counts())}")
        return X_res, y_res, resampling_time
        
    except Exception as e:
        logging.error(f"Error al aplicar NearMiss: {e}. ATENCIÓN: Se usará el conjunto de entrenamiento sin balancear.")
        return X_train_scaled, y_train, 0.0



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
# #def prepare_data(df_original, target_col_name, random_seed):
#    """Prepara los datos siguiendo el flujo correcto: Split -> Scale -> Balance."""
#    logging.info(">> Iniciando preparación de datos (Split -> Scale -> Balance con NearMiss)")
#    X_original, y_original = df_original.drop(target_col_name, axis=1), df_original[target_col_name]
#    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, stratify=y_original, random_state=random_seed)
#    scaler = MinMaxScaler()
#    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
#    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)
#    
#    # Aplicar NearMiss SOLO al conjunto de entrenamiento escalado
#    X_train_final, y_train_final, resampling_time = apply_nearmiss_on_train_data(X_train_scaled, y_train, random_seed)
#    
#    return X_train_final, X_test_scaled, y_train_final, y_test, scaler, resampling_time


# =========================================================
#          FUNCIONES ESPECÍFICAS DE CADA MODELO
# =========================================================

def train_nn_model(X_train, y_train, output_dir, exp_name, random_seed):
    logging.info(">>> Iniciando entrenamiento del modelo de Red Neuronal (NN)...")
    if X_train.empty: logging.error("X_train está vacío. No se puede entrenar el modelo NN."); return None
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed)

    model = Sequential([Dense(32, input_shape=(X_train_new.shape[1],), activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']),
                        Dense(32, activation='softplus'), Dropout(OPTIMAL_NN_PARAMS['dropout_rate']), Dense(2, activation='softmax')])
    
    model.compile(optimizer=Adam(learning_rate=OPTIMAL_NN_PARAMS['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=1e-6, verbose=0)]
   
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
    #logging.info(f" Modelo LR (joblib) guardado en: {model_path}")
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
    #logging.info(f" Modelo SVM (joblib) guardado en: {model_path}")
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
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=random_seed
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
    #logging.info(f" Modelo XGBoost (joblib) guardado en: {model_path}")
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
#           <<< VALIDACIÓN CRUZADA CON NEARMISS >>>
# =========================================================
# Ejecuta Stratified 5-fold cross-validation sin leakage.
# En cada fold, el escalamiento se ajusta solo con el training fold,
# NearMiss se aplica exclusivamente sobre ese training fold, y los modelos
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

        # CAMBIO CLAVE: NEARMISS SOLO EN TRAIN
        X_res, y_res, r_time = apply_nearmiss_on_train_data(
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
            # # tiempo total por fold = tiempo de NearMiss + tiempo de entrenamiento del modelo.
            time_records.append({
                'tecnica': 'NearMiss',
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
#             BLOQUE PRINCIPAL DE EJECUCIÓN
# =========================================================

global_start_time = time.perf_counter()


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    all_results = []
    time_records = []
    df_original = load_data(CSV_PATH)

    for i in range(1, N_SIMULATIONS + 1):
        # Nombres y directorios actualizados para NearMiss
        sim_start_time = time.perf_counter()
        exp_name_sim = f"NearMiss_Compare_Sim_{i}"
        output_dir_sim = os.path.join(BASE_OUTPUT, exp_name_sim)
        log_file_sim = os.path.join(output_dir_sim, f"run_{exp_name_sim}.log")
        setup_logging(log_file_sim)

        logging.info(f"Intel Extension for Scikit-learn activa: {INTEL_PATCH_ACTIVE}")
        
        print(f"\n{'='*25} INICIANDO SIMULACIÓN {i}/{N_SIMULATIONS} {'='*25}")
        logging.info(f"=================================================")
        logging.info(f"=== Iniciando Simulación Comparativa (NearMiss): {i}/{N_SIMULATIONS} ===")
        
        current_seed = i 
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)
        logging.info(f"Semillas (Numpy, TF, Split, Sklearn, XGB, RF) establecidas en: {current_seed}")
        
        sim_results = {'Simulacion': i}

        # =================================================
        # Línea del flujo anterior con train_test_split.
        # =================================================
        #X_train, X_test, y_train, y_test, _, resampling_time  = prepare_data(df_original, TARGET_COLUMN_NAME, random_seed=current_seed)
        

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


        for model_name, (train_func, eval_func) in model_runners.items():
            logging.info(f"\n--- Procesando Modelo: {model_name} ---")
            
            sim_results[f'recall_{model_name}_NearMiss'] = cv_results.get(f'recall_{model_name}')
            sim_results[f'precision_{model_name}_NearMiss'] = cv_results.get(f'precision_{model_name}')
            sim_results[f'roc_auc_{model_name}_NearMiss'] = cv_results.get(f'roc_auc_{model_name}')
            sim_results[f'pr_auc_{model_name}_NearMiss'] = cv_results.get(f'pr_auc_{model_name}')
            sim_results[f'time_{model_name}_NearMiss'] = cv_results.get(f'time_{model_name}')
            sim_results[f'total_time_{model_name}_NearMiss'] = cv_results.get(f'total_time_{model_name}')
            sim_results[f'f1_{model_name}_NearMiss'] = cv_results.get(f'f1_{model_name}')
            sim_results[f'accuracy_{model_name}_NearMiss'] = cv_results.get(f'accuracy_{model_name}')


        sim_total_time = time.perf_counter() - sim_start_time
        logging.info(f"Tiempo total de la simulación {i}: {sim_total_time:.4f} segundos")

        all_results.append(sim_results)
        
        print(f"--- Simulación {i} completada. ---")
        logging.info(f"=== Fin Simulación {i} ===\n")

    print(f"\n{'='*25} TODAS LAS SIMULACIONES COMPLETADAS {'='*25}")

    global_total_time = time.perf_counter() - global_start_time
    print(f"\n Tiempo TOTAL de ejecución del script: {global_total_time:.4f} segundos")
 
 
    # =========================================================
    #               REPORTE FINAL COMPLETO
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

        stats_summary = "Estadísticas de Robustez (NEARMISS + 5-Fold CV)\n"
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
        stats_summary += "Tiempo considerado por fold: tiempo de NearMiss + tiempo de entrenamiento del modelo.\n"
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
                stats_summary += f"--- total_time_{row['model']}_NEARMISS ---\n"
                stats_summary += f"   - Media:           {row['media_segundos']:.4f} \n"
                stats_summary += f"   - Desv. Estandar: {row['desv_segundos']:.4f} \n"
                stats_summary += f"   - Minimo:         {row['min_segundos']:.4f} \n"
                stats_summary += f"   - Maximo:         {row['max_segundos']:.4f} \n\n"

        #print("\n" + stats_summary)

        # Guardar CSV completo
        summary_csv_path = os.path.join(BASE_OUTPUT, 'ALL_MODELS_NEARMISS_summary.csv')
        results_df.to_csv(summary_csv_path, index=False)
        print(f"Resultados guardados en: {summary_csv_path}")

        # Guardar resumen estadístico
        stats_summary_path = os.path.join(BASE_OUTPUT, 'Statistics_summary_NEARMISS.txt')
        with open(stats_summary_path, 'w') as f:
            f.write(stats_summary)
        print(f"Estadísticas guardadas en: {stats_summary_path}")