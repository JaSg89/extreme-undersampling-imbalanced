<h1 align="center">
  Addressing Class Imbalance in Credit Card Fraud Detection<br>
  Using Mahalanobis-Based Undersampling of Extreme Observations
</h1>

<p align="center">
  <em>Source code accompanying the research paper submitted to Elsevier</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-1.x-006600?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/imbalanced--learn-0.x-brightgreen" alt="imbalanced-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

---

## 👥 Authors

**Jorge Saavedra-Garrido**<sup>a</sup>, **Daira Velandia**<sup>a,b,*</sup>, **Keyla Canibilo Rojas**<sup>a</sup>, **Cristian Ubal**<sup>c</sup>, **Eyleen Spencer**<sup>d</sup>, **Rodrigo Salas**<sup>d,e,f</sup>

<sup>a</sup> Statistical Institute, Faculty of Science, Universidad de Valparaíso, Valparaíso, Chile  
<sup>b</sup> Center for Atmospheric Studies and Climate Change (CEACC), Universidad de Valparaíso, Chile  
<sup>c</sup> Department of Data Science, Faculty of Engineering, Universidad de Playa Ancha, Valparaíso, Chile  
<sup>d</sup> Biomedical Engineering School, Faculty of Engineering, Universidad de Valparaíso, Chile  
<sup>e</sup> Center of Interdisciplinary Biomedical and Engineering Research for Health (MEDING), Universidad de Valparaíso, Chile  
<sup>f</sup> Millennium Institute for Intelligent Healthcare Engineering (iHealth), Santiago, Chile

<sup>*</sup> Corresponding author: daira.velandia@uv.cl

---

## 📋 Abstract

Credit card fraud remains a significant challenge for financial institutions, exacerbated by the rapid growth of digital payments. A key difficulty is the **extreme class imbalance**, which biases models toward the majority class. While resampling is widely used, it is often applied before the train–test split, causing **data leakage** and yielding overestimated model performance.

In this study, we propose a **Mahalanobis distance-based undersampling strategy** that selects extreme observations to improve class separation and reduce computational complexity. We evaluate two variants:

- **MEUS** (*Majority-class Extreme-aware UnderSampling*) — performs 1-to-1 nearest-neighbor matching between minority and majority classes using Mahalanobis distance, producing a perfectly balanced dataset.
- **FEUS** (*Furthest-point Extreme UnderSampling*) — selects the *N* samples furthest from the data centroid using Mahalanobis distance, retaining the most informative boundary and outlier instances.

Under a **leakage-free protocol** where resampling is limited to the training folds, our variant **FEUS achieves the highest macro F1-Score and the highest mean precision**, with a modest decrease in mean recall. At the model level, FEUS delivers the best F1-Score for SVM, ANN, and XGBoost, while MEUS performs best for LR and RF. This profile positions **FEUS as a strategy that maximizes prediction reliability without sacrificing detection ability**.

**Keywords:** Credit Card Fraud · Class Imbalance · Extreme Values · Mahalanobis Distance · Machine Learning · Data Leakage

---

## 🔬 Resampling Techniques Evaluated

### Proposed Methods

| Technique | Type | Script | Description |
|-----------|------|--------|-------------|
| **FEUS** | Under-sampling | `new_FEUS.py` | Calculates the Mahalanobis distance of each sample from the global centroid and retains the *N<sub>keep</sub>* most extreme (furthest) observations from the entire training set. Enhances class separability by preserving boundary-relevant instances. |
| **MEUS** | Under-sampling | `new_MEUS.py` | Uses 1-to-1 nearest-neighbor matching via Mahalanobis distance: for each minority instance, the closest unclaimed majority instance is selected, producing a balanced dataset of size 2·|S<sub>min</sub>|. |

### Baseline Methods

| Technique | Type | Script | Description |
|-----------|------|--------|-------------|
| **SMOTE** | Over-sampling | `new_SMOTE.py` | Synthetic Minority Over-sampling Technique — generates synthetic minority samples by interpolating between existing instances and their k-nearest neighbors. |
| **NearMiss** | Under-sampling | `new_NEARMISS.py` | Selects majority samples closest to minority instances using k-NN heuristics, focusing retention on decision-boundary regions. |
| **RUS** | Under-sampling | `new_UNDERSAMPLE.py` | Random Under-Sampling — randomly discards majority-class samples until class balance is achieved. |
| **ENN** | Under-sampling | `new_ENN.py` | Edited Nearest Neighbours — removes samples whose class label disagrees with the majority of their *k* nearest neighbors (data cleaning). |
| **Tomek Links** | Under-sampling | `new_TOMEKLINKS.py` | Removes majority-class members of Tomek pairs (nearest-neighbor pairs from opposite classes), cleaning the decision boundary. |

### Additional Scripts

| Script | Description |
|--------|-------------|
| `FEUS_euclidean_mahalanobis.py` | Ablation study comparing **Euclidean vs. Mahalanobis** distance metrics within the FEUS framework. |
| `TEST.py` | **Statistical analysis** — Friedman test, Nemenyi post-hoc comparisons, bootstrap confidence intervals (2,000 resamples), heatmaps, boxplots, and precision-recall scatter plots. |
| `opentSNE.py` | **t-SNE visualization** of resampled data distributions using [openTSNE](https://opentsne.readthedocs.io/), generating publication-quality figures for both MEUS and FEUS. |

---

## 🤖 Classification Models

Each resampling technique is evaluated using five supervised classifiers, as described in the paper:

| Model | Abbreviation | Framework | Key Configuration |
|-------|-------------|-----------|-------------------|
| Artificial Neural Network (MLP) | ANN | TensorFlow / Keras | 32→16 neurons, softplus activation, dropout, Adam optimizer |
| Logistic Regression | LR | scikit-learn | L2 penalty, SAGA solver |
| Support Vector Machine | SVM | scikit-learn | RBF kernel, probability calibration |
| Extreme Gradient Boosting | XGBoost | XGBoost | Early stopping, regularization |
| Random Forest | RF | scikit-learn | 500 trees, entropy criterion, OOB scoring |

Additionally, a **1-Nearest Neighbor (1NN)** classifier is included as a baseline reference.

---

## 📊 Evaluation Metrics

All experiments report the following metrics (macro-averaged where applicable):

| Metric | Description |
|--------|-------------|
| **Recall** (Sensitivity) | Ability to correctly identify fraudulent transactions |
| **Precision** | Reliability of positive predictions |
| **F1-Score** | Harmonic mean of precision and recall — the most informative metric under severe imbalance |
| **Accuracy** | Overall correctness (>99% due to imbalance, limited diagnostic value) |
| **ROC-AUC** | Threshold-independent overall discrimination capability |
| **PR-AUC** | Precision-Recall trade-off — especially relevant under extreme imbalance |

Results are summarized with:
- Mean ± Standard Deviation across 30 independent simulations
- 95% Confidence Intervals (bootstrap with 2,000 resamples)
- Friedman test + Nemenyi post-hoc comparisons for statistical significance

---

## 🏗️ Experimental Design

```
┌────────────────────────────────────────────────────────────────┐
│                    FOR EACH SIMULATION (1..30)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              STRATIFIED 5-FOLD CROSS-VALIDATION          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  FOR EACH FOLD (1..5)                              │  │  │
│  │  │                                                    │  │  │
│  │  │  1. Split → Train / Validation                     │  │  │
│  │  │  2. Scale (MinMaxScaler) — fit on Train ONLY       │  │  │
│  │  │  3. Resample Train ONLY (no data leakage)          │  │  │
│  │  │  4. Train classifiers on resampled Train           │  │  │
│  │  │  5. Evaluate on original (unmodified) Validation   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  Average metrics across 5 folds                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Record averaged metrics for this simulation                   │
└────────────────────────────────────────────────────────────────┘
        ↓
   Aggregate statistics over 30 simulations
   (Mean, Std, CI, Min, Max) + Friedman/Nemenyi tests
```

### Key Design Principles

- **No data leakage**: Scaling (`MinMaxScaler`) is fitted exclusively on the training fold; the validation fold is transformed using the same scaler but never participates in fitting. Resampling is applied **after** the train–test split.
- **Resampling isolation**: All resampling methods are applied strictly to the training fold. The validation fold remains unmodified, preserving its statistical independence.
- **Reproducibility**: Each simulation uses a deterministic seed (simulation index) for NumPy, TensorFlow, and all stochastic components across partitioning, shuffling, and model initialization.
- **Hardware acceleration**: Optional Intel Extension for scikit-learn (`sklearnex`) is integrated for transparent CPU acceleration on supported hardware.
- **Hyperparameter optimization**: Conducted via stratified 4-fold cross-validated GridSearchCV with ROC-AUC as the optimization metric (details in Supplementary Table S1 of the paper).

---

## 📂 Dataset

The primary dataset used in this study is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, containing **284,807** anonymized credit card transactions made by European cardholders in September 2013:

- **Fraudulent transactions**: 492 (0.17%)
- **Legitimate transactions**: 284,315 (99.83%)
- **Imbalance Ratio (IR)**: 578.88
- **Features**: 28 PCA-transformed components (V1–V28) + Time + Amount
- **Target**: `Class` (1 = Fraud, 0 = Genuine)

---

## ⚙️ Requirements

### Python Dependencies

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
tensorflow >= 2.x
xgboost
imbalanced-learn
joblib
matplotlib
seaborn
scikit-posthocs       # for Nemenyi post-hoc test (TEST.py)
openTSNE              # for t-SNE visualization (opentSNE.py)
sklearnex             # optional: Intel Extension for scikit-learn
```

### Installation

```bash
pip install numpy pandas scipy scikit-learn tensorflow xgboost imbalanced-learn joblib matplotlib seaborn scikit-posthocs openTSNE
```

For Intel acceleration (optional):

```bash
pip install scikit-learn-intelex
```

---

## 📂 Repository Structure

```
extreme-undersampling-imbalanced/
│
├── README.md                            # This file
├── LICENSE                              # MIT License
├── .gitignore                           # Git ignore rules
│
│── ── Proposed Methods ──────────────────
├── new_FEUS.py                          # FEUS — Furthest-point Extreme UnderSampling
├── new_MEUS.py                          # MEUS — Majority-class Extreme-aware UnderSampling
├── FEUS_euclidean_mahalanobis.py         # Ablation: Euclidean vs. Mahalanobis in FEUS
│
│── ── Baseline Methods ──────────────────
├── new_SMOTE.py                         # SMOTE (over-sampling baseline)
├── new_NEARMISS.py                      # NearMiss (under-sampling baseline)
├── new_UNDERSAMPLE.py                   # RUS — Random Under-Sampling (baseline)
├── new_ENN.py                           # ENN — Edited Nearest Neighbours (baseline)
├── new_TOMEKLINKS.py                    # TL  — Tomek Links (baseline)
│
│── ── Analysis & Visualization ──────────
├── TEST.py                              # Statistical tests (Friedman/Nemenyi) & plots
└── opentSNE.py                          # t-SNE embedding visualization
```

---

## 🚀 Usage

### Running an Experiment

Each `new_*.py` script is a self-contained experiment. To run, for example, the FEUS experiment:

```bash
python new_FEUS.py
```

> **⚠️ Important:** Before running, update the `CSV_PATH` and `BASE_OUTPUT` variables at the top of each script to point to your local data files and desired output directory.

### Running Statistical Analysis

After all experiments have completed and CSV summaries are generated:

```bash
python TEST.py
```

> **Note:** Update the CSV file paths inside `TEST.py` to match your output directories.

### Generating t-SNE Visualizations

```bash
python opentSNE.py
```

---

## 📄 Output Files

Each experiment generates:

| File | Content |
|------|---------|
| `ALL_MODELS_<TECHNIQUE>_summary.csv` | Per-simulation metrics for all classifiers |
| `Statistics_summary_<TECHNIQUE>.txt` | Aggregated statistics: mean, std, CI, min, max |
| `run_<experiment>.log` | Detailed execution log per simulation |

---

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{saavedra2026mahalanobis,
  title     = {Addressing Class Imbalance in Credit Card Fraud Detection 
               Using Mahalanobis-Based Undersampling of Extreme Observations},
  author    = {Saavedra-Garrido Jorge, Velandia Daira, Canibilo-Rojas Keyla, Ubal Cristian, Spencer Eyleen and Salas Rodrigo},
  journal   = {Preprint submitted to Elsevier},
  year      = {2026},
  keywords  = {Credit Card Fraud, Class Imbalance, Extreme Values, 
               Mahalanobis Distance, Machine Learning, Data Leakage}
}
```

---

## 💰 Funding

This research was funded by:
- **FONDECYT** project N° 1221938
- **ANID Millennium Science Initiative Program** ICN2021_004
- **Beca de Doctorado Nacional** 21231546

The authors acknowledge the support given by the Center of Interdisciplinary Biomedical and Engineering Research for Health (MEDING), Universidad de Valparaíso, Chile.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Developed for reproducible machine learning research in imbalanced classification</em>
</p>
