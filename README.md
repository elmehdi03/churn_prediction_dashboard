# 📈 Moroccan Telecom Churn Prediction Dashboard

Welcome to this **customer churn detection project** based on a synthetic dataset representing subscribers of a Moroccan telecom operator. This is an **interactive application built with Streamlit**, integrating optimized machine learning models, a polished visual interface, and advanced analytics tools.

---

## 🎯 Objectives

- Detect at-risk customers (churn) with high precision
- Provide targeted recommendations for customer retention
- Enable dynamic and personalized data visualization
- Showcase expertise in **Big Data Analytics**, **ML Explainability**, and **Hyperparameter Optimization**

---

## ✨ Key Features

- **1M synthetic data points** with realistic probabilistic churn modeling
- **60+ engineered features** (interactions, ratios, risk indicators, loyalty score)
- **Optimized models** with Bayesian hyperparameter tuning (Optuna)
- **Optimized decision threshold** (0.300) to maximize business profit
- **ROC-AUC: 0.726** with 62% recall on churners
- **Interactive interface** with Streamlit dashboard
- **Explainability** with SHAP analysis and feature importance

---

## 🧰 Technologies Used

- **Python 3.10+**
- **Streamlit** (interactive web application)
- **LightGBM & XGBoost** (optimized classification models)
- **Optuna** (Bayesian hyperparameter optimization)
- **Scikit-learn** (preprocessing, metrics, validation)
- **SHAP** (prediction explainability)
- **Plotly & Matplotlib** (interactive visualizations)
- **Joblib** (model serialization)
- **Pandas / NumPy** (data processing)

---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/elmehdi03/churn_prediction_dashboard.git
cd churn_prediction_dashboard
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch the application**
```bash
streamlit run streamlitApp.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 📁 Project Structure
```
churn_prediction_dashboard/
├── data/                                    # Data
│   ├── synthetic_moroccan_churn_1M.csv      # RAW dataset (1M rows, 11 columns)
│   └── README.md                            # Data documentation
├── models/                                  # Trained models and artifacts
│   ├── model_lightgbm_churn.joblib          # LightGBM model (baseline)
│   ├── model_lightgbm_tuned_churn.joblib    # LightGBM model (optimized)
│   ├── model_xgboost_churn.joblib           # XGBoost model
│   ├── model_best_churn.joblib              # Best model
│   ├── best_hyperparameters.joblib          # Optimal hyperparameters
│   ├── encoder.joblib                       # OneHotEncoder
│   ├── scaler_churn.joblib                  # StandardScaler
│   ├── features.joblib                      # Feature names (60)
│   ├── categorical_columns.joblib           # Categorical columns
│   ├── numerical_columns.joblib             # Numerical columns
│   ├── binary_columns.joblib                # Binary columns
│   ├── scaler_features.joblib               # Features to standardize
│   ├── optimal_threshold.joblib             # Optimal threshold (0.300)
│   └── README.md                            # Model documentation
├── streamlitApp.py                          # Main Streamlit application
├── NoteBook.ipynb                           # Complete ML pipeline (24 cells)
│                                            # - Data generation (realistic churn)
│                                            # - Feature engineering (21 features)
│                                            # - Preprocessing & encoding
│                                            # - Model training & evaluation
│                                            # - Hyperparameter tuning (Optuna)
│                                            # - Threshold optimization (business value)
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Files ignored by Git
├── LICENSE                                  # Project license (MIT)
└── README.md                                # Documentation (this file)
```

---

## 🔬 Machine Learning Pipeline

The `NoteBook.ipynb` notebook contains the complete pipeline:

### 1. **Synthetic Data Generation**
   - 1M customers with 10 base features
   - Probabilistic churn modeling (9 weighted factors)
   - Realistic churn rate: 60.46%
   - **RAW format**: 11 columns (7 categorical, 3 numerical, 1 target)

### 2. **Feature Engineering (automatic)**
   - The Streamlit app automatically creates all features:
   - 4 interaction features (e.g., `is_young_prepaid`)
   - 2 ratio features (e.g., `tenure_income_ratio`)
   - 3 binned categorical features (age_group, revenue_tier, tenure_category)
   - 7 binary risk indicators
   - 1 composite loyalty score
   - 3 statistical features (z-score, percentiles)
   - **Total: 31 features created → 60 features after encoding**

### 3. **Preprocessing**
   - One-Hot Encoding (10 categorical columns → 40 features)
   - Standardization of numerical features only
   - No PCA (preserves interpretability)

### 4. **Training & Evaluation**
   - Train/Test split: 800k/200k (stratified)
   - Models: XGBoost & LightGBM
   - 5-fold cross-validation
   - Metrics: ROC-AUC, Precision, Recall, F1-Score

### 5. **Hyperparameter Optimization**
   - Framework: Optuna (Bayesian optimization)
   - 50 trials with TPE sampler
   - Optimization on 200k sample (3-fold CV)
   - Time: ~3.5 minutes
   - **Improvement: +0.06% ROC-AUC**

### 6. **Decision Threshold Optimization**
   - Tests 50 thresholds (0.30 - 0.80)
   - Business cost analysis:
     - False Negative (missed churner): $100
     - False Positive (wasted campaign): $10
     - True Positive (saved customer): -$20 (net gain)
   - **Optimal threshold: 0.300** (maximizes profit)
   - Improves recall while minimizing costs

### 7. **Final Results**
   - **Optimized LightGBM (threshold 0.300):**
     - ROC-AUC: **0.7263**
     - Accuracy: **65.38%**
     - Precision: **76%**
     - Recall: **62%** (detects 62% of churners)
     - F1-Score: **0.68**
     - **Decision threshold: 0.300** (optimized for profit)

---

## 📊 Performance Comparison

| Metric | Before Optimization | After Optuna | Improvement |
|----------|-------------------|--------------|-------------|
| ROC-AUC | 0.7257 | 0.7263 | +0.06% |
| Accuracy | 65.00% | 65.38% | +0.38% |
| Recall | 61% | 62% | +1% |
| F1-Score | 0.68 | 0.68 | Stable |

**Business impact:** On 120,925 churners in the test set, the optimized model detects **~75,000**, approximately 600 additional customers compared to the baseline model.

---

## 👤 Author

Developed by **El Mehdi EL YOUBI RMICH**  
📍 Morocco | 📧 mehdi.eloubi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/el-mehdi-el-youbi-rmich-574941249/)  

