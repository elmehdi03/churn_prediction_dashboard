# 📈 Moroccan Telecom Churn Prediction Dashboard

Bienvenue dans ce projet de **détection du churn client** basé sur un jeu de données synthétique représentant les abonnés d'un opérateur télécom au Maroc. Il s'agit d'une **application interactive développée avec Streamlit**, intégrant des modèles de machine learning optimisés, une interface visuelle soignée et des outils d'analyse avancée.

---

## 🎯 Objectifs

- Détecter les clients à risque de résiliation (churn) avec haute précision
- Proposer des recommandations ciblées pour la fidélisation
- Permettre une visualisation dynamique et personnalisée des données
- Valoriser les compétences en **Big Data Analytics**, **ML Explainability** et **Hyperparameter Optimization**

---

## ✨ Caractéristiques principales

- **1M données synthétiques** avec modélisation probabiliste réaliste du churn
- **60+ features engineered** (interactions, ratios, indicateurs de risque, loyalty score)
- **Modèles optimisés** avec hyperparameter tuning Bayésien (Optuna)
- **Seuil de décision optimisé** (0.300) pour maximiser le profit business
- **ROC-AUC: 0.726** avec 62% de recall sur les churners
- **Interface interactive** avec dashboard Streamlit
- **Explicabilité** avec analyse SHAP et feature importance

---

## 🧰 Technologies utilisées

- **Python 3.10+**
- **Streamlit** (application web interactive)
- **LightGBM & XGBoost** (modèles de classification optimisés)
- **Optuna** (optimisation bayésienne des hyperparamètres)
- **Scikit-learn** (prétraitement, métriques, validation)
- **SHAP** (explicabilité des prédictions)
- **Plotly & Matplotlib** (visualisations interactives)
- **Joblib** (sérialisation des modèles)
- **Pandas / NumPy** (traitement de données)

---

## 🚀 Installation et lancement

### Prérequis
- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/elmehdi03/churn_prediction_dashboard.git
cd churn_prediction_dashboard
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer l'application**
```bash
streamlit run streamlitApp.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

---

## 📁 Structure du projet
```
churn_prediction_dashboard/
├── data/                                    # Données
│   ├── synthetic_moroccan_churn_1M.csv     # Dataset RAW (1M lignes, 11 colonnes)
│   └── README.md                            # Documentation des données
├── models/                                  # Modèles entraînés et artefacts
│   ├── model_lightgbm_churn.joblib         # Modèle LightGBM (baseline)
│   ├── model_lightgbm_tuned_churn.joblib   # Modèle LightGBM optimisé
│   ├── model_xgboost_churn.joblib          # Modèle XGBoost
│   ├── model_best_churn.joblib             # Meilleur modèle
│   ├── best_hyperparameters.joblib         # Hyperparamètres optimaux
│   ├── encoder.joblib                       # OneHotEncoder
│   ├── scaler_churn.joblib                 # StandardScaler
│   ├── features.joblib                      # Noms des features (60)
│   ├── categorical_columns.joblib           # Colonnes catégorielles
│   ├── numerical_columns.joblib             # Colonnes numériques
│   ├── binary_columns.joblib                # Colonnes binaires
│   ├── scaler_features.joblib              # Features à standardiser
│   ├── optimal_threshold.joblib            # Seuil optimal (0.300)
│   └── README.md                            # Documentation des modèles
├── streamlitApp.py                          # Application Streamlit principale
├── NoteBook.ipynb                           # Pipeline ML complet (24 cellules)
│                                            # - Data generation (realistic churn)
│                                            # - Feature engineering (21 features)
│                                            # - Preprocessing & encoding
│                                            # - Model training & evaluation
│                                            # - Hyperparameter tuning (Optuna)
│                                            # - Threshold optimization (business value)
├── requirements.txt                         # Dépendances Python
├── .gitignore                               # Fichiers à ignorer par Git
├── LICENSE                                  # Licence du projet (MIT)
└── README.md                                # Documentation (ce fichier)
```

---

## 🔬 Pipeline Machine Learning

Le notebook `NoteBook.ipynb` contient le pipeline complet :

### 1. **Génération de données synthétiques**
   - 1M clients avec 10 features de base
   - Modélisation probabiliste du churn (9 facteurs pondérés)
   - Taux de churn réaliste : 60.46%
   - **Format RAW**: 11 colonnes (7 catégorielles, 3 numériques, 1 target)

### 2. **Feature Engineering (automatique)**
   - Le Streamlit app crée automatiquement toutes les features:
   - 4 features d'interaction (ex: `is_young_prepaid`)
   - 2 features de ratio (ex: `tenure_income_ratio`)
   - 3 features catégorielles binées (age_group, revenue_tier, tenure_category)
   - 7 indicateurs de risque binaires
   - 1 score de fidélité composite
   - 3 features statistiques (z-score, percentiles)
   - **Total: 31 features créées → 60 features après encoding**

### 3. **Prétraitement**
   - One-Hot Encoding (10 colonnes catégorielles → 40 features)
   - Standardisation des features numériques uniquement
   - Pas de PCA (préserve l'interprétabilité)

### 4. **Entraînement & Évaluation**
   - Train/Test split: 800k/200k (stratifié)
   - Modèles: XGBoost & LightGBM
   - Cross-validation 5-fold
   - Métriques: ROC-AUC, Precision, Recall, F1-Score

### 5. **Optimisation des hyperparamètres**
   - Framework: Optuna (Bayesian optimization)
   - 50 trials avec TPE sampler
   - Optimisation sur 200k échantillon (3-fold CV)
   - Temps: ~3.5 minutes
   - **Amélioration: +0.06% ROC-AUC**

### 6. **Optimisation du seuil de décision**
   - Tests de 50 seuils (0.30 - 0.80)
   - Analyse coûts business:
     - Faux Négatif (churner manqué): $100
     - Faux Positif (campagne inutile): $10
     - Vrai Positif (client sauvé): -$20 (gain net)
   - **Seuil optimal: 0.300** (maximise le profit)
   - Améliore le recall tout en minimisant les coûts

### 7. **Résultats finaux**
   - **LightGBM optimisé (seuil 0.300):**
     - ROC-AUC: **0.7263**
     - Accuracy: **65.38%**
     - Precision: **76%**
     - Recall: **62%** (détecte 62% des churners)
     - F1-Score: **0.68**
     - **Seuil de décision: 0.300** (optimisé pour profit)

---

## 📊 Comparaison des performances

| Métrique | Avant optimisation | Après Optuna | Amélioration |
|----------|-------------------|--------------|-------------|
| ROC-AUC | 0.7257 | 0.7263 | +0.06% |
| Accuracy | 65.00% | 65.38% | +0.38% |
| Recall | 61% | 62% | +1% |
| F1-Score | 0.68 | 0.68 | Stable |

**Impact business:** Sur 120,925 churners dans le test set, le modèle optimisé en détecte **~75,000**, soit environ 600 clients supplémentaires par rapport au modèle de base.

---

## 👤 Auteur

Développé par **El Mehdi El Youbi Rmich**  
📍 Maroc | 📧 mehdi.eloubi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/el-mehdi-el-youbi-rmich-574941249/)  
