# 📈 Churn Prediction Dashboard

Ce projet est une application Streamlit interactive pour prédire le churn des clients dans le secteur des télécoms marocain.

## 🔧 Fonctionnalités
- Visualisation des importances SHAP & LightGBM
- Courbes ROC & Precision-Recall
- Analyse des clients à haut risque
- Chargement de CSV personnalisé
- Design responsive et moderne

## 📁 Structure
- `streamlitApp.py` : App principale
- `model_lightgbm_churn.joblib` : Modèle LightGBM
- `scaler_churn.joblib` / `scaler_features.joblib` : Standardisation
- `features.joblib` : Liste des variables utilisées
- `pca.joblib` : (optionnel) réduction de dimension

## ▶️ Lancer l'application

```bash
streamlit run app/streamlitApp.py
