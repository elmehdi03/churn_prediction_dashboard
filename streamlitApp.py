import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

# Configuration de la page avec icône
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
page_style = """
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #e0f7fa, #80deea);}
[data-testid="stSidebar"] {background: linear-gradient(135deg, #80deea, #26c6da);}
.css-1aumxhk .stSlider > div > div > div:nth-child(2) {background: linear-gradient(90deg, #0288d1, #26c6da);}
h1.custom-title {color: #01579b; font-family: 'Arial Black', sans-serif; text-align: center; margin-top: 20px;}
.reco-box {background: rgba(255,255,255,0.8); padding: 20px; border-left: 5px solid #0288d1; border-radius: 8px; margin-top: 20px;}
.reco-box ul {list-style-type: none; padding-left: 0;}
.reco-box li::before {content: "✔️"; margin-right: 8px;}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Utilitaire pour trouver un .joblib
def find_joblib_file(prefix):
    return next((f for f in os.listdir() if f.lower().startswith(prefix.lower()) and f.endswith('.joblib')), None)

# Titre principal
st.markdown("<h1 class='custom-title'>📈 Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Chargement des données
def load_data(path='synthetic_moroccan_churn_1M.csv'):
    if not os.path.exists(path): st.error(f"Fichier introuvable: {path}"); st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_cached():
    """Charge et met en cache le jeu de données par défaut."""
    return load_data()

# Sidebar: import et aide
st.sidebar.markdown("## 📂 Importer les données CSV")
uploader = st.sidebar.file_uploader("Upload un .csv", type=["csv"] )
if uploader:
    try:
        data = pd.read_csv(uploader)
        st.sidebar.success("Données chargées.")
    except Exception as e:
        st.sidebar.error(f"Erreur lecture: {e}")
        data = load_cached()
else:
    data = load_cached()

# Aide utilisateur
with st.sidebar.expander("❓ Aide", expanded=False):
    st.markdown(
        "- Sélectionnez une section dans le menu pour explorer les données."
        "\n- Changez le seuil de risque pour visualiser les clients à identifier."
        "\n- Importez un CSV personnalisé si besoin." 
    )

# Préprocessing global pour métriques et sections
cat_cols = ["contrat","internet_service","langue_preferee","region","methode_paiement","type_appareil","operateur"]
# Pipeline pour X_final
model = joblib.load(find_joblib_file('model_lightgbm') or 'model_lightgbm_churn.joblib')
scaler = joblib.load("scaler_churn.joblib")
scale_feats = joblib.load("scaler_features.joblib")
X_enc = pd.get_dummies(data.drop(columns=['churn']), columns=cat_cols).reindex(columns=scale_feats, fill_value=0)
X_scaled = scaler.transform(X_enc)
pca_file = find_joblib_file('pca')
if pca_file:
    pca = joblib.load(pca_file)
    X_final = pca.transform(X_scaled)
else:
    X_final = X_scaled
# Vérités et prédictifs pour KPI
y_true = data['churn']
try:
    y_probs = model.predict_proba(X_final)[:,1]
    y_pred = model.predict(X_final)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
except:
    acc = prec = auc = None

# KPI en en-tête
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total clients", f"{len(data)}")
col2.metric("Taux de churn", f"{y_true.mean():.2%}")
if acc is not None:
    col3.metric("Accuracy", f"{acc:.2%}")
    col4.metric("AUC", f"{auc:.2%}")
st.markdown("---")

# Sections
st.sidebar.markdown("## 🔍 Sections")
section = st.sidebar.radio(
    "Choisissez une vue :",
    ["Distribution du churn", "Importances LightGBM", "Performance modèle", "SHAP Summary", "Heatmap corrélation", "Clients à risques"]
)

# 1) Distribution du churn
if section == "Distribution du churn":
    st.subheader("🎯 Répartition churn vs non-churn")
    cnt = data['churn'].value_counts().sort_index()
    fig = go.Figure(go.Pie(labels=['Non-churn','Churn'], values=cnt.values,
        hole=0.4, marker=dict(colors=['#0288d1','#d81b60'], line=dict(color='#FFF',width=2)), pull=[0.05,0]
    ))
    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
                      annotations=[dict(text='Churn',x=0.5,y=0.5,font_size=20,showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

# 2) Importances LightGBM
elif section == "Importances LightGBM":
    st.subheader("🔑 Feature Importance (LightGBM)")
    raw_feats = joblib.load(find_joblib_file('features') or 'features.joblib')
    imp = model.feature_importances_
    labels = raw_feats if len(raw_feats)==len(imp) else [f'PC{i+1}' for i in range(len(imp))]
    df_imp = pd.DataFrame({'Feature': labels, 'Importance': imp}).sort_values('Importance', ascending=False)
    fig = go.Figure(go.Bar(x=df_imp['Importance'], y=df_imp['Feature'], orientation='h',
        marker=dict(color=df_imp['Importance'], colorscale='Viridis', showscale=True, line=dict(color='#FFF', width=1))
    ))
    fig.update_layout(title='Feature Importance', xaxis_title='Importance', yaxis_title='Feature',
                      template='plotly_white', height=600, margin=dict(l=150, r=20, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

# 3) Performance modèle
elif section == "Performance modèle":
    st.subheader("📈 Courbes de performance du modèle")
    if acc is None:
        st.warning("Impossible de calculer les métriques de performance.")
    else:
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc:.2f}', line=dict(color='#d81b60')))
        fig1.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline', line=dict(dash='dash')))
        fig1.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                           template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall', line=dict(color='#0288d1')))
        fig2.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision',
                           template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

# 4) SHAP Summary
elif section == "SHAP Summary":
    st.subheader("🔍 SHAP Summary Plot")
    import shap
    scaler = joblib.load("scaler_churn.joblib")
    feats = joblib.load("scaler_features.joblib")
    X_enc = pd.get_dummies(data.drop(columns=['churn']), columns=cat_cols).reindex(columns=feats, fill_value=0)
    X_scaled = scaler.transform(X_enc)
    n = len(model.feature_importances_)
    pca_file = find_joblib_file('pca')
    pca = joblib.load(pca_file) if pca_file else PCA(n_components=n).fit(X_scaled)
    df_pca = pd.DataFrame(pca.transform(X_scaled), columns=[f'PC{i+1}' for i in range(n)])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df_pca, check_additivity=False)
    fig = plt.figure(figsize=(8,6))
    shap.summary_plot(sv[1] if isinstance(sv, list) else sv, df_pca, show=False)
    st.pyplot(fig)

# 5) Heatmap corrélation
elif section == "Heatmap corrélation":
    st.subheader("📊 Heatmap de corrélation")
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    corr = data[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                    title='Matrice de corrélation')
    st.plotly_chart(fig, use_container_width=True)

# 6) Clients à risques
else:
    st.subheader("⚠️ Clients à Haut Risque")
    scaler = joblib.load("scaler_churn.joblib")
    feats = joblib.load("scaler_features.joblib")
    X_enc = pd.get_dummies(data.drop(columns=['churn']), columns=cat_cols).reindex(columns=feats, fill_value=0)
    X_scaled = scaler.transform(X_enc)
    pca_file = find_joblib_file('pca')
    X_final = joblib.load(pca_file).transform(X_scaled) if pca_file else X_scaled
    probs = model.predict_proba(X_final)[:,1]
    data['risk_score'] = probs
    threshold = st.slider('Seuil de risque', 0.0, 1.0, 0.5)
    # Histogramme des scores
    st.subheader("Distribution des scores de risque")
    fig_hist = px.histogram(data, x='risk_score', nbins=30, title='Histogramme des risk_score',
                            labels={'risk_score':'Score de risque'})
    st.plotly_chart(fig_hist, use_container_width=True)
    high = data[data['risk_score']>=threshold]
    st.sidebar.markdown("## ℹ️ Infos modèle")
    st.sidebar.markdown(f"- **Modèle**: LightGBM<br>- **Seuil**: {threshold:.2f}<br>- **Clients à risque**: {len(high)}", unsafe_allow_html=True)
    st.metric("Clients à risque", len(high))
    st.dataframe(high.head(10), use_container_width=True)
    st.markdown(
        """
        <div class='reco-box'>
          <h4>💡 Recommandations :</h4>
          <ul>
            <li>🎯 Campagne ciblée sur segments à fort churn</li>
            <li>📧 Emails/SMS personnalisés</li>
            <li>📞 Appels proactifs VIP</li>
            <li>📊 Suivi temps réel et ajustement du seuil</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
# Conclusion
st.markdown("---")
st.subheader("📊 Conclusion")

st.markdown(
    "Ce tableau de bord permet d'explorer les données de churn, d'évaluer la performance du modèle et d'identifier les clients à risque. "
    "Utilisez les sections pour naviguer entre les visualisations et insights."
)       
st.markdown(
    "Pour plus d'informations, consultez la documentation sur Github ou contactez l'équipe de développement."
)
# Footer
st.markdown("---")  
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "© 2025 Churn Detection Team - Tous droits réservés."
    "</p>",
    unsafe_allow_html=True
)

