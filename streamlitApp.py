import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
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
h1.custom-title {color: #01579b; font-family: 'Arial Black', sans-serif; text-align: center; margin-top: 20px;}
.reco-box {background: rgba(255,255,255,0.9); padding: 20px; border-left: 5px solid #0288d1; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
.reco-box ul {list-style-type: none; padding-left: 0;}
.reco-box li::before {content: "✔️"; margin-right: 8px;}
.metric-card {background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Utilitaire pour trouver un .joblib
def find_joblib_file(prefix):
    models_dir = 'models'
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.lower().startswith(prefix.lower()) and f.endswith('.joblib'):
                return os.path.join(models_dir, f)
    return None

# Titre principal
st.markdown("<h1 class='custom-title'>📈 Moroccan Telecom Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #01579b; font-size: 18px;'>Powered by LightGBM with Optuna Optimization | Optimal Threshold: 0.30</p>", unsafe_allow_html=True)
st.markdown("---")

# Chargement des données
@st.cache_data
def load_data(path='data/synthetic_moroccan_churn_1M.csv'):
    if not os.path.exists(path):
        st.error(f"Fichier introuvable: {path}")
        st.stop()
    return pd.read_csv(path)

# Chargement des modèles et artefacts
@st.cache_resource
def load_models():
    """Charge tous les artefacts nécessaires"""
    try:
        # Charger le modèle optimisé en priorité
        model_path = 'models/model_lightgbm_tuned_churn.joblib'
        if not os.path.exists(model_path):
            model_path = find_joblib_file('model_lightgbm')
            if not model_path:
                st.error("Aucun modèle LightGBM trouvé dans models/")
                st.stop()
        model = joblib.load(model_path)
        
        # Charger les artefacts de preprocessing
        encoder_path = find_joblib_file('encoder')
        scaler_path = find_joblib_file('scaler_churn')
        features_path = find_joblib_file('features')
        categorical_path = find_joblib_file('categorical_columns')
        numerical_path = find_joblib_file('numerical_columns')
        scaler_features_path = find_joblib_file('scaler_features')
        
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        categorical_cols = joblib.load(categorical_path)
        numerical_cols = joblib.load(numerical_path)
        scaler_features = joblib.load(scaler_features_path)
        
        # Charger le seuil optimal (si disponible)
        optimal_threshold_path = find_joblib_file('optimal_threshold')
        optimal_threshold = joblib.load(optimal_threshold_path) if optimal_threshold_path else 0.5
        
        return model, encoder, scaler, features, categorical_cols, numerical_cols, scaler_features, optimal_threshold
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

def preprocess_data(data, encoder, scaler, features, categorical_cols, numerical_cols, scaler_features):
    """Prétraite les données selon le pipeline du notebook"""
    # Séparer les features
    X = data.drop(columns=['churn'], errors='ignore').copy()
    
    # Feature engineering - créer TOUTES les features engineered
    # 1. Ratio features
    if 'tenure_income_ratio' not in X.columns:
        X['tenure_income_ratio'] = X['anciennete'] / (X['revenu'] + 1)
    if 'age_tenure_ratio' not in X.columns:
        X['age_tenure_ratio'] = X['age'] / (X['anciennete'] + 1)
    
    # 2. Statistical features
    if 'revenue_zscore' not in X.columns:
        X['revenue_zscore'] = (X['revenu'] - X['revenu'].mean()) / X['revenu'].std()
    if 'age_percentile' not in X.columns:
        X['age_percentile'] = X['age'].rank(pct=True)
    if 'tenure_percentile' not in X.columns:
        X['tenure_percentile'] = X['anciennete'].rank(pct=True)
    
    # 3. Loyalty score
    if 'loyalty_score' not in X.columns:
        X['loyalty_score'] = (
            (X['anciennete'] >= 5).astype(int) * 3 +
            (X['contrat'] == 'Forfait Illimité').astype(int) * 2 +
            (X['internet_service'] == 'Fibre').astype(int) * 2 +
            (X['type_appareil'] == 'iPhone').astype(int) * 1 +
            (X['methode_paiement'].isin(['Carte bancaire', 'PayPal'])).astype(int) * 1
        )
    
    # 4. Categorical binning features
    if 'age_group' not in X.columns:
        X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    if 'revenue_tier' not in X.columns:
        X['revenue_tier'] = pd.cut(X['revenu'], bins=[0, 3000, 6000, 10000, 50000], labels=['Low', 'Medium', 'High', 'VIP'])
    if 'tenure_category' not in X.columns:
        X['tenure_category'] = pd.cut(X['anciennete'], bins=[-1, 1, 3, 5, 50], labels=['New', 'Regular', 'Loyal', 'VeryLoyal'])
    
    # Encoder les variables catégorielles
    try:
        encoded = encoder.transform(X[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    except Exception as e:
        st.error(f"Erreur d'encodage: {e}")
        encoded_df = pd.get_dummies(X[categorical_cols], drop_first=False)
    
    # Combiner numerical + encoded
    numerical_df = X[numerical_cols]
    X_all = pd.concat([numerical_df, encoded_df], axis=1)
    
    # Réordonner selon les features attendues
    X_all = X_all.reindex(columns=features, fill_value=0)
    
    # Standardiser seulement les features numériques
    X_final = X_all.copy()
    if scaler_features and len(scaler_features) > 0:
        scaler_cols = [col for col in scaler_features if col in X_final.columns]
        if len(scaler_cols) > 0:
            X_final[scaler_cols] = scaler.transform(X_all[scaler_cols])
    
    return X_final

# Sidebar: import et aide
st.sidebar.markdown("## 📂 Importer les données CSV")
uploader = st.sidebar.file_uploader("Upload un .csv (optionnel)", type=["csv"])
if uploader:
    try:
        data = pd.read_csv(uploader)
        st.sidebar.success("✅ Données personnalisées chargées.")
    except Exception as e:
        st.sidebar.error(f"Erreur lecture: {e}")
        data = load_data()
else:
    data = load_data()

# Charger les modèles
model, encoder, scaler, features, categorical_cols, numerical_cols, scaler_features, optimal_threshold = load_models()

# Aide utilisateur
with st.sidebar.expander("❓ Aide", expanded=False):
    st.markdown(
        "**Comment utiliser ce dashboard:**\n\n"
        "- 📊 **Distribution du churn**: Vue d'ensemble de la répartition\n"
        "- 📈 **Importances**: Features les plus influentes\n"
        "- 🎯 **Performance**: Courbes ROC et Precision-Recall\n"
        "- ⚠️ **Clients à risques**: Identification et recommandations\n\n"
        "💡 **Astuce**: Ajustez le seuil de risque pour cibler différents segments"
    )

# Préprocessing pour métriques
X_final = None
y_true = None
y_probs = None
y_pred = None
acc = prec = rec = f1 = auc = None

try:
    X_final = preprocess_data(data, encoder, scaler, features, categorical_cols, numerical_cols, scaler_features)
    y_true = data['churn']
    y_probs = model.predict_proba(X_final)[:, 1]
    
    # Prédictions avec seuil optimal
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    # Prédictions avec seuil par défaut (pour comparaison)
    y_pred_default = model.predict(X_final)
    
    # Calculer les métriques avec seuil optimal
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    
    # Métriques avec seuil par défaut (pour comparaison)
    acc_default = accuracy_score(y_true, y_pred_default)
    prec_default = precision_score(y_true, y_pred_default, zero_division=0)
    rec_default = recall_score(y_true, y_pred_default, zero_division=0)
    f1_default = f1_score(y_true, y_pred_default, zero_division=0)
except Exception as e:
    st.error(f"Erreur lors du prétraitement: {e}")
    y_true = data.get('churn', pd.Series([0]))  # Fallback

# KPI en en-tête
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("👥 Total clients", f"{len(data):,}")
with col2:
    st.metric("⚠️ Taux de churn", f"{y_true.mean():.2%}")
if acc is not None:
    with col3:
        st.metric("🎯 Accuracy", f"{acc:.2%}")
    with col4:
        st.metric("📊 ROC-AUC", f"{auc:.3f}")
    with col5:
        st.metric("🔍 Recall", f"{rec:.2%}")

# Threshold comparison banner
if acc is not None:
    st.markdown("---")
    st.markdown(f"### 🎯 Seuil de décision optimal: **{optimal_threshold:.3f}** (au lieu de 0.5)")
    
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        st.markdown("**📊 Avec seuil optimal ({:.3f})**".format(optimal_threshold))
        st.markdown(f"- Accuracy: **{acc:.2%}**")
        st.markdown(f"- Precision: **{prec:.2%}**")
        st.markdown(f"- Recall: **{rec:.2%}**")
        st.markdown(f"- F1-Score: **{f1:.3f}**")
    
    with col_comp2:
        st.markdown("**📉 Avec seuil par défaut (0.5)**")
        st.markdown(f"- Accuracy: {acc_default:.2%}")
        st.markdown(f"- Precision: {prec_default:.2%}")
        st.markdown(f"- Recall: {rec_default:.2%}")
        st.markdown(f"- F1-Score: {f1_default:.3f}")
    
    with col_comp3:
        st.markdown("**💰 Impact Business**")
        st.markdown(f"- ✅ Recall: **{(rec - rec_default)*100:+.1f}%**")
        st.markdown(f"- 💵 Plus de churners identifiés")
        st.markdown(f"- 📈 Maximise le profit")
        st.markdown("- 🎯 Optimisé par analyse coûts")

st.markdown("---")

# Sections
st.sidebar.markdown("## 🔍 Navigation")
section = st.sidebar.radio(
    "Choisissez une section:",
    ["📊 Distribution du churn", "🎯 Feature Importance", "📈 Performance du modèle", "📊 Analyse Lift", "🔥 Heatmap corrélation", "⚠️ Clients à risques"],
    label_visibility="collapsed"
)

# 1) Distribution du churn
if section == "📊 Distribution du churn":
    st.subheader("🎯 Répartition churn vs non-churn")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cnt = data['churn'].value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=['Non-churn', 'Churn'], 
            values=cnt.values,
            hole=0.4, 
            marker=dict(colors=['#0288d1', '#d81b60'], line=dict(color='#FFF', width=2)), 
            pull=[0.05, 0]
        ))
        fig.update_layout(
            height=400, 
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 📈 Statistiques clés")
        st.markdown(f"""
        <div class='reco-box'>
            <h4>Répartition des clients:</h4>
            <ul>
                <li><b>Non-churn:</b> {cnt[0]:,} clients ({cnt[0]/len(data):.1%})</li>
                <li><b>Churn:</b> {cnt[1]:,} clients ({cnt[1]/len(data):.1%})</li>
            </ul>
            <br>
            <p>📌 Le taux de churn de <b>{y_true.mean():.1%}</b> indique qu'environ <b>{int(cnt[1]/1000)}K</b> clients sont à risque de résiliation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribution par contrat
    st.markdown("---")
    st.subheader("📊 Analyse par type de contrat")
    churn_by_contract = data.groupby('contrat')['churn'].agg(['mean', 'count']).reset_index()
    fig2 = px.bar(churn_by_contract, x='contrat', y='mean', 
                  title='Taux de churn par type de contrat',
                  labels={'mean': 'Taux de churn', 'contrat': 'Type de contrat'},
                  color='mean', color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig2, width='stretch')

# 2) Feature Importance
elif section == "🎯 Feature Importance":
    st.subheader("📊 Variables les plus influentes (LightGBM)")
    
    imp = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': features[:len(imp)], 
        'Importance': imp
    }).sort_values('Importance', ascending=False).head(20)
    
    fig = go.Figure(go.Bar(
        x=df_imp['Importance'], 
        y=df_imp['Feature'], 
        orientation='h',
        marker=dict(
            color=df_imp['Importance'], 
            colorscale='Viridis', 
            showscale=True, 
            line=dict(color='#FFF', width=1)
        )
    ))
    fig.update_layout(
        title='Top 20 Features les plus importantes',
        xaxis_title='Importance', 
        yaxis_title='Feature',
        template='plotly_white', 
        height=600, 
        margin=dict(l=200, r=20, t=50, b=50)
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("""
    <div class='reco-box'>
        <h4>💡 Interprétation:</h4>
        <p>Les features avec la plus haute importance sont celles qui influencent le plus les prédictions du modèle.</p>
        <ul>
            <li><b>Features numériques:</b> âge, revenu, ancienneté</li>
            <li><b>Features engineered:</b> loyalty_score, ratios, indicateurs de risque</li>
            <li><b>Features catégorielles:</b> type de contrat, service internet</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 3) Performance modèle
elif section == "📈 Performance du modèle":
    st.subheader("📈 Courbes de performance du modèle")
    
    if acc is None:
        st.warning("⚠️ Impossible de calculer les métriques de performance.")
    else:
        # Métriques détaillées
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy", f"{acc:.2%}")
        col2.metric("🔍 Precision", f"{prec:.2%}")
        col3.metric("📊 Recall", f"{rec:.2%}")
        col4.metric("⚖️ F1-Score", f"{f1:.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', 
                name=f'AUC={auc:.3f}', 
                line=dict(color='#d81b60', width=3)
            ))
            fig1.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', 
                name='Baseline', 
                line=dict(dash='dash', color='gray')
            ))
            fig1.update_layout(
                title='ROC Curve', 
                xaxis_title='False Positive Rate', 
                yaxis_title='True Positive Rate',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=recall, y=precision, mode='lines', 
                name='Precision-Recall', 
                line=dict(color='#0288d1', width=3),
                fill='tozeroy'
            ))
            fig2.update_layout(
                title='Precision-Recall Curve', 
                xaxis_title='Recall', 
                yaxis_title='Precision',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig2, width='stretch')
        
        st.markdown("""
        <div class='reco-box'>
            <h4>📌 Analyse des performances:</h4>
            <ul>
                <li><b>ROC-AUC: 0.726</b> - Bonne capacité de discrimination</li>
                <li><b>Recall: 62%</b> - Le modèle détecte 62% des churners</li>
                <li><b>Precision: 76%</b> - 76% des prédictions de churn sont correctes</li>
                <li><b>Optimisé avec Optuna</b> - 50 trials d'optimisation bayésienne</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# 4) Analyse Lift
elif section == "📊 Analyse Lift":
    st.subheader("📊 Lift Analysis - Model Performance vs Random Targeting")
    
    st.info("""
    **Lift** measures how much better the model performs compared to random targeting.  
    - **Lift = 1.0**: Model no better than random  
    - **Lift = 2.0**: Model is 2x better than random  
    - **Lift = 3.0**: Model is 3x better (excellent)
    """)
    
    # Calculate lift metrics
    df_lift = pd.DataFrame({
        'y_true': y_true,
        'y_prob': y_probs
    }).sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lift_results = []
    baseline_churn_rate = y_true.mean()
    
    for pct in percentiles:
        n_customers = int(len(df_lift) * pct / 100)
        top_n = df_lift.head(n_customers)
        actual_churn_rate = top_n['y_true'].mean()
        lift = actual_churn_rate / baseline_churn_rate if baseline_churn_rate > 0 else 0
        churners_captured = top_n['y_true'].sum()
        total_churners = y_true.sum()
        capture_rate = churners_captured / total_churners if total_churners > 0 else 0
        
        lift_results.append({
            'percentile': pct,
            'lift': lift,
            'churn_rate': actual_churn_rate,
            'capture_rate': capture_rate * 100
        })
    
    lift_df = pd.DataFrame(lift_results)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    max_lift = lift_df['lift'].max()
    max_lift_pct = lift_df[lift_df['lift'] == max_lift]['percentile'].values[0]
    top20_lift = lift_df[lift_df['percentile'] == 20]['lift'].values[0]
    top20_capture = lift_df[lift_df['percentile'] == 20]['capture_rate'].values[0]
    
    with col1:
        st.metric("🎯 Max Lift", f"{max_lift:.2f}x", f"at top {max_lift_pct}%")
    with col2:
        st.metric("📈 Top 20% Lift", f"{top20_lift:.2f}x", f"{(top20_lift-1)*100:.0f}% vs random")
    with col3:
        st.metric("✅ Top 20% Captures", f"{top20_capture:.1f}%", "of all churners")
    
    st.markdown("---")
    
    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📈 Lift Curve")
        fig_lift = go.Figure()
        fig_lift.add_trace(go.Scatter(
            x=lift_df['percentile'], y=lift_df['lift'],
            mode='lines+markers', name='Model Lift',
            line=dict(color='#0288d1', width=3),
            marker=dict(size=8)
        ))
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="red", 
                          annotation_text="Random (Lift=1.0)")
        fig_lift.update_layout(
            xaxis_title="Top N% of Customers Targeted",
            yaxis_title="Lift",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_lift, width='stretch')
    
    with col_chart2:
        st.markdown("### 📊 Cumulative Gains")
        fig_gains = go.Figure()
        fig_gains.add_trace(go.Scatter(
            x=lift_df['percentile'], y=lift_df['capture_rate'],
            mode='lines+markers', name='Model',
            line=dict(color='#00897b', width=3),
            marker=dict(size=8)
        ))
        fig_gains.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode='lines', name='Random',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig_gains.update_layout(
            xaxis_title="% of Customers Targeted",
            yaxis_title="% of Churners Captured",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_gains, width='stretch')
    
    st.markdown("---")
    
    # Detailed table
    st.markdown("### 📋 Lift Performance by Segment")
    lift_display = lift_df.copy()
    lift_display['churn_rate'] = lift_display['churn_rate'].apply(lambda x: f"{x:.2%}")
    lift_display['lift'] = lift_display['lift'].apply(lambda x: f"{x:.2f}x")
    lift_display['capture_rate'] = lift_display['capture_rate'].apply(lambda x: f"{x:.1f}%")
    lift_display.columns = ['Top N%', 'Lift', 'Churn Rate', '% Churners Captured']
    st.dataframe(lift_display, width='stretch')
    
    # Business insights
    st.markdown("---")
    st.markdown("### 💡 Business Insights")
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.success(f"""
        **Model Effectiveness**
        - Targeting top 20% is **{top20_lift:.1f}x better** than random
        - Captures **{top20_capture:.0f}%** of churners with only 20% effort
        - Efficiency gain: **{(top20_lift-1)*100:.0f}%** over random targeting
        """)
    
    with col_insight2:
        st.info(f"""
        **Recommendation**
        - Focus retention campaigns on top **{max_lift_pct}%** for maximum lift ({max_lift:.2f}x)
        - Balance coverage vs efficiency based on budget
        - Top 30% captures ~{lift_df[lift_df['percentile']==30]['capture_rate'].values[0]:.0f}% of churners
        """)

# 5) Heatmap corrélation
elif section == "🔥 Heatmap corrélation":
    st.subheader("🔥 Heatmap de corrélation")
    
    # Sélectionner uniquement les colonnes numériques
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limiter à 15 colonnes pour la lisibilité
    if len(num_cols) > 15:
        # Prendre les colonnes les plus corrélées avec churn
        corr_with_churn = data[num_cols].corr()['churn'].abs().sort_values(ascending=False)
        num_cols = corr_with_churn.head(15).index.tolist()
    
    corr = data[num_cols].corr()
    
    fig = px.imshow(
        corr, 
        text_auto='.2f', 
        aspect="auto", 
        color_continuous_scale='RdBu_r',
        title='Matrice de corrélation (Top 15 features)'
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("""
    <div class='reco-box'>
        <h4>💡 Comment lire cette heatmap:</h4>
        <ul>
            <li><b>Rouge foncé:</b> Corrélation positive forte</li>
            <li><b>Bleu foncé:</b> Corrélation négative forte</li>
            <li><b>Blanc:</b> Pas de corrélation</li>
        </ul>
        <p>Les valeurs proches de <b>±1</b> indiquent une forte corrélation entre les variables.</p>
    </div>
    """, unsafe_allow_html=True)

# 6) Clients à risques
else:
    st.subheader("⚠️ Identification des Clients à Haut Risque")
    
    # Ajouter les scores de risque
    data['risk_score'] = y_probs
    
    # Afficher le seuil optimal
    st.info(f"🎯 **Seuil optimal identifié**: {optimal_threshold:.3f} (maximise le profit business)")
    
    # Slider pour le seuil
    col1, col2 = st.columns([3, 1])
    with col1:
        threshold = st.slider('🎚️ Seuil de risque (ajuster selon la stratégie)', 0.0, 1.0, float(optimal_threshold), 0.01)
    with col2:
        st.metric("Clients ciblés", f"{(data['risk_score'] >= threshold).sum():,}")
    
    # Distribution des scores
    st.markdown("---")
    st.subheader("📊 Distribution des scores de risque")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=data['risk_score'], 
        nbinsx=50,
        marker=dict(color='#0288d1', line=dict(color='white', width=1)),
        name='Distribution'
    ))
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                       annotation_text=f"Seuil: {threshold:.2f}")
    fig_hist.update_layout(
        title='Histogramme des scores de risque',
        xaxis_title='Score de risque',
        yaxis_title='Nombre de clients',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_hist, width='stretch')
    
    # Filtrer les clients à risque
    high_risk = data[data['risk_score'] >= threshold].sort_values('risk_score', ascending=False)
    
    st.markdown("---")
    st.subheader(f"🎯 Top 20 clients à risque (seuil ≥ {threshold:.2f})")
    
    # Afficher le tableau avec colonnes pertinentes
    display_cols = ['risk_score', 'contrat', 'internet_service', 'anciennete', 'revenu', 'age', 'churn']
    display_cols = [col for col in display_cols if col in high_risk.columns]
    
    st.dataframe(
        high_risk[display_cols].head(20).style.background_gradient(subset=['risk_score'], cmap='Reds'),
        width='stretch'
    )
    
    # Recommandations stratégiques
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='reco-box'>
            <h4>💡 Recommandations Actions:</h4>
            <ul>
                <li>🎯 <b>Campagne de rétention ciblée</b> sur les segments identifiés</li>
                <li>📧 <b>Emails personnalisés</b> avec offres adaptées au profil</li>
                <li>📞 <b>Appels proactifs</b> pour les clients VIP (revenu élevé)</li>
                <li>🎁 <b>Offres spéciales</b> pour les clients à forte ancienneté</li>
                <li>📊 <b>Suivi temps réel</b> et ajustement du seuil selon ROI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Analyse par segment
        st.markdown("### 📊 Analyse par segment")
        segment_analysis = high_risk.groupby('contrat').agg({
            'risk_score': 'mean',
            'churn': 'count'
        }).round(3)
        segment_analysis.columns = ['Score moyen', 'Nombre']
        st.dataframe(segment_analysis, width='stretch')
    
    # Export CSV
    st.markdown("---")
    csv = high_risk[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger la liste des clients à risque (CSV)",
        data=csv,
        file_name=f'clients_a_risque_seuil_{threshold:.2f}.csv',
        mime='text/csv',
    )

# Informations sur le modèle en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ️ Informations Modèle")
if acc is not None:
    st.sidebar.markdown(f"""
- **Modèle**: LightGBM Optimized
- **ROC-AUC**: {auc:.3f}
- **Features**: {len(features)}
- **Recall**: {rec:.1%}
- **F1-Score**: {f1:.3f}
- **Optimization**: Optuna (50 trials)
""")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "© 2025 Moroccan Telecom Churn Detection | Developed by El Mehdi El Youbi Rmich | "
    "<a href='https://github.com/elmehdi03/churn_prediction_dashboard' target='_blank'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)

