import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ─── Configuration de la page ───────────────────────────────
st.set_page_config(
    page_title="Segmentation Clients",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Personnalisé (thème bleu océan) ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #eff6ff 0%, #ecfeff 100%);
        color: #0c4a6e;
        font-family: 'Nunito', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0369a1 0%, #0891b2 100%);
        border: none;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0891b2, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        letter-spacing: -1px;
    }
    .stat-card {
        background: white;
        border: 1px solid #bae6fd;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(8, 145, 178, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 30px rgba(8, 145, 178, 0.18);
    }
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0891b2;
    }
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .segment-box {
        background: linear-gradient(135deg, #0891b2, #06b6d4);
        border: none;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 800;
        color: white !important;
        box-shadow: 0 8px 25px rgba(8, 145, 178, 0.35);
        animation: popIn 0.4s ease;
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to   { opacity: 1; transform: scale(1); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #0891b2, #06b6d4);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(8, 145, 178, 0.3);
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(8, 145, 178, 0.45);
    }
    [data-testid="stNumberInput"] input {
        border-radius: 10px;
        border: 1.5px solid #bae6fd;
    }
    h3 { color: #075985 !important; font-weight: 700 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement du modèle et des données ─────────────────────
@st.cache_resource
def charger_modele():
    modele = joblib.load("model/modele_kmeans.pkl")
    normaliseur = joblib.load("model/normaliseur.pkl")
    return modele, normaliseur

@st.cache_data
def charger_donnees():
    return pd.read_csv("rfm_data.csv")

modele, normaliseur = charger_modele()
rfm = charger_donnees()

# ─── Noms des segments ───────────────────────────────────────
noms_segments = {
    0: " Clients Réguliers",
    1: " Clients à Risque",
    2: " Clients VIP",
    3: " Clients Premium"
}

# ─── Barre latérale ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Segmentation Clients")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🔮 Prédiction", "📊 Tableau de bord", "ℹ️ À propos"],
        label_visibility="hidden"
    )
    st.markdown("---")
    st.markdown("### 📈 Aperçu")
    st.metric("Clients analysés", len(rfm))
    st.metric("Segments", 4)

# ─── PAGE ACCUEIL ────────────────────────────────────────────
if page == "🏠 Accueil":
    st.markdown('<p class="main-title">🌊 Segmentation de la Clientèle</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Objectif")
        st.markdown("""
        Comprendre le comportement d'achat des clients d'un site
        e-commerce et les regrouper en segments cohérents grâce à
        la méthode **RFM** et au clustering **K-Means**.
        """)
        st.markdown("### 📊 La méthode RFM")
        st.markdown("""
        - **R** (Récence) : jours depuis le dernier achat
        - **F** (Fréquence) : nombre de commandes
        - **M** (Montant) : total dépensé
        """)
    
        st.markdown("### 📋 Mode d'emploi")
        st.markdown("""
        1. Ouvrez **🔮 Prédiction**
        2. Renseignez R, F et M d'un client
        3. Découvrez son segment et son profil
        """)

    st.markdown("---")
    st.markdown("###  Les 4 segments")
    cols = st.columns(4)
    infos = [
        ("🌊", "Réguliers", "Clients actifs courants"),
        ("🌧️", "À Risque", "Clients qui s'éloignent"),
        ("👑", "VIP", "Grands comptes précieux"),
        ("⭐", "Premium", "Meilleurs clients fidèles")
    ]
    for col, (emoji, nom, desc) in zip(cols, infos):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{emoji}</div>
                <div class="stat-label"><b>{nom}</b><br>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── PAGE PRÉDICTION ─────────────────────────────────────────
if page == "🔮 Prédiction":
    st.markdown('<p class="main-title">🔮 Trouver le Segment d\'un Client</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Données du client")
        recence = st.number_input("Récence (jours depuis le dernier achat)", min_value=0, max_value=500, value=30)
        frequence = st.number_input("Fréquence (nombre de commandes)", min_value=1, max_value=300, value=5)
        montant = st.number_input("Montant total dépensé (€)", min_value=0.0, max_value=300000.0, value=1000.0)
        bouton = st.button("🔍 Identifier le segment")

    with col2:
        st.markdown("### 📊 Résultat")
        if bouton:
            valeurs = np.array([[recence, frequence, montant]])
            valeurs_norm = normaliseur.transform(valeurs)
            seg = modele.predict(valeurs_norm)[0]

            st.markdown(f'<div class="segment-box">{noms_segments[seg]}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            descriptions = {
                0: "Client actif au comportement standard. Achète régulièrement avec un budget modéré.",
                1: "Client en perte de vitesse. Sans achat depuis longtemps — à relancer avec des offres ciblées.",
                2: "Client exceptionnel à très forte valeur, probablement un revendeur. À suivre en priorité.",
                3: "Excellent client fidèle, récent et bon panier — un pilier de l'entreprise."
            }
            st.info(descriptions[seg])
        else:
            st.info(" Renseignez les valeurs puis cliquez sur le bouton")

# ─── PAGE TABLEAU DE BORD ────────────────────────────────────
if page == "📊 Tableau de bord":
    st.markdown('<p class="main-title">📊 Tableau de Bord</p>', unsafe_allow_html=True)
    st.markdown("---")

    effectifs = rfm["Segment"].value_counts().sort_index()
    profils = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()

    st.markdown("### Répartition des clients")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{effectifs[i]}</div>
                <div class="stat-label">{noms_segments[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###  Répartition des segments")
        fig, ax = plt.subplots(figsize=(5, 5))
        couleurs = ["#0891b2", "#ef4444", "#f59e0b", "#6366f1"]
        labels = ["Réguliers", "À Risque", "VIP", "Premium"]
        ax.pie(effectifs, labels=labels, colors=couleurs, autopct="%1.1f%%",
               startangle=90, pctdistance=0.85, labeldistance=1.1)
        st.pyplot(fig)

    with col2:
        st.markdown("### 📊 Profil moyen par segment")
        profil_mm = (profils - profils.min()) / (profils.max() - profils.min())
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        x = np.arange(4)
        larg = 0.25
        ax2.bar(x - larg, profil_mm["Recency"], larg, label="Récence", color="#ef4444")
        ax2.bar(x, profil_mm["Frequency"], larg, label="Fréquence", color="#0891b2")
        ax2.bar(x + larg, profil_mm["Monetary"], larg, label="Montant", color="#6366f1")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["S0", "S1", "S2", "S3"])
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("###  Profil détaillé des segments")
    recap = profils.copy()
    recap["Effectif"] = effectifs
    recap["Segment"] = [noms_segments[i] for i in range(4)]
    recap = recap.round(1)[["Segment", "Recency", "Frequency", "Monetary", "Effectif"]]
    st.dataframe(recap, use_container_width=True)

# ─── PAGE À PROPOS ───────────────────────────────────────────
if page == "ℹ️ À propos":
    st.markdown('<p class="main-title">ℹ️ À propos du projet</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Le projet")
        st.markdown("""
        Projet réalisé dans le cadre du cours **Framework Machine
        Learning — Apprentissage non supervisé**. Il segmente les
        clients d'un site e-commerce selon leur comportement d'achat
        pour orienter les actions marketing.
        """)
        st.markdown("###  La démarche")
        st.markdown("""
        1. Nettoyage des données
        2. Calcul des variables RFM
        3. Normalisation (StandardScaler)
        4. Clustering K-Means (K=4)
        5. Validation par coude et silhouette
        """)
    with col2:
        st.markdown("###  Technologies")
        st.markdown("""
        - **Python**
        - **Pandas**
        - **Scikit-learn**
        - **Matplotlib**
        - **Streamlit**
        """)
        st.markdown("### 📊 Le dataset")
        st.markdown("""
        - **Source** : Online Retail (e-commerce UK)
        - **Initial** : 541 909 transactions
        - **Après nettoyage** : 401 604 transactions
        - **Clients** : 4 338
        - **Période** : 2010 - 2011
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#6b7280; padding:1rem;">
        Développé par <b>Deffo Myriam</b> — KEYCE Informatique<br>
        IABD B3 — 2025/2026
    </div>
    """, unsafe_allow_html=True)
