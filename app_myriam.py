import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ─── Configuration de la page ───────────────────────────────
st.set_page_config(
    page_title="Segmentation Clients",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS — design vif et coloré ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(160deg, #fdf4ff 0%, #fff1f2 50%, #fffbeb 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #2d1b4e;
    }

    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #c026d3, #f43f5e, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        text-align: center;
        color: #9333ea;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .card {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 8px 30px rgba(192, 38, 211, 0.10);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(244, 63, 94, 0.18);
    }
    .card-emoji { font-size: 2rem; }
    .card-num {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #c026d3, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .card-title { font-weight: 700; font-size: 1.1rem; margin-top: 0.3rem; }
    .card-desc { color: #8b7aa8; font-size: 0.88rem; margin-top: 0.2rem; }

    .segment-box {
        background: linear-gradient(135deg, #c026d3, #f43f5e);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: white !important;
        box-shadow: 0 10px 30px rgba(244, 63, 94, 0.35);
        animation: pop 0.4s ease;
    }
    @keyframes pop { from {opacity:0; transform:scale(0.9);} to {opacity:1; transform:scale(1);} }

    h3 { color: #9333ea !important; font-weight: 700 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #c026d3, #f43f5e);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        width: 100%;
        box-shadow: 0 6px 18px rgba(244, 63, 94, 0.3);
        transition: all 0.25s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(244, 63, 94, 0.45); }

    [data-testid="stNumberInput"] input {
        border-radius: 12px;
        border: 1.5px solid #f5d0fe;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement modèle + données ─────────────────────────────
@st.cache_resource
def charger_modele():
    return joblib.load("model/modele_kmeans.pkl"), joblib.load("model/normaliseur.pkl")

@st.cache_data
def charger_donnees():
    return pd.read_csv("rfm_data.csv")

modele, normaliseur = charger_modele()
rfm = charger_donnees()

noms_segments = {
    0: "💜 Clients Réguliers",
    1: "🧡 Clients à Risque",
    2: "👑 Clients VIP",
    3: "✨ Clients Premium"
}

# ─── NAVBAR HORIZONTALE EN HAUT ──────────────────────────────
page = option_menu(
    menu_title=None,
    options=["Accueil", "Prédiction", "Tableau de bord", "À propos"],
    icons=["house-fill", "magic", "bar-chart-fill", "info-circle-fill"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "8px", "background": "#ffffff",
                      "border-radius": "16px", "box-shadow": "0 4px 20px rgba(192,38,211,0.10)",
                      "margin-bottom": "1rem"},
        "icon": {"color": "#c026d3", "font-size": "16px"},
        "nav-link": {"font-size": "15px", "font-weight": "600", "color": "#6b21a8",
                     "border-radius": "12px", "margin": "0 4px",
                     "--hover-color": "#fce7f3"},
        "nav-link-selected": {"background": "linear-gradient(135deg, #c026d3, #f43f5e)",
                              "color": "white"},
    }
)

COULEURS = ["#c026d3", "#f43f5e", "#f59e0b", "#8b5cf6"]

# ─── PAGE ACCUEIL ────────────────────────────────────────────
if page == "Accueil":
    st.markdown("<div class='hero-title'>Segmentation de la Clientèle</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Comprendre ses clients grâce au RFM et au clustering K-Means</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🎯 Objectif")
        st.write("Regrouper les clients d'un site e-commerce selon leur comportement d'achat, pour adapter les actions marketing à chaque profil.")
        st.markdown("### 📊 La méthode RFM")
        st.write("**Récence** : jours depuis le dernier achat. **Fréquence** : nombre de commandes. **Montant** : total dépensé.")
    with col2:
        st.markdown("### 🤖 Le modèle")
        st.write("Algorithme **K-Means**, 4 segments, validés par la méthode du coude et le score de silhouette.")
        st.markdown("### 📋 Utilisation")
        st.write("Ouvrez l'onglet **Prédiction**, renseignez les trois valeurs d'un client et découvrez son segment.")

    st.markdown("### 🗂️ Les 4 segments")
    cols = st.columns(4, gap="large")
    infos = [("💜", "Réguliers", "Clients actifs courants"),
             ("🧡", "À Risque", "Clients qui s'éloignent"),
             ("👑", "VIP", "Grands comptes précieux"),
             ("✨", "Premium", "Meilleurs clients fidèles")]
    for col, (e, nom, desc) in zip(cols, infos):
        with col:
            st.markdown(f"<div class='card'><div class='card-emoji'>{e}</div><div class='card-title'>{nom}</div><div class='card-desc'>{desc}</div></div>", unsafe_allow_html=True)

# ─── PAGE PRÉDICTION ─────────────────────────────────────────
if page == "Prédiction":
    st.markdown("<div class='hero-title'>Prédire le Segment</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Saisissez les valeurs d'un client</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 📝 Données du client")
        recence = st.number_input("Récence — jours depuis le dernier achat", min_value=0, max_value=500, value=30)
        frequence = st.number_input("Fréquence — nombre de commandes", min_value=1, max_value=300, value=5)
        montant = st.number_input("Montant — total dépensé (€)", min_value=0.0, max_value=300000.0, value=1000.0)
        bouton = st.button("🔍 Identifier le segment")
    with col2:
        st.markdown("### 📊 Résultat")
        if bouton:
            valeurs = np.array([[recence, frequence, montant]])
            seg = modele.predict(normaliseur.transform(valeurs))[0]
            st.markdown(f"<div class='segment-box'>{noms_segments[seg]}</div>", unsafe_allow_html=True)
            st.write("")
            descriptions = {
                0: "Client actif au comportement standard. Achète régulièrement avec un budget modéré.",
                1: "Client en perte de vitesse, sans achat depuis longtemps. À relancer avec des offres ciblées.",
                2: "Client exceptionnel à très forte valeur, probablement un revendeur. À suivre en priorité.",
                3: "Excellent client fidèle, récent et bon panier. Un pilier de l'entreprise."
            }
            st.info(descriptions[seg])
        else:
            st.info("👈 Renseignez les valeurs puis cliquez sur le bouton")

# ─── PAGE TABLEAU DE BORD ────────────────────────────────────
if page == "Tableau de bord":
    st.markdown("<div class='hero-title'>Tableau de Bord</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Vue d'ensemble des segments</div>", unsafe_allow_html=True)

    effectifs = rfm["Segment"].value_counts().sort_index()
    profils = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean()

    cols = st.columns(4, gap="large")
    for i in range(4):
        with cols[i]:
            st.markdown(f"<div class='card'><div class='card-num'>{effectifs[i]}</div><div class='card-desc'>{noms_segments[i]}</div></div>", unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🍩 Répartition des segments")
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_alpha(0)
        labels = ["Réguliers", "À Risque", "VIP", "Premium"]
        ax.pie(effectifs, labels=labels, colors=COULEURS, autopct="%1.1f%%",
               startangle=90, pctdistance=0.8, labeldistance=1.08,
               wedgeprops=dict(width=0.45, edgecolor="white", linewidth=3),
               textprops={"color": "#2d1b4e"})
        st.pyplot(fig)
    with col2:
        st.markdown("### 📊 Profil moyen par segment")
        profil_mm = (profils - profils.min()) / (profils.max() - profils.min())
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")
        x = np.arange(4); larg = 0.25
        ax2.bar(x - larg, profil_mm["Recency"], larg, label="Récence", color="#c026d3")
        ax2.bar(x, profil_mm["Frequency"], larg, label="Fréquence", color="#f43f5e")
        ax2.bar(x + larg, profil_mm["Monetary"], larg, label="Montant", color="#f59e0b")
        ax2.set_xticks(x); ax2.set_xticklabels(["S0", "S1", "S2", "S3"])
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        ax2.legend(frameon=False)
        st.pyplot(fig2)

    st.markdown("### 📋 Profil détaillé")
    recap = profils.copy()
    recap["Effectif"] = effectifs
    recap["Segment"] = [noms_segments[i] for i in range(4)]
    recap = recap.round(1)[["Segment", "Recency", "Frequency", "Monetary", "Effectif"]]
    st.dataframe(recap, use_container_width=True, hide_index=True)

# ─── PAGE À PROPOS ───────────────────────────────────────────
if page == "À propos":
    st.markdown("<div class='hero-title'>À propos du projet</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Contexte et informations</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 🎯 Le projet")
        st.write("Réalisé dans le cadre du cours Framework Machine Learning — Apprentissage non supervisé. L'application segmente les clients selon leur comportement d'achat pour orienter les actions marketing.")
        st.markdown("### 🔬 La démarche")
        st.write("Nettoyage, calcul des variables RFM, normalisation, clustering K-Means à 4 segments, validation par coude et silhouette.")
    with col2:
        st.markdown("### 🛠️ Technologies")
        st.write("Python, Pandas, Scikit-learn, Matplotlib, Streamlit.")
        st.markdown("### 📊 Le dataset")
        st.write("Online Retail (UK). 541 909 transactions au départ, 401 604 après nettoyage, 4 338 clients, période 2010–2011.")

    st.markdown("<div style='text-align:center; color:#8b7aa8; padding-top:1rem;'>Développé par <b>Deffo Myriam</b> — KEYCE Informatique — IABD B3 — 2025/2026</div>", unsafe_allow_html=True)
