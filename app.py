"""
app.py - Tableau de bord de detection de phishing (Jalon 3)

Lance avec :
    streamlit run app.py

Ce tableau de bord permet de :
  - Coller une URL et l'analyser en direct (extraction de features +
    prediction du modele entraine au Jalon 3)
  - Analyser plusieurs URLs d'un coup via un fichier CSV
  - Voir le detail des features extraites, avec des badges lisibles
  - Consulter un historique PERSISTANT des URLs deja analysees
    (stocke dans historique.db, survit a la fermeture du dashboard),
    avec repartition graphique et export CSV
"""

import sqlite3
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_extractor import PhishingFeatureExtractor
from email_alert import envoyer_alerte_phishing
from ip_reputation import resoudre_ip, verifier_ip


# --- Configuration de la page ---
st.set_page_config(
    page_title="Detecteur de Phishing - PME",
    page_icon="🛡️",
    layout="centered",
)

# --- Petits styles additionnels (au-dela de ce que le theme gere) ---
st.markdown("""
<style>
    .stat-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-ok { background-color: rgba(30,130,76,0.18); color: #4ade80; border: 1px solid #1e824c; }
    .badge-warn { background-color: rgba(200,140,20,0.18); color: #fbbf24; border: 1px solid #c88c14; }
    .badge-danger { background-color: rgba(190,30,45,0.18); color: #f87171; border: 1px solid #be1e2d; }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, #1a1f2b, #151922);
        border: 1px solid #2a3040;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #9aa4b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f5f5f5;
    }
    .kpi-sub {
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
    .kpi-sub-danger { color: #f87171; }
    .kpi-sub-ok { color: #4ade80; }
    .country-bar-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .country-bar-track {
        flex-grow: 1;
        background: #232838;
        border-radius: 6px;
        height: 10px;
        margin: 0 0.6rem;
        overflow: hidden;
    }
    .country-bar-fill {
        background: linear-gradient(90deg, #e11d48, #f87171);
        height: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- Chargement du modele et de l'imputer (mis en cache : ne se recharge
# pas a chaque interaction, seulement au premier lancement) ---
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    imputer = joblib.load("imputer.pkl")
    return model, imputer


@st.cache_resource
def load_extractor():
    return PhishingFeatureExtractor(timeout=6, rate_limit=0.0, cache_db="cache_dashboard.db")


# --- Historique persistant (SQLite local, survit a la fermeture du dashboard) ---
HISTORIQUE_DB = "historique.db"


def init_historique_db():
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historique (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                horodatage TEXT NOT NULL,
                url TEXT NOT NULL,
                resultat TEXT NOT NULL,
                confiance TEXT NOT NULL
            )
        """)
        # Migration douce : ajoute la colonne 'pays' si elle n'existe pas
        # deja (installations anterieures a cette fonctionnalite).
        try:
            conn.execute("ALTER TABLE historique ADD COLUMN pays TEXT")
        except sqlite3.OperationalError:
            pass  # la colonne existe deja
        conn.commit()


def ajouter_historique(url: str, resultat: str, confiance: str, pays: str = None):
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        conn.execute(
            "INSERT INTO historique (horodatage, url, resultat, confiance, pays) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), url, resultat, confiance, pays),
        )
        conn.commit()


def charger_historique(limite: int = 200) -> pd.DataFrame:
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT horodatage AS Horodatage, url AS URL, resultat AS Resultat, "
            "confiance AS Confiance FROM historique ORDER BY id DESC LIMIT ?",
            conn, params=(limite,),
        )
    return df


def charger_activite_par_jour(jours: int = 14) -> pd.DataFrame:
    """Compte le nombre d'analyses par jour, pour le graphique d'activite."""
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT substr(horodatage, 1, 10) AS jour, COUNT(*) AS nombre "
            "FROM historique GROUP BY jour ORDER BY jour DESC LIMIT ?",
            conn, params=(jours,),
        )
    return df.sort_values("jour").set_index("jour") if not df.empty else df


def charger_top_pays(limite: int = 5) -> pd.DataFrame:
    """Classement des pays les plus frequemment associes a une IP verifiee."""
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT pays, COUNT(*) AS nombre FROM historique "
            "WHERE pays IS NOT NULL AND pays != '' AND pays != '?' "
            "GROUP BY pays ORDER BY nombre DESC LIMIT ?",
            conn, params=(limite,),
        )
    return df


def vider_historique():
    with sqlite3.connect(HISTORIQUE_DB) as conn:
        conn.execute("DELETE FROM historique")
        conn.commit()


init_historique_db()


def build_feature_vector(raw_features: dict, feature_columns: list) -> pd.DataFrame:
    """
    Construit une ligne de donnees dans l'ORDRE EXACT attendu par le
    modele. On ajoute aussi 'PageUnreachable', calculee comme au moment
    de l'entrainement : 1 si la page n'a pas pu etre chargee (is_https
    est None), 0 sinon.
    """
    row = {col: raw_features.get(col) for col in feature_columns if col != "PageUnreachable"}
    row["PageUnreachable"] = 1 if raw_features.get("is_https") is None else 0

    df_row = pd.DataFrame([row])[feature_columns]

    df_row = df_row.apply(pd.to_numeric, errors="coerce")
    return df_row


def analyser_url(url: str, model, imputer, extractor, feature_columns: list):
    """Extrait les features d'une URL et renvoie la prediction du modele."""
    raw_features = extractor.extract(url)
    df_row = build_feature_vector(raw_features, feature_columns)

    df_row_imputed = pd.DataFrame(imputer.transform(df_row), columns=feature_columns)

    prediction = model.predict(df_row_imputed)[0]
    proba = model.predict_proba(df_row_imputed)[0]
    proba_phishing = proba[0]
    proba_legitime = proba[1]

    return {
        "prediction": "legitime" if prediction == 1 else "phishing",
        "proba_phishing": proba_phishing,
        "proba_legitime": proba_legitime,
        "raw_features": raw_features,
        "page_injoignable": raw_features.get("is_https") is None,
        "bot_challenge_detected": raw_features.get("_bot_challenge_detected", False),
    }


def categoriser_resultat(resultat: str) -> str:
    """Regroupe les statuts detailles en 4 grandes categories pour le graphique."""
    if "non concluant" in resultat:
        return "Non concluant"
    if "incertain" in resultat:
        return "Incertain"
    if resultat.startswith("phishing"):
        return "Phishing"
    if resultat.startswith("legitime"):
        return "Legitime"
    return "Autre"


def generer_badges(raw_features: dict, page_injoignable: bool) -> list:
    """
    Traduit les features brutes en badges lisibles pour un humain non
    technique -- plus parlant qu'un tableau de 30 valeurs numeriques.
    """
    badges = []

    if page_injoignable:
        badges.append(("❌ Page injoignable", "badge-danger"))
        return badges

    if raw_features.get("is_https") is True:
        badges.append(("✅ Connexion HTTPS", "badge-ok"))
    elif raw_features.get("is_https") is False:
        badges.append(("❌ Pas de HTTPS", "badge-danger"))

    if raw_features.get("has_ip_address"):
        badges.append(("⚠️ Adresse IP au lieu d'un nom de domaine", "badge-warn"))

    if raw_features.get("has_at_symbol"):
        badges.append(("⚠️ Symbole @ dans l'URL (camouflage possible)", "badge-warn"))

    if raw_features.get("has_favicon_mismatch"):
        badges.append(("⚠️ Favicon ne correspond pas au domaine", "badge-warn"))

    ext_forms = raw_features.get("external_form_count") or 0
    if ext_forms > 0:
        badges.append((f"⚠️ {int(ext_forms)} formulaire(s) vers un autre domaine", "badge-warn"))

    total_links = raw_features.get("total_links")
    if total_links is not None and total_links < 5:
        badges.append((f"⚠️ Tres peu de liens sur la page ({int(total_links)})", "badge-warn"))
    elif total_links is not None and total_links >= 20:
        badges.append((f"✅ Page riche en liens ({int(total_links)})", "badge-ok"))

    redirects = raw_features.get("redirect_count") or 0
    if redirects >= 2:
        badges.append((f"⚠️ {int(redirects)} redirections", "badge-warn"))

    if raw_features.get("has_security_headers") is True:
        badges.append(("✅ En-tetes de securite presents", "badge-ok"))

    return badges


def afficher_badges(badges: list):
    if not badges:
        return
    html = "".join(f'<span class="stat-badge {cls}">{texte}</span>' for texte, cls in badges)
    st.markdown(html, unsafe_allow_html=True)


def generer_badges_ip(resultat_ip: dict) -> list:
    """Traduit le resultat AbuseIPDB en badges lisibles."""
    badges = []
    score = resultat_ip.get("abuse_score", 0)

    if score >= 75:
        badges.append((f"🚨 Score d'abus tres eleve ({score}%)", "badge-danger"))
    elif score >= 25:
        badges.append((f"⚠️ Score d'abus moyen ({score}%)", "badge-warn"))
    else:
        badges.append((f"✅ Score d'abus faible ({score}%)", "badge-ok"))

    if resultat_ip.get("is_tor"):
        badges.append(("⚠️ Noeud de sortie Tor", "badge-warn"))

    if resultat_ip.get("is_whitelisted"):
        badges.append(("✅ IP whitelistee (infrastructure connue)", "badge-ok"))

    total_reports = resultat_ip.get("total_reports", 0)
    if total_reports > 0:
        badges.append((f"ℹ️ {total_reports} signalement(s) au total", "badge-warn" if score >= 25 else "badge-ok"))

    return badges


def afficher_camembert_css(repartition: dict):
    """
    Construit un camembert en pur CSS (conic-gradient), sans dependance
    externe (pas de matplotlib/plotly a installer).
    """
    couleurs = {
        "Phishing": "#f87171",
        "Legitime": "#4ade80",
        "Incertain": "#fbbf24",
        "Non concluant": "#94a3b8",
        "Autre": "#64748b",
    }
    total = sum(repartition.values())
    if total == 0:
        return

    segments = []
    angle_courant = 0
    for categorie, nombre in repartition.items():
        if nombre == 0:
            continue
        angle = (nombre / total) * 360
        couleur = couleurs.get(categorie, "#64748b")
        segments.append(f"{couleur} {angle_courant}deg {angle_courant + angle}deg")
        angle_courant += angle

    gradient = ", ".join(segments)

    col_camembert, col_legende = st.columns([1, 1])
    with col_camembert:
        st.markdown(
            f"""
            <div style="
                width: 160px; height: 160px; border-radius: 50%;
                background: conic-gradient({gradient});
                margin: 0.5rem auto;
            "></div>
            """,
            unsafe_allow_html=True,
        )
    with col_legende:
        for categorie, nombre in repartition.items():
            if nombre == 0:
                continue
            pct = nombre / total * 100
            couleur = couleurs.get(categorie, "#64748b")
            st.markdown(
                f'<span style="color:{couleur};">●</span> {categorie} : '
                f'**{nombre}** ({pct:.0f}\u00a0%)',
                unsafe_allow_html=True,
            )


def afficher_kpi_card(label: str, valeur, sous_texte: str = None, sous_classe: str = ""):
    sous_html = f'<div class="kpi-sub {sous_classe}">{sous_texte}</div>' if sous_texte else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{valeur}</div>
            {sous_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def afficher_donut_css(repartition: dict):
    """
    Version donut du camembert CSS (meme principe que afficher_camembert_css,
    avec un cercle central perce pour l'effet 'donut', plus proche des
    dashboards de securite classiques).
    """
    couleurs = {
        "Phishing": "#f87171",
        "Legitime": "#4ade80",
        "Incertain": "#fbbf24",
        "Non concluant": "#94a3b8",
        "Autre": "#64748b",
    }
    total = sum(repartition.values())
    if total == 0:
        st.caption("Pas encore de donnees.")
        return

    segments = []
    angle_courant = 0
    for categorie, nombre in repartition.items():
        if nombre == 0:
            continue
        angle = (nombre / total) * 360
        couleur = couleurs.get(categorie, "#64748b")
        segments.append(f"{couleur} {angle_courant}deg {angle_courant + angle}deg")
        angle_courant += angle
    gradient = ", ".join(segments)

    col_donut, col_legende = st.columns([1, 1])
    with col_donut:
        st.markdown(
            f"""
            <div style="
                width: 150px; height: 150px; border-radius: 50%;
                background: conic-gradient({gradient});
                margin: 0.5rem auto;
                display: flex; align-items: center; justify-content: center;
            ">
                <div style="
                    width: 82px; height: 82px; border-radius: 50%;
                    background: #0e1117;
                    display: flex; align-items: center; justify-content: center;
                    flex-direction: column;
                ">
                    <div style="font-size:1.3rem; font-weight:700; color:#f5f5f5;">{total}</div>
                    <div style="font-size:0.65rem; color:#9aa4b8;">total</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_legende:
        for categorie, nombre in repartition.items():
            if nombre == 0:
                continue
            pct = nombre / total * 100
            couleur = couleurs.get(categorie, "#64748b")
            st.markdown(
                f'<span style="color:{couleur};">●</span> {categorie} : '
                f'**{nombre}** ({pct:.0f}\u00a0%)',
                unsafe_allow_html=True,
            )


def afficher_top_pays(df_pays: pd.DataFrame):
    if df_pays.empty:
        st.caption("Aucune verification IP effectuee pour le moment.")
        return
    maximum = df_pays["nombre"].max()
    for _, ligne in df_pays.iterrows():
        largeur_pct = (ligne["nombre"] / maximum) * 100
        st.markdown(
            f"""
            <div class="country-bar-row">
                <span style="width:2.2rem; font-weight:600;">{ligne['pays']}</span>
                <div class="country-bar-track">
                    <div class="country-bar-fill" style="width:{largeur_pct}%;"></div>
                </div>
                <span>{ligne['nombre']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


model, imputer = load_model()
extractor = load_extractor()

FEATURE_COLUMNS = PhishingFeatureExtractor.ALL_FEATURES + ["PageUnreachable"]


# ============================== INTERFACE ==============================

st.title("🛡️ Detecteur de Phishing")
st.caption("Systeme de detection et d'alerte precoce des cybermenaces ciblant les PME")

with st.sidebar:
    st.markdown("### ⚙️ Alertes e-mail")
    alertes_activees = st.checkbox("Activer les alertes automatiques", value=False)
    email_destinataire = st.text_input(
        "E-mail du destinataire",
        placeholder="responsable-securite@pme-exemple.com",
        disabled=not alertes_activees,
    )
    st.caption(
        "Une alerte est envoyee automatiquement lorsqu'une URL est "
        "classee phishing avec une confiance suffisante (hors cas "
        "incertains ou non concluants)."
    )
    st.divider()
    st.markdown("### 📊 Statistiques")
    df_stats = charger_historique(limite=10_000)
    if not df_stats.empty:
        total = len(df_stats)
        n_phishing = df_stats["Resultat"].str.startswith("phishing").sum()
        n_legitime = df_stats["Resultat"].eq("legitime").sum()
        n_incertain = df_stats["Resultat"].str.contains("incertain|non concluant").sum()
        st.metric("URLs analysees", total)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🚨", n_phishing)
        col_b.metric("✅", n_legitime)
        col_c.metric("🤔", n_incertain)
    else:
        st.caption("Aucune donnee pour le moment.")

tab_accueil, tab_analyser, tab_lot, tab_ip, tab_historique = st.tabs(
    ["🏠 Vue d'ensemble", "🔍 Analyser une URL", "📦 Analyse par lot", "🌐 Adresses IP", "📜 Historique"]
)

# ============================== ONGLET VUE D'ENSEMBLE ==============================
with tab_accueil:
    df_hist_complet = charger_historique(limite=10_000)

    if df_hist_complet.empty:
        st.info("Aucune analyse effectu\u00e9e pour le moment. Utilisez les autres onglets pour commencer.")
    else:
        total_analyses = len(df_hist_complet)
        categories_complet = df_hist_complet["Resultat"].apply(categoriser_resultat)
        n_phishing_total = (categories_complet == "Phishing").sum()
        n_legitime_total = (categories_complet == "Legitime").sum()
        taux_detection = (n_phishing_total / total_analyses * 100) if total_analyses else 0
        n_ip_verifiees = df_hist_complet["URL"].str.startswith("[IP]").sum()

        # --- Cartes KPI ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            afficher_kpi_card("Analyses totales", total_analyses)
        with col2:
            afficher_kpi_card(
                "Taux de d\u00e9tection", f"{taux_detection:.1f}%",
                f"🚨 {n_phishing_total} phishing", "kpi-sub-danger",
            )
        with col3:
            afficher_kpi_card(
                "URLs l\u00e9gitimes", n_legitime_total,
                "✅ confirm\u00e9es s\u00fbres", "kpi-sub-ok",
            )
        with col4:
            afficher_kpi_card("IPs v\u00e9rifi\u00e9es", n_ip_verifiees, "via AbuseIPDB")

        st.markdown("---")

        col_activite, col_donut = st.columns([3, 2])
        with col_activite:
            st.markdown("#### Activit\u00e9 (14 derniers jours)")
            df_activite = charger_activite_par_jour(jours=14)
            if not df_activite.empty:
                st.area_chart(df_activite["nombre"], height=220)
            else:
                st.caption("Pas encore assez de donn\u00e9es.")

        with col_donut:
            st.markdown("#### R\u00e9partition")
            repartition_complete = categories_complet.value_counts().to_dict()
            for cle in ["Phishing", "Legitime", "Incertain", "Non concluant"]:
                repartition_complete.setdefault(cle, 0)
            afficher_donut_css(repartition_complete)

        st.markdown("---")

        col_pays, col_recents = st.columns([2, 3])
        with col_pays:
            st.markdown("#### 🌍 Top pays (IPs v\u00e9rifi\u00e9es)")
            afficher_top_pays(charger_top_pays(limite=5))

        with col_recents:
            st.markdown("#### ⚡ Alertes r\u00e9centes")
            df_alertes_recentes = df_hist_complet[
                df_hist_complet["Resultat"].str.startswith("phishing")
                | df_hist_complet["Resultat"].str.contains("suspecte")
            ].head(5)
            if not df_alertes_recentes.empty:
                st.dataframe(
                    df_alertes_recentes[["Horodatage", "URL", "Confiance"]],
                    width="stretch", hide_index=True,
                )
            else:
                st.caption("Aucune alerte r\u00e9cente.")

# ============================== ONGLET ANALYSER ==============================
with tab_analyser:
    with st.container(border=True):
        url_input = st.text_input(
            "URL a analyser",
            placeholder="https://exemple-suspect.com/verifier-compte",
            label_visibility="collapsed",
        )
        analyser_clicked = st.button("🔍 Analyser", type="primary", width="stretch")

    if analyser_clicked:
        if not url_input.strip():
            st.warning("Merci de coller une URL avant d'analyser.")
        else:
            url = url_input.strip()
            if not url.startswith(("http://", "https://")):
                url = "http://" + url

            with st.spinner(f"Analyse de {url} en cours..."):
                debut = time.time()
                resultat = analyser_url(url, model, imputer, extractor, FEATURE_COLUMNS)
                duree = time.time() - debut

            marge_confiance = abs(resultat["proba_phishing"] - resultat["proba_legitime"])
            est_incertain = marge_confiance < 0.30

            with st.container(border=True):
                st.caption(f"🔗 {url}")

                if resultat["bot_challenge_detected"]:
                    st.warning(
                        "⚠️ **Analyse non concluante** — le site a probablement "
                        "renvoye une page de v\u00e9rification anti-robot (CAPTCHA, "
                        "controle de navigateur) plutot que son contenu reel. "
                        "V\u00e9rification manuelle recommand\u00e9e."
                    )
                    st.caption(f"Prediction brute du modele : *{resultat['prediction']}* (a interpreter avec prudence)")

                elif est_incertain:
                    confiance_max = max(resultat["proba_phishing"], resultat["proba_legitime"])
                    st.warning(
                        f"🤔 **Resultat incertain** — confiance de seulement "
                        f"{confiance_max*100:.1f}\u00a0% pour '{resultat['prediction']}'. "
                        f"V\u00e9rification manuelle recommand\u00e9e."
                    )
                    col_gauge, col_val = st.columns([3, 1])
                    with col_gauge:
                        st.progress(float(resultat["proba_phishing"]))
                    with col_val:
                        st.markdown(f"**{resultat['proba_phishing']*100:.1f}\u00a0%** phishing")

                elif resultat["prediction"] == "phishing":
                    st.error("🚨 **URL suspecte : PHISHING probable**")
                    col_gauge, col_val = st.columns([3, 1])
                    with col_gauge:
                        st.progress(float(resultat["proba_phishing"]))
                    with col_val:
                        st.markdown(f"**{resultat['proba_phishing']*100:.1f}\u00a0%**")

                    if alertes_activees:
                        if not email_destinataire.strip():
                            st.warning("⚠️ Alertes activees mais aucune adresse e-mail renseignee.")
                        else:
                            try:
                                with st.spinner("Envoi de l'alerte par e-mail..."):
                                    envoyer_alerte_phishing(
                                        url=url,
                                        destinataire=email_destinataire.strip(),
                                        confiance=resultat["proba_phishing"],
                                        details={
                                            "is_https": resultat["raw_features"].get("is_https"),
                                            "total_links": resultat["raw_features"].get("total_links"),
                                            "form_count": resultat["raw_features"].get("form_count"),
                                        },
                                    )
                                st.success(f"📧 Alerte envoyee a {email_destinataire}")
                            except Exception as e:
                                st.error(f"❌ Echec de l'envoi de l'alerte : {e}")

                else:
                    st.success("✅ **URL probablement legitime**")
                    col_gauge, col_val = st.columns([3, 1])
                    with col_gauge:
                        st.progress(float(resultat["proba_legitime"]))
                    with col_val:
                        st.markdown(f"**{resultat['proba_legitime']*100:.1f}\u00a0%**")

                if resultat["page_injoignable"]:
                    st.info(
                        "ℹ️ La page n'a pas pu etre chargee (site hors ligne, "
                        "timeout, ou blocage). Cette absence de reponse est elle-meme "
                        "prise en compte par le modele."
                    )

                st.caption(f"⏱️ Analyse effectuee en {duree:.2f} secondes")

                st.markdown("**Signaux cles detectes :**")
                afficher_badges(generer_badges(resultat["raw_features"], resultat["page_injoignable"]))

                with st.expander("Voir le detail complet des features extraites"):
                    df_features = pd.DataFrame(
                        resultat["raw_features"].items(), columns=["Feature", "Valeur"]
                    )
                    df_features["Valeur"] = df_features["Valeur"].astype(str)
                    st.dataframe(df_features, width="stretch", hide_index=True)

            if resultat["bot_challenge_detected"]:
                statut_historique = "non concluant (anti-bot)"
            elif est_incertain:
                statut_historique = f"{resultat['prediction']} (incertain)"
            else:
                statut_historique = resultat["prediction"]

            ajouter_historique(
                url=url,
                resultat=statut_historique,
                confiance=f"{max(resultat['proba_phishing'], resultat['proba_legitime'])*100:.1f} %",
            )

# ============================== ONGLET ANALYSE PAR LOT ==============================
with tab_lot:
    st.markdown(
        "Uploade un fichier CSV contenant une colonne **url** (une URL par ligne) "
        "pour analyser plusieurs sites d'un coup."
    )

    fichier_csv = st.file_uploader("Fichier CSV", type=["csv"])

    if fichier_csv is not None:
        try:
            df_upload = pd.read_csv(fichier_csv)
        except Exception as e:
            st.error(f"Impossible de lire ce fichier CSV : {e}")
            df_upload = None

        if df_upload is not None:
            colonne_url = None
            for candidate in ["url", "URL", "Url"]:
                if candidate in df_upload.columns:
                    colonne_url = candidate
                    break
            if colonne_url is None:
                colonne_url = df_upload.columns[0]
                st.info(f"Aucune colonne 'url' trouvee, utilisation de la premiere colonne : '{colonne_url}'")

            urls_a_traiter = df_upload[colonne_url].dropna().astype(str).tolist()
            LIMITE_LOT = 50
            if len(urls_a_traiter) > LIMITE_LOT:
                st.warning(
                    f"Le fichier contient {len(urls_a_traiter)} URLs. Seules les "
                    f"{LIMITE_LOT} premieres seront analysees pour cette demo."
                )
                urls_a_traiter = urls_a_traiter[:LIMITE_LOT]

            st.write(f"**{len(urls_a_traiter)} URL(s)** prete(s) a analyser.")

            if st.button("🚀 Lancer l'analyse par lot", type="primary"):
                resultats_lot = []
                barre_progression = st.progress(0.0)
                texte_progression = st.empty()

                for i, url_brute in enumerate(urls_a_traiter):
                    url = url_brute.strip()
                    if not url.startswith(("http://", "https://")):
                        url = "http://" + url

                    texte_progression.caption(f"Analyse de {url} ({i+1}/{len(urls_a_traiter)})...")

                    try:
                        r = analyser_url(url, model, imputer, extractor, FEATURE_COLUMNS)
                        marge = abs(r["proba_phishing"] - r["proba_legitime"])
                        incertain = marge < 0.30

                        if r["bot_challenge_detected"]:
                            statut = "non concluant (anti-bot)"
                        elif incertain:
                            statut = f"{r['prediction']} (incertain)"
                        else:
                            statut = r["prediction"]

                        confiance = max(r["proba_phishing"], r["proba_legitime"]) * 100

                        resultats_lot.append({
                            "URL": url,
                            "Resultat": statut,
                            "Confiance (%)": round(confiance, 1),
                        })
                        ajouter_historique(url=url, resultat=statut, confiance=f"{confiance:.1f} %")

                    except Exception as e:
                        resultats_lot.append({
                            "URL": url,
                            "Resultat": f"erreur : {e}",
                            "Confiance (%)": None,
                        })

                    barre_progression.progress((i + 1) / len(urls_a_traiter))

                texte_progression.empty()
                st.success(f"Analyse termin\u00e9e pour {len(resultats_lot)} URL(s).")

                df_resultats_lot = pd.DataFrame(resultats_lot)
                st.dataframe(df_resultats_lot, width="stretch", hide_index=True)

                csv_export = df_resultats_lot.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Telecharger les resultats (CSV)",
                    data=csv_export,
                    file_name=f"analyse_lot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

# ============================== ONGLET ADRESSES IP ==============================
with tab_ip:
    st.markdown(
        "V\u00e9rifiez la r\u00e9putation d'une adresse IP (ou d'un nom de domaine, "
        "r\u00e9solu automatiquement) via la base communautaire **AbuseIPDB** -- "
        "un second type de menace, distinct de la d\u00e9tection de phishing par URL."
    )

    with st.container(border=True):
        ip_input = st.text_input(
            "Adresse IP ou URL",
            placeholder="8.8.8.8 ou https://exemple.com",
            label_visibility="collapsed",
            key="ip_input_field",
        )
        verifier_clicked = st.button("🌐 V\u00e9rifier", type="primary", width="stretch")

    if verifier_clicked:
        if not ip_input.strip():
            st.warning("Merci de saisir une adresse IP ou une URL.")
        else:
            with st.spinner("R\u00e9solution et v\u00e9rification en cours..."):
                ip_resolue = resoudre_ip(ip_input.strip())

            if not ip_resolue:
                st.error("❌ Impossible de r\u00e9soudre cette entr\u00e9e en adresse IP.")
            else:
                with st.spinner(f"Interrogation d'AbuseIPDB pour {ip_resolue}..."):
                    resultat_ip = verifier_ip(ip_resolue)

                with st.container(border=True):
                    st.caption(f"🌐 IP r\u00e9solue : {ip_resolue}")

                    if resultat_ip.get("erreur"):
                        st.error(f"❌ {resultat_ip['erreur']}")
                    else:
                        score = resultat_ip["abuse_score"]

                        if score >= 75:
                            st.error(f"🚨 **IP \u00e0 haut risque** (score d'abus : {score}%)")
                        elif score >= 25:
                            st.warning(f"⚠️ **IP potentiellement suspecte** (score d'abus : {score}%)")
                        else:
                            st.success(f"✅ **IP a priori s\u00fbre** (score d'abus : {score}%)")

                        st.progress(score / 100)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pays", resultat_ip.get("country", "?"))
                        col2.metric("Signalements", resultat_ip.get("total_reports", 0))
                        col3.metric("Fournisseur", resultat_ip.get("isp", "Inconnu")[:15])

                        st.markdown("**Signaux cl\u00e9s d\u00e9tect\u00e9s :**")
                        afficher_badges(generer_badges_ip(resultat_ip))

                    # --- Historique (reutilise la meme table, prefixe pour distinguer) ---
                    statut_ip = (
                        f"IP suspecte ({resultat_ip.get('abuse_score', '?')}%)"
                        if not resultat_ip.get("erreur") and resultat_ip.get("abuse_score", 0) >= 25
                        else "IP sure" if not resultat_ip.get("erreur") else "erreur verification IP"
                    )
                    ajouter_historique(
                        url=f"[IP] {ip_resolue}",
                        resultat=statut_ip,
                        confiance=f"{resultat_ip.get('abuse_score', '?')} %" if not resultat_ip.get("erreur") else "-",
                        pays=resultat_ip.get("country") if not resultat_ip.get("erreur") else None,
                    )

    st.caption(
        "ℹ️ N\u00e9cessite une cl\u00e9 API AbuseIPDB gratuite (voir ABUSEIPDB_API_KEY dans .env). "
        "Les r\u00e9sultats sont mis en cache 6h pour \u00e9conomiser le quota gratuit (1000 v\u00e9rifications/jour)."
    )

# ============================== ONGLET HISTORIQUE ==============================
with tab_historique:
    df_historique = charger_historique()

    if not df_historique.empty:
        st.markdown("#### Repartition des analyses")
        categories = df_historique["Resultat"].apply(categoriser_resultat)
        repartition = categories.value_counts().to_dict()
        for cle in ["Phishing", "Legitime", "Incertain", "Non concluant"]:
            repartition.setdefault(cle, 0)
        afficher_donut_css(repartition)

        st.markdown("---")
        st.markdown("#### D\u00e9tail des analyses")
        st.dataframe(df_historique, width="stretch", hide_index=True)
        st.caption(f"{len(df_historique)} analyse(s) enregistree(s) dans {HISTORIQUE_DB}")

        col_export, col_vider = st.columns(2)
        with col_export:
            csv_historique = df_historique.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Exporter l'historique (CSV)",
                data=csv_historique,
                file_name=f"historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
            )
        with col_vider:
            with st.popover("🗑️ Vider l'historique", width="stretch"):
                st.warning("Cette action est irr\u00e9versible.")
                if st.button("Confirmer la suppression", type="primary"):
                    vider_historique()
                    st.success("Historique vid\u00e9.")
                    st.rerun()
    else:
        st.caption("Aucune URL analysee pour le moment.")
