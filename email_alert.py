"""
email_alert.py - Jalon 3 : envoi d'alertes automatiques par e-mail

Utilise le module standard smtplib avec un compte Gmail (mot de passe
d'application requis -- voir README).

Les identifiants ne sont JAMAIS ecrits en dur dans ce fichier : ils sont
lus depuis des variables d'environnement (fichier .env, charge via
python-dotenv). Cela evite d'exposer un mot de passe si ce code est
partage ou mis sur un depot Git.
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()  # charge les variables depuis le fichier .env s'il existe

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

SMTP_SERVEUR = "smtp.gmail.com"
SMTP_PORT = 587


def envoyer_alerte_phishing(url: str, destinataire: str, confiance: float, details: dict = None):
    """
    Envoie un e-mail d'alerte lorsqu'une URL est detectee comme phishing.

    Parametres :
        url         -- l'URL suspecte detectee
        destinataire -- adresse e-mail qui recoit l'alerte
        confiance   -- probabilite de phishing (entre 0 et 1)
        details     -- dictionnaire optionnel de features cles a inclure
                        dans le corps du message (pour contexte)

    Leve une exception si l'envoi echoue (identifiants manquants ou
    incorrects, probleme reseau, etc.) -- a gerer par l'appelant.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise ValueError(
            "Identifiants Gmail manquants. Verifiez que le fichier .env "
            "contient GMAIL_ADDRESS et GMAIL_APP_PASSWORD."
        )

    msg = MIMEMultipart()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = destinataire
    msg["Subject"] = "🚨 Alerte : URL de phishing detectee"

    lignes_details = ""
    if details:
        lignes_details = "\n".join(f"  - {cle} : {valeur}" for cle, valeur in details.items())

    corps = f"""Une URL suspecte a ete detectee par le systeme de surveillance.

URL analysee : {url}
Probabilite de phishing : {confiance*100:.1f} %
Date de detection : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{lignes_details}

--
Ceci est une alerte automatique generee par le systeme de detection et
d'alerte precoce des cybermenaces ciblant les PME (CMRPI / EMC).
Ne pas repondre a cet e-mail.
"""
    msg.attach(MIMEText(corps, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_SERVEUR, SMTP_PORT) as serveur:
        serveur.starttls()
        serveur.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        serveur.send_message(msg)


if __name__ == "__main__":
    # Test manuel : python email_alert.py
    destinataire_test = input("Adresse e-mail de test : ").strip()
    try:
        envoyer_alerte_phishing(
            url="http://exemple-suspect-test.com/verifier-compte",
            destinataire=destinataire_test,
            confiance=0.97,
            details={"is_https": False, "total_links": 2, "form_count": 1},
        )
        print("✅ E-mail envoye avec succes.")
    except Exception as e:
        print(f"❌ Echec de l'envoi : {e}")
