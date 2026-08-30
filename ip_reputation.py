"""
ip_reputation.py - Jalon 3 (extension) : verification de reputation IP

Deuxieme type de menace, en complement de la detection de phishing par
URL : verifie une adresse IP contre la base communautaire AbuseIPDB
(https://www.abuseipdb.com), qui recense les IP impliquees dans des
activites malveillantes signalees par la communaute (scans, brute-force,
spam, etc.).

Necessite une cle API gratuite (voir README) -- inscription simple,
1000 verifications/jour sur le tier gratuit.
"""

import os
import socket
import sqlite3
import json
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

ABUSEIPDB_API_KEY = os.environ.get("ABUSEIPDB_API_KEY")
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"

CACHE_DB = "ip_reputation_cache.db"
CACHE_MAX_AGE_SECONDS = 6 * 3600  # 6 heures : evite de re-interroger l'API trop souvent


def _init_cache():
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                ip TEXT PRIMARY KEY,
                resultat_json TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        conn.commit()


def _get_cache(ip: str):
    with sqlite3.connect(CACHE_DB) as conn:
        cursor = conn.execute("SELECT resultat_json, timestamp FROM cache WHERE ip = ?", (ip,))
        row = cursor.fetchone()
        if row:
            resultat_json, timestamp = row
            if time.time() - timestamp < CACHE_MAX_AGE_SECONDS:
                return json.loads(resultat_json)
    return None


def _set_cache(ip: str, resultat: dict):
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (ip, resultat_json, timestamp) VALUES (?, ?, ?)",
            (ip, json.dumps(resultat), int(time.time())),
        )
        conn.commit()


_init_cache()


def resoudre_ip(url_ou_ip: str) -> str | None:
    """
    Si l'entree est deja une adresse IP, la renvoie telle quelle.
    Sinon, tente de resoudre le nom de domaine en adresse IP (DNS).
    Renvoie None si la resolution echoue.
    """
    valeur = url_ou_ip.strip()

    # Si c'est deja une IP (format simple, verification basique)
    parties = valeur.replace("http://", "").replace("https://", "").split("/")[0]
    try:
        socket.inet_aton(parties)
        return parties  # deja une IP valide
    except OSError:
        pass

    # Sinon, on extrait le hostname et on resout via DNS
    if not valeur.startswith(("http://", "https://")):
        valeur = "http://" + valeur
    hostname = urlparse(valeur).hostname

    if not hostname:
        return None

    try:
        return socket.gethostbyname(hostname)
    except (socket.gaierror, socket.timeout):
        return None


def verifier_ip(ip: str, max_age_days: int = 90) -> dict:
    """
    Interroge AbuseIPDB pour une adresse IP donnee. Utilise un cache
    local (6h) pour eviter de consommer le quota API inutilement sur
    des verifications repetees.

    Renvoie un dictionnaire avec : abuse_score (0-100), total_reports,
    country, isp, is_tor, is_whitelisted, domain, erreur (si echec).
    """
    cached = _get_cache(ip)
    if cached:
        return cached

    if not ABUSEIPDB_API_KEY:
        return {
            "ip": ip,
            "erreur": "Cle API AbuseIPDB manquante (verifiez ABUSEIPDB_API_KEY dans .env)",
        }

    headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": max_age_days, "verbose": ""}

    try:
        response = requests.get(ABUSEIPDB_URL, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()["data"]

        resultat = {
            "ip": ip,
            "abuse_score": data.get("abuseConfidenceScore", 0),
            "total_reports": data.get("totalReports", 0),
            "country": data.get("countryCode", "?"),
            "isp": data.get("isp", "Inconnu"),
            "domain": data.get("domain", ""),
            "is_tor": data.get("isTor", False),
            "is_whitelisted": data.get("isWhitelisted", False),
            "erreur": None,
        }
        _set_cache(ip, resultat)
        return resultat

    except requests.exceptions.RequestException as e:
        return {"ip": ip, "erreur": f"Erreur reseau : {e}"}
    except (KeyError, ValueError) as e:
        return {"ip": ip, "erreur": f"Reponse AbuseIPDB inattendue : {e}"}


if __name__ == "__main__":
    # Test manuel : python ip_reputation.py
    test_ip = input("Adresse IP ou URL a tester : ").strip()
    ip_resolue = resoudre_ip(test_ip)
    if not ip_resolue:
        print("❌ Impossible de resoudre cette entree en adresse IP.")
    else:
        print(f"IP resolue : {ip_resolue}")
        resultat = verifier_ip(ip_resolue)
        print(json.dumps(resultat, indent=2, ensure_ascii=False))
