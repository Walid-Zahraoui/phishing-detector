"""
feature_extractor.py

Module d'extraction de features, utilise par le tableau de bord Streamlit
pour analyser une URL en direct. C'est EXACTEMENT la meme logique que
celle utilisee pour construire le jeu d'entrainement (Jalon 3, notebook
Colab, Cell 2) -- il est essentiel que les deux soient identiques, sinon
le modele recoit des features differentes de celles sur lesquelles il a
appris (train/serve skew).
"""

import re
import math
import hashlib
import json
import time
import threading
import sqlite3
from urllib.parse import urlparse
from collections import Counter

import requests
from bs4 import BeautifulSoup


# Expressions typiques des pages de "challenge" anti-robot (CAPTCHA,
# verification de navigateur, blocage Cloudflare, etc.). Quand une de
# ces expressions apparait, la page recuperee n'est probablement PAS le
# vrai contenu du site -- c'est une interstitielle de protection, qui
# ressemble structurellement a du phishing (peu de liens, pas de
# formulaire) sans en etre. On la detecte pour eviter un faux verdict.
INDICES_PAGE_CHALLENGE = [
    "checking your browser",
    "just a moment",
    "attention required",
    "cf-browser-verification",
    "captcha",
    "verify you are human",
    "verify you're human",
    "unusual traffic",
    "enable javascript and cookies",
    "please verify you are a human",
    "ddos protection by",
    "are you a robot",
    "access denied",
    "please enable cookies",
]


def detecter_page_challenge(html: str) -> bool:
    """
    Renvoie True si le HTML recu ressemble a une page de verification
    anti-robot plutot qu'au vrai contenu du site.
    """
    if not html:
        return False
    texte = html.lower()[:5000]  # les indices apparaissent generalement tot dans la page
    return any(indice in texte for indice in INDICES_PAGE_CHALLENGE)


class PhishingFeatureExtractor:
    LEXICAL_FEATURES = [
        "url_length", "hostname_length", "path_length", "query_length",
        "dot_count", "hyphen_count", "underscore_count", "equal_count",
        "question_mark_count", "has_ip_address", "has_at_symbol", "has_port",
        "subdomain_count", "url_entropy", "domain_entropy"
    ]
    HTTP_FEATURES = [
        "is_https", "http_status_code", "redirect_count", "response_time_ms",
        "has_security_headers", "content_length"
    ]
    DOM_FEATURES = [
        "form_count", "external_form_count", "empty_form_action",
        "total_links", "external_link_ratio", "empty_href_count",
        "iframe_count", "right_click_disabled", "has_favicon_mismatch"
    ]
    ALL_FEATURES = LEXICAL_FEATURES + HTTP_FEATURES + DOM_FEATURES

    def __init__(self, timeout=5, rate_limit=0.1, cache_db="cache_dashboard.db"):
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.cache_db = cache_db
        self.session = requests.Session()
        # En-tetes plus proches d'un vrai navigateur : reduit (sans
        # eliminer) le taux de blocage par les protections anti-bot des
        # grandes plateformes (Google, Amazon, YouTube, GitHub), qui
        # servaient jusqu'ici une page de "challenge" (CAPTCHA,
        # verification navigateur) au lieu du vrai contenu -- page qui
        # ressemble structurellement a du phishing (peu de liens, pas
        # de formulaire) et causait des faux positifs.
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        })
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._lock = threading.Lock()
        self._init_cache()

    def _init_cache(self):
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        url_hash TEXT PRIMARY KEY,
                        url TEXT NOT NULL,
                        features_json TEXT NOT NULL,
                        timestamp INTEGER NOT NULL
                    )
                """)
                conn.commit()

    def _get_cache(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("SELECT features_json FROM cache WHERE url_hash = ?", (url_hash,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        return None

    def _set_cache(self, url, features):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (url_hash, url, features_json, timestamp) VALUES (?, ?, ?, ?)",
                    (url_hash, url, json.dumps(features), int(time.time()))
                )
                conn.commit()

    def _is_ip_address(self, hostname):
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
            return True
        if ":" in hostname and re.match(r"^[0-9a-fA-F:]+$", hostname):
            return True
        return False

    def _calculate_entropy(self, text):
        if not text:
            return 0.0
        counter = Counter(text)
        length = len(text)
        return -sum((count / length) * math.log2(count / length) for count in counter.values() if count > 0)

    def _get_lexical_features(self, url):
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return {
            "url_length": len(url), "hostname_length": len(hostname),
            "path_length": len(parsed.path), "query_length": len(parsed.query),
            "dot_count": url.count("."), "hyphen_count": url.count("-"),
            "underscore_count": url.count("_"), "equal_count": url.count("="),
            "question_mark_count": url.count("?"), "has_ip_address": self._is_ip_address(hostname),
            "has_at_symbol": "@" in url, "has_port": parsed.port is not None,
            "subdomain_count": max(0, hostname.count(".") - 1),
            "url_entropy": self._calculate_entropy(url), "domain_entropy": self._calculate_entropy(hostname)
        }

    def _get_http_features(self, response):
        parsed_url = urlparse(response.url)
        security_headers = ["Strict-Transport-Security", "X-Frame-Options", "X-Content-Type-Options", "Content-Security-Policy"]
        return {
            "is_https": parsed_url.scheme == "https",
            "http_status_code": response.status_code,
            "redirect_count": len(response.history),
            "response_time_ms": response.elapsed.total_seconds() * 1000,
            "has_security_headers": any(h in response.headers for h in security_headers),
            "content_length": len(response.content)
        }

    def _get_dom_features(self, url, soup):
        parsed_url = urlparse(url)
        current_domain = parsed_url.hostname or ""
        forms = soup.find_all("form")

        ext_forms = sum(1 for f in forms if f.get("action", "").startswith("http") and urlparse(f["action"]).hostname != current_domain)
        empty_actions = sum(1 for f in forms if not f.get("action") or f["action"] == "#")

        links = soup.find_all("a")
        ext_links = sum(1 for l in links if l.get("href", "").startswith("http") and urlparse(l["href"]).hostname != current_domain)
        empty_hrefs = sum(1 for l in links if not l.get("href") or l["href"] == "#")

        body = soup.find("body")
        favicon_links = soup.find_all("link", rel=lambda x: x and "icon" in x.lower() if x else False)
        favicon_mismatch = any(urlparse(l.get("href", "")).hostname != current_domain for l in favicon_links if l.get("href", "").startswith("http"))

        return {
            "form_count": len(forms), "external_form_count": ext_forms, "empty_form_action": empty_actions > 0,
            "total_links": len(links), "external_link_ratio": (ext_links / len(links) if links else 0.0),
            "empty_href_count": empty_hrefs, "iframe_count": len(soup.find_all("iframe")),
            "right_click_disabled": bool(body and "oncontextmenu" in body.attrs and "return false" in body["oncontextmenu"]),
            "has_favicon_mismatch": favicon_mismatch
        }

    def extract(self, url):
        cached = self._get_cache(url)
        if cached:
            return cached

        features = self._get_lexical_features(url)
        # Metadonnee separee des features du modele : ne sert qu'a
        # l'affichage/decision au niveau du tableau de bord, jamais
        # donnee au modele (pour ne pas devoir re-entrainer).
        features["_bot_challenge_detected"] = False

        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            features.update(self._get_http_features(response))
            features.update(self._get_dom_features(url, BeautifulSoup(response.text, "html.parser")))
            features["_bot_challenge_detected"] = detecter_page_challenge(response.text)
        except requests.exceptions.RequestException:
            for feat in self.HTTP_FEATURES + self.DOM_FEATURES:
                features[feat] = None

        self._set_cache(url, features)
        return features
