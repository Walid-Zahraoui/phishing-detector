"""
poc_headless.py - Preuve de concept : navigateur headless vs scraping leger

Compare les features extraites par les DEUX methodes sur des sites
fortement bases sur JavaScript (GitHub, YouTube), pour verifier
l'hypothese du rapport : le scraping leger (requests + BeautifulSoup)
manque le contenu genere dynamiquement, ce qui cause les faux positifs
observes sur ces sites.

Ce script ne modifie PAS le pipeline existant -- c'est une experience
isolee, pour valider (ou invalider) l'hypothese avant d'investir le
temps d'un re-scraping + re-entrainement complets.

Pre-requis :
    pip install playwright requests beautifulsoup4
    playwright install chromium

Usage :
    python poc_headless.py
"""

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# Features comparees : celles qui dependent du DOM/contenu de la page
# (les features purement lexicales, basees sur le texte de l'URL, ne
# peuvent pas changer selon la methode d'extraction -- inutile de les
# comparer ici)
FEATURES_COMPAREES = ["content_length", "total_links", "form_count", "external_link_ratio"]


def extraire_leger(url: str) -> dict:
    """Methode actuelle du projet : requests + BeautifulSoup, sans JS."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    domaine = url.split("/")[2]
    liens = soup.find_all("a")
    liens_externes = sum(
        1 for l in liens
        if l.get("href", "").startswith("http") and domaine not in l.get("href", "")
    )

    return {
        "http_status_code": response.status_code,
        "content_length": len(response.content),
        "total_links": len(liens),
        "form_count": len(soup.find_all("form")),
        "external_link_ratio": round(liens_externes / len(liens), 3) if liens else 0.0,
    }


def extraire_headless(url: str) -> dict:
    """Nouvelle methode : Playwright, execute le JavaScript avant d'extraire."""
    with sync_playwright() as p:
        navigateur = p.chromium.launch(headless=True)
        page = navigateur.new_page(user_agent=HEADERS["User-Agent"])
        reponse = page.goto(url, wait_until="networkidle", timeout=20000)
        html = page.content()
        statut = reponse.status if reponse else None
        navigateur.close()

    soup = BeautifulSoup(html, "html.parser")
    domaine = url.split("/")[2]
    liens = soup.find_all("a")
    liens_externes = sum(
        1 for l in liens
        if l.get("href", "").startswith("http") and domaine not in l.get("href", "")
    )

    return {
        "http_status_code": statut,
        "content_length": len(html.encode("utf-8")),
        "total_links": len(liens),
        "form_count": len(soup.find_all("form")),
        "external_link_ratio": round(liens_externes / len(liens), 3) if liens else 0.0,
    }


def comparer(url: str):
    print(f"\n{'='*70}")
    print(f"Comparaison pour : {url}")
    print(f"{'='*70}")

    print("Extraction en cours (requests, SANS JavaScript)...")
    leger = extraire_leger(url)

    print("Extraction en cours (Playwright, AVEC JavaScript)...")
    headless = extraire_headless(url)

    print(f"\n{'Feature':<22} {'Sans JS (actuel)':<20} {'Avec JS (headless)':<20} {'Ecart':<10}")
    print("-" * 72)
    for cle in ["http_status_code"] + FEATURES_COMPAREES:
        val_leger = leger.get(cle)
        val_headless = headless.get(cle)
        try:
            ecart = f"{(val_headless - val_leger):+.1f}" if isinstance(val_leger, (int, float)) else "-"
        except TypeError:
            ecart = "-"
        print(f"{cle:<22} {str(val_leger):<20} {str(val_headless):<20} {ecart:<10}")

    return leger, headless


if __name__ == "__main__":
    urls_a_tester = [
        "https://github.com",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]

    resultats = {}
    for url in urls_a_tester:
        resultats[url] = comparer(url)

    print(f"\n{'='*70}")
    print("=== Conclusion ===")
    print(f"{'='*70}")
    for url, (leger, headless) in resultats.items():
        diff_liens = headless["total_links"] - leger["total_links"]
        print(f"{url} : {diff_liens:+d} liens detectes en plus avec le rendu JS")
