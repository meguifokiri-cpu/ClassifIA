import os
import time
import requests

ESPECES = [
    # "Rosa multiflora",
    # "Tulipa gesneriana",
    # "Hedera helix",
    "Hydrangea macrophylla",
    # "Salvia rosmarinus",
    # "Monstera deliciosa",
    # "Echeveria gibbiflora",
    # "Pteridium aquilinum",
]

IMAGES_PAR_ESPECE = 100
DOSSIER_SORTIE    = "data/raw/autre_espece_vegetale"
MAX_TENTATIVES    = 3

os.makedirs(DOSSIER_SORTIE, exist_ok=True)


def requete_avec_retry(url, params, tentatives=MAX_TENTATIVES):
    """Réessaie une requête en cas d'erreur réseau ponctuelle, avec pause croissante."""
    for essai in range(1, tentatives + 1):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"    [RETRY {essai}/{tentatives}] Erreur réseau : {type(e).__name__}")
            if essai < tentatives:
                time.sleep(2 * essai)  # pause croissante : 2s, 4s, 6s...
            else:
                print(f"    [ABANDON] Échec après {tentatives} tentatives")
                return None
    return None


def rechercher_taxon_id(nom_espece: str) -> int | None:
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": nom_espece, "rank": "species", "per_page": 1}
    r = requete_avec_retry(url, params)
    if r is None:
        return None
    data = r.json()
    if data["results"]:
        return data["results"][0]["id"]
    return None


def telecharger_observations(taxon_id: int, nom_espece: str, limite: int) -> int:
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(limite, 200),
        "order_by": "votes",
    }
    r = requete_avec_retry(url, params)
    if r is None:
        return 0

    data = r.json()
    compteur = 0
    nom_fichier_safe = nom_espece.replace(" ", "_").lower()

    for obs in data.get("results", []):
        if compteur >= limite:
            break
        photos = obs.get("photos", [])
        if not photos:
            continue

        url_photo = photos[0]["url"].replace("square", "medium")

        try:
            img_data = requests.get(url_photo, timeout=10).content
            chemin = os.path.join(DOSSIER_SORTIE, f"{nom_fichier_safe}_{compteur:03d}.jpg")
            with open(chemin, "wb") as f:
                f.write(img_data)
            compteur += 1
        except Exception as e:
            print(f"    [ERREUR PHOTO] {type(e).__name__} — ignorée, on continue")

        time.sleep(0.5)

    return compteur


if __name__ == "__main__":
    total = 0
    for espece in ESPECES:
        print(f"Recherche : {espece}...")
        try:
            taxon_id = rechercher_taxon_id(espece)
            if taxon_id is None:
                print(f"  [IGNORÉ] Espèce introuvable ou erreur réseau persistante : {espece}")
                continue

            nb = telecharger_observations(taxon_id, espece, IMAGES_PAR_ESPECE)
            print(f"  {nb} images téléchargées pour {espece}")
            total += nb
        except Exception as e:
            print(f"  [ERREUR INATTENDUE] {type(e).__name__}: {e} — on passe à l'espèce suivante")
            continue

    print(f"\n[OK] Total : {total} images dans {DOSSIER_SORTIE}")