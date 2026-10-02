import requests

BASE_URL = "http://localhost:5000"


def test_health():
    """L'API répond et se déclare en bonne santé."""
    reponse = requests.get(f"{BASE_URL}/health", timeout=30)

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_liste_plantes():
    """La base de données répond et contient les 30 espèces."""
    reponse = requests.get(f"{BASE_URL}/plants", timeout=30)

    assert reponse.status_code == 200

    plantes = reponse.json()
    assert len(plantes) == 30
    assert "nom_commun" in plantes[0]

def test_prediction_plante_connue():
    """Une image d'aloe vera est correctement identifiée."""
    with open("data/test/aloe-vera.webp", "rb") as fichier:
        reponse = requests.post(
            f"{BASE_URL}/predict",
            files={"image": fichier},
            timeout=60
        )

    assert reponse.status_code == 200

    resultat = reponse.json()
    assert resultat["nom_dossier"] == "aloe_vera"
    assert resultat["confiance"] > 80


def test_rejet_image_non_plante():
    """Une image d'avion est rejetée par la détection OOD."""
    with open("data/test/avion.webp", "rb") as fichier:
        reponse = requests.post(
            f"{BASE_URL}/predict",
            files={"image": fichier},
            timeout=60
        )

    assert reponse.status_code == 422

    resultat = reponse.json()
    assert "diagnostics" in resultat
    assert "distance_mahalanobis" in resultat["diagnostics"]

def test_format_non_supporte():
    """Un fichier qui n'est pas une image est refusé proprement."""
    reponse = requests.post(
        f"{BASE_URL}/predict",
        files={"image": ("document.txt", b"ceci n'est pas une image", "text/plain")},
        timeout=30
    )

    assert reponse.status_code == 400


def test_champ_image_manquant():
    """Une requête sans fichier est refusée."""
    reponse = requests.post(f"{BASE_URL}/predict", timeout=30)

    assert reponse.status_code == 400