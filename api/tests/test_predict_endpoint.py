"""
test_predict_endpoint.py — Tests de l'endpoint POST /predict.

On utilise monkeypatch pour remplacer predict_species() et get_db()
par de fausses versions (mocks). Pourquoi ?
  - On teste la LOGIQUE de l'endpoint (gestion des erreurs, format
    de la réponse), pas la précision du modèle IA lui-même.
  - Les tests restent rapides et ne dépendent pas du contenu réel
    de la base de données (qui peut changer).
"""

import io
import sqlite3
import pytest
from PIL import Image

import predict as predict_module


def _image_test_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 180, 90))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class FauxRow(dict):
    """Simule un sqlite3.Row : accessible par dict(row)."""
    pass


def _faux_get_db_avec_resultat():
    """Simule une connexion DB qui retourne toujours une fiche 'mango'."""
    class FauxConnexion:
        def execute(self, query, params=()):
            self._resultat = FauxRow(
                id_espece=1, nom_dossier="mango", nom_commun="Manguier",
                nom_latin="Mangifera indica", image_ref=None,
                famille="Anacardiaceae", utilite_principale="Fruit comestible",
                description_wikipedia="La mangue est...", avertissement_securite=None,
                hauteur="10-30m", besoin_eau="Modéré", exposition="Plein soleil",
                saison_floraison="Printemps", origine="Asie du Sud"
            )
            return self
        def fetchone(self):
            return self._resultat
        def close(self):
            pass
    return FauxConnexion()


def _faux_get_db_sans_resultat():
    """Simule une connexion DB où l'espèce prédite n'existe pas en base."""
    class FauxConnexion:
        def execute(self, query, params=()):
            return self
        def fetchone(self):
            return None
        def close(self):
            pass
    return FauxConnexion()


# ══════════════════════════════════════════════════════════════
# Cas nominal : image valide, espèce trouvée en base
# ══════════════════════════════════════════════════════════════

def test_predict_retourne_200_et_la_fiche_espece(client, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_species", lambda img: ("mango", 0.9134))
    monkeypatch.setattr(predict_module, "get_db", _faux_get_db_avec_resultat)

    reponse = client.post(
        "/predict",
        data={"image": (_image_test_bytes(), "test.png")},
        content_type="multipart/form-data"
    )

    assert reponse.status_code == 200
    data = reponse.get_json()
    assert data["nom_dossier"] == "mango"
    assert data["confiance"] == 91.34
    assert data["famille"] == "Anacardiaceae"


# ══════════════════════════════════════════════════════════════
# Cas d'erreur : champ image manquant
# ══════════════════════════════════════════════════════════════

def test_predict_sans_image_retourne_400(client):
    reponse = client.post("/predict", data={}, content_type="multipart/form-data")
    assert reponse.status_code == 400
    assert "error" in reponse.get_json()


# ══════════════════════════════════════════════════════════════
# Cas d'erreur : format de fichier non supporté
# ══════════════════════════════════════════════════════════════

def test_predict_format_non_supporte_retourne_400(client):
    faux_pdf = io.BytesIO(b"pas une vraie image")
    reponse = client.post(
        "/predict",
        data={"image": (faux_pdf, "document.pdf")},
        content_type="multipart/form-data"
    )
    assert reponse.status_code == 400


# ══════════════════════════════════════════════════════════════
# Cas d'erreur : espèce prédite absente de la base de données
# ══════════════════════════════════════════════════════════════

def test_predict_espece_absente_bdd_retourne_404(client, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_species", lambda img: ("espece_inconnue", 0.5))
    monkeypatch.setattr(predict_module, "get_db", _faux_get_db_sans_resultat)

    reponse = client.post(
        "/predict",
        data={"image": (_image_test_bytes(), "test.png")},
        content_type="multipart/form-data"
    )

    assert reponse.status_code == 404
    data = reponse.get_json()
    assert data["nom_dossier"] == "espece_inconnue"
