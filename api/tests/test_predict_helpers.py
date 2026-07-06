"""
test_predict_helpers.py — Tests unitaires des fonctions utilitaires
de predict.py : allowed_file() et prepare_image().

Ce sont des fonctions pures (pas d'appel réseau, pas de base de
données, pas de modèle) — idéales pour des tests unitaires rapides
et déterministes.
"""

import io
import numpy as np
from PIL import Image

from predict import allowed_file, prepare_image, ALLOWED_EXTENSIONS, IMG_SIZE


# ══════════════════════════════════════════════════════════════
# Tests de allowed_file()
# ══════════════════════════════════════════════════════════════

def test_allowed_file_accepte_jpg():
    assert allowed_file("mango.jpg") is True


def test_allowed_file_accepte_png():
    assert allowed_file("banane.PNG") is True  # insensible à la casse


def test_allowed_file_refuse_extension_non_supportee():
    assert allowed_file("document.pdf") is False


def test_allowed_file_refuse_fichier_sans_extension():
    assert allowed_file("fichier_sans_extension") is False


def test_allowed_file_toutes_extensions_supportees():
    for ext in ALLOWED_EXTENSIONS:
        assert allowed_file(f"test.{ext}") is True


# ══════════════════════════════════════════════════════════════
# Tests de prepare_image()
# ══════════════════════════════════════════════════════════════

def _creer_image_test(mode="RGB", couleur=(100, 150, 80), taille=(300, 200)) -> bytes:
    """Génère une image en mémoire (sans fichier disque) pour les tests."""
    img = Image.new(mode, taille, color=couleur)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_prepare_image_redimensionne_a_224x224():
    image_bytes = _creer_image_test(taille=(500, 350))
    resultat = prepare_image(image_bytes)
    assert resultat.shape == (IMG_SIZE[0], IMG_SIZE[1], 3)


def test_prepare_image_convertit_rgba_en_rgb():
    # Une image RGBA (avec transparence) doit être convertie en RGB
    # (3 canaux), sinon le modèle plante (il attend 3 canaux, pas 4)
    image_bytes = _creer_image_test(mode="RGBA", couleur=(50, 100, 50, 128))
    resultat = prepare_image(image_bytes)
    assert resultat.shape[-1] == 3


def test_prepare_image_retourne_un_array_numpy():
    image_bytes = _creer_image_test()
    resultat = prepare_image(image_bytes)
    assert isinstance(resultat, np.ndarray)


def test_prepare_image_valeurs_dans_la_plage_0_255():
    image_bytes = _creer_image_test(couleur=(255, 0, 128))
    resultat = prepare_image(image_bytes)
    assert resultat.min() >= 0
    assert resultat.max() <= 255
