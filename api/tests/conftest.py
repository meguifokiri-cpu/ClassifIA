"""
conftest.py — Fixtures pytest partagées par tous les tests.

Ce fichier est chargé automatiquement par pytest (pas besoin de
l'importer). Il permet de définir des objets réutilisables
(ici : l'application Flask et son client de test) sans dupliquer
le code d'initialisation dans chaque fichier de test.
"""

import os
import sys
import pytest

# Permet d'importer app.py, predict.py, etc. depuis la racine du projet
# quand pytest est lancé depuis le dossier tests/ ou depuis la racine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


@pytest.fixture(scope='session')
def app():
    """
    Fournit l'application Flask configurée en mode test.
    scope='session' : créée une seule fois pour tous les tests,
    car charger le modèle TensorFlow prend plusieurs secondes —
    inutile de le refaire à chaque test.
    """
    flask_app.config.update({"TESTING": True})
    yield flask_app


@pytest.fixture()
def client(app):
    """Fournit un client HTTP simulé pour appeler les endpoints
    sans lancer un vrai serveur (pas de port, pas de réseau)."""
    return app.test_client()
