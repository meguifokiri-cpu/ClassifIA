"""
db.py — Connexion à la base de données
Responsabilité unique : ouvrir et fermer la connexion SQLite.
Importé par routes.py, jamais exécuté directement.
"""

import sqlite3
import os

# Chemin vers flowers.db — remonte de api/ vers la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, 'data', 'flowers.db')


def get_db() -> sqlite3.Connection:
    """
    Ouvre une connexion SQLite et retourne les lignes
    sous forme de dictionnaires grâce à row_factory.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # accès par nom de colonne : row['nom_commun']
    return conn
