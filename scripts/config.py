"""
config.py — chemins et constantes partagés par tous les scripts
Détecte automatiquement si on est sur Colab ou sur PC local.
"""

import os

# ─────────────────────────────────────────────
# MODIFICATION 1 — Détection automatique Colab vs PC
# ─────────────────────────────────────────────
# On vérifie si le dossier /content/drive existe.
# Ce dossier n'existe QUE sur Google Colab après mount du Drive.
# Sur ton PC Windows, il n'existera jamais → on prend le chemin local.
# ─────────────────────────────────────────────
if os.path.exists('/content/drive'):
    # On est sur Colab → les fichiers sont dans ton Google Drive
    BASE_DIR = '/content/drive/MyDrive/classif-IA'
else:
    # On est sur ton PC → chemin relatif au fichier config.py
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Données — inchangé, fonctionne sur les deux environnements
# ─────────────────────────────────────────────
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PREPARE_DIR = os.path.join(BASE_DIR, "data", "dataset_prepare")
TRAIN_DIR   = os.path.join(PREPARE_DIR, "train")
VAL_DIR     = os.path.join(PREPARE_DIR, "val")
TEST_DIR    = os.path.join(PREPARE_DIR, "test")

# ─────────────────────────────────────────────
# MODIFICATION 2 — Dossier modèle explicite
# Avant : le dossier model/ n'était pas défini ici → chaque script
# le définissait différemment → risque de chemin incorrect.
# Maintenant : un seul endroit pour le changer.
# ─────────────────────────────────────────────
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "modele_final.h5")
BEST_PATH    = os.path.join(MODEL_DIR, "modele_best.h5")
INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.png")

# ─────────────────────────────────────────────
# MODIFICATION 3 — Création automatique du dossier model/
# Avant : si le dossier model/ n'existait pas, train_model.py
# crashait au moment de sauvegarder. Maintenant il est créé
# automatiquement dès que config.py est importé.
# C'est aussi la raison pour laquelle class_indices.json
# n'était pas créé — le dossier model/ n'existait pas !
# ─────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Paramètres image & entraînement — inchangés
# ─────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

AUGMENTATION = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)