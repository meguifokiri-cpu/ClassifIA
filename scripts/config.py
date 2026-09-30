"""
config.py — chemins et constantes partagés par tous les scripts
Adaptez BASE_DIR selon votre machine.
"""

import os

# ─────────────────────────────────────────────
# Racine du projet (dossier classifia/)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Données
# ─────────────────────────────────────────────
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")          # images brutes (30 sous-dossiers d'espèces)
PREPARE_DIR = os.path.join(BASE_DIR, "data", "dataset_prepare")
TRAIN_DIR   = os.path.join(PREPARE_DIR, "train")
VAL_DIR     = os.path.join(PREPARE_DIR, "val")
TEST_DIR    = os.path.join(PREPARE_DIR, "test")

# ─────────────────────────────────────────────
# Paramètres image & entraînement
# ─────────────────────────────────────────────
IMG_SIZE   = (224, 224)    # taille d'entrée MobileNetV2
BATCH_SIZE = 32
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# ─────────────────────────────────────────────
# Division train / val / test
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


# ─────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────
AUGMENTATION = dict(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="reflect",
)