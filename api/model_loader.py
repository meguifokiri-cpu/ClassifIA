"""
model_loader.py — Chargement du modèle IA au démarrage
Responsabilité unique : charger modele_final.h5 et class_indices.json
une seule fois en mémoire au lancement de l'API.

Importé par predict.py, jamais exécuté directement.

Pourquoi charger une seule fois ?
→ Charger un .h5 prend 2 à 3 secondes.
  Sans ce fichier, chaque requête /predict rechargerait le modèle
  depuis le disque → l'API serait inutilisable en pratique.
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from monitoring import MODEL_CLASSES_LOADED

# ── Chemins ───────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'modele_best.h5')
INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

# ── Chargement au démarrage ───────────────────────────────────────────
# Ces deux variables sont initialisées une fois et réutilisées
# par chaque appel à predict.py sans rechargement.

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}\n"
        "→ Exécutez d'abord scripts/train_model.py"
    )

if not os.path.exists(INDICES_PATH):
    raise FileNotFoundError(
        f"Indices introuvables : {INDICES_PATH}\n"
        "→ Exécutez d'abord scripts/train_model.py"
    )

print("  Chargement du modèle IA...")
model = load_model(MODEL_PATH)

with open(INDICES_PATH, encoding='utf-8') as f:
    # {"0": "banana", "1": "mango", ...}
    idx_to_class = json.load(f)

print(f"  Modèle chargé — {len(idx_to_class)} classes disponibles")
MODEL_CLASSES_LOADED.set(len(idx_to_class))


# ── Fonction de prédiction ────────────────────────────────────────────
IMG_SIZE = (224, 224)

def predict_species(img_array: np.ndarray) -> tuple[str, float]:
    """
    Reçoit une image sous forme de tableau numpy (224, 224, 3),
    retourne le nom_dossier prédit et le score de confiance.

    Paramètre :
        img_array : tableau numpy shape (224, 224, 3), valeurs [0, 255]

    Retour :
        (nom_dossier, confiance)
        Ex: ("mango", 0.9134)
    """
    # Normalisation identique à celle du générateur d'entraînement
    img_array = img_array.astype('float32') / 255.0

    # Ajout de la dimension batch : (224, 224, 3) → (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Prédiction : retourne un tableau de probabilités pour chaque classe
    predictions = model.predict(img_batch, verbose=0)  # shape (1, 30)

    # Récupération de l'indice avec la probabilité la plus haute
    indice     = int(np.argmax(predictions[0]))
    confiance  = float(predictions[0][indice])
    nom_dossier = idx_to_class.get(str(indice), "inconnu")

    return nom_dossier, confiance
