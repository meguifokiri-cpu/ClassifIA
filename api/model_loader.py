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
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from monitoring import MODEL_CLASSES_LOADED

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'modele_best.h5')
INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

if not os.path.exists(INDICES_PATH):
    raise FileNotFoundError(f"Indices introuvables : {INDICES_PATH}")

print("  Chargement du modèle IA...")
model = load_model(MODEL_PATH)          # ← 'model' est défini ICI

with open(INDICES_PATH, encoding='utf-8') as f:
    idx_to_class = json.load(f)

print(f"  Modèle chargé — {len(idx_to_class)} classes disponibles")
MODEL_CLASSES_LOADED.set(len(idx_to_class))

# ── Extracteur de features pour Mahalanobis ────────────────────
# DOIT être placé APRÈS "model = load_model(...)", jamais avant
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
with open(os.path.join(BASE_DIR, 'model', 'ood_stats.pkl'), 'rb') as f:
    ood_stats = pickle.load(f)


def distance_mahalanobis_min(img_batch: np.ndarray) -> float:
    """
    Calcule la distance de Mahalanobis entre l'embedding de l'image
    et le cluster de classe connu le plus proche.
    Une distance élevée = l'image "ressemble" statistiquement peu
    à tout ce que le modèle a appris, même si le softmax est confiant.
    """
    embedding = feature_extractor.predict(img_batch, verbose=0)[0]  # (128,)
    cov_inv = ood_stats["cov_inv"]

    distances = []
    for classe, moyenne in ood_stats["moyennes"].items():
        diff = embedding - moyenne
        d = np.sqrt(diff @ cov_inv @ diff.T)
        distances.append(d)

    return float(min(distances))



# ── Fonction de prédiction ────────────────────────────────────────────
IMG_SIZE = (224, 224)

def predict_species(img_array: np.ndarray) -> tuple[str, float, dict]:
    img_array = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch, verbose=0)[0]

    indice = int(np.argmax(predictions))
    confiance = float(predictions[indice])
    nom_dossier = idx_to_class.get(str(indice), "inconnu")

    eps = 1e-9
    entropie = -np.sum(predictions * np.log(predictions + eps))
    entropie_normalisee = float(entropie / np.log(len(predictions)))

    top2_idx = np.argsort(predictions)[-2:][::-1]
    ecart_top1_top2 = float(predictions[top2_idx[0]] - predictions[top2_idx[1]])

    distance_mahalanobis = distance_mahalanobis_min(img_batch)

    diagnostics = {
        "entropie": round(entropie_normalisee, 4),
        "ecart_top1_top2": round(ecart_top1_top2, 4),
        "distance_mahalanobis": round(distance_mahalanobis, 2),
    }

    return nom_dossier, confiance, diagnostics