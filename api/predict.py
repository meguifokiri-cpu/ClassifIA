"""
predict.py — Endpoint POST /predict
Responsabilité unique : recevoir une image, appeler le modèle,
interroger la base de données et retourner la fiche complète.

C'est le pont entre le modèle IA et la base de données.
Importé par app.py, jamais exécuté directement.

Flux complet :
    Image reçue
        → redimensionnement 224×224
        → model_loader.predict_species()  → nom_dossier + confiance
        → requête SQL sur flowers.db      → fiche botanique complète
        → réponse JSON au frontend
"""

import io
import numpy as np
from PIL import Image

from flask import Blueprint, jsonify, request
from db import get_db
from model_loader import predict_species

predict_bp = Blueprint('predict', __name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
IMG_SIZE           = (224, 224)


def allowed_file(filename: str) -> bool:
    """Vérifie que le fichier envoyé est bien une image supportée."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(file_bytes: bytes) -> np.ndarray:
    """
    Convertit les bytes reçus en tableau numpy 224×224×3.
    Gère les images RGBA (PNG avec transparence) en les convertissant en RGB.
    """
    img = Image.open(io.BytesIO(file_bytes))

    # Conversion RGBA → RGB si nécessaire (ex: PNG avec transparence)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Redimensionnement à 224×224 avec interpolation NEAREST
    # → aligné sur l'interpolation utilisée par flow_from_directory()
    #   pendant l'entraînement (Keras utilise 'nearest' par défaut)
    img = img.resize(IMG_SIZE, Image.NEAREST)

    return np.array(img)   # shape (224, 224, 3), valeurs [0, 255]

# ══════════════════════════════════════════════════════════════════════
# POST /predict
# Reçoit une image, retourne la fiche botanique de l'espèce prédite
# ══════════════════════════════════════════════════════════════════════

# Les seuils restent en dehors de la fonction (constantes globales) — ça c'est correct
CONFIDENCE_THRESHOLD   = 0.55
ENTROPY_THRESHOLD      = 0.15
MARGIN_THRESHOLD       = 0.70
MAHALANOBIS_THRESHOLD  = 52.0

@predict_bp.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({'error': "Champ 'image' manquant dans la requête"}), 400

    fichier = request.files['image']

    if fichier.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    if not allowed_file(fichier.filename):
        return jsonify({
            'error': f"Format non supporté. Formats acceptés : {ALLOWED_EXTENSIONS}"
        }), 400

    try:
        file_bytes = fichier.read()
        img_array  = prepare_image(file_bytes)
    except Exception as e:
        return jsonify({'error': f"Impossible de lire l'image : {str(e)}"}), 500

    # ── Prédiction via le modèle ──────────────────────────────────────
    nom_dossier, confiance, diagnostics = predict_species(img_array)

    # ── Filtre OOD Couche 1+2 : rejet multi-signal ─────────────────────
    est_rejete = (
        confiance < CONFIDENCE_THRESHOLD or
        diagnostics["entropie"] > ENTROPY_THRESHOLD or
        diagnostics["ecart_top1_top2"] < MARGIN_THRESHOLD or
        diagnostics["distance_mahalanobis"] > MAHALANOBIS_THRESHOLD
    )

    if est_rejete:
        return jsonify({
            'error': "Cette image ne semble pas correspondre à une plante.",
            'suggestion_ignoree': nom_dossier,
            'confiance': round(confiance * 100, 2),
            'diagnostics': diagnostics
        }), 422

    # ── Classe "autre_espece_vegetale" : plante réelle, hors référentiel ──
    if nom_dossier == 'autre_espece_vegetale':
        message = ("Il s'agit bien d'une plante, mais elle ne fait pas partie "
                   "des 30 espèces du référentiel.")
        return jsonify({
            'espece_reconnue'      : False,
            'nom_commun'           : 'Plante non répertoriée',
            'description_wikipedia': message,
            'confiance'            : round(confiance * 100, 2),
            'diagnostics'          : diagnostics
        }), 200

    # ── Récupération de la fiche en base de données ───────────────────
    conn = get_db()
    row  = conn.execute('''
        SELECT s.id_espece, s.nom_dossier, s.nom_commun, s.nom_latin,
               s.image_ref,
               i.famille, i.utilite_principale, i.description_wikipedia,
               i.avertissement_securite, i.hauteur, i.besoin_eau,
               i.exposition, i.saison_floraison, i.origine
        FROM   species s
        LEFT JOIN species_info i ON s.id_espece = i.id_espece
        WHERE  s.nom_dossier = ?
    ''', (nom_dossier,)).fetchone()
    conn.close()

    if row is None:
        return jsonify({
            'error'      : f"Espèce '{nom_dossier}' non trouvée en base de données",
            'nom_dossier': nom_dossier,
            'confiance'  : round(confiance * 100, 2)
        }), 404

    resultat = dict(row)
    resultat['confiance'] = round(confiance * 100, 2)

    return jsonify(resultat), 200