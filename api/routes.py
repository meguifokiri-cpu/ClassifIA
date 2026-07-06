from flask import Blueprint, jsonify, request
from db import get_db

# Blueprint = groupe de routes enregistré dans app.py
plants_bp = Blueprint('plants', __name__)


# ──────────────────────────────────────────────────────────────
# GET /plants
# Retourne la liste complète des 30 espèces avec leurs infos
# ──────────────────────────────────────────────────────────────
@plants_bp.route('/plants', methods=['GET'])
def get_all_plants():
    conn = get_db()
    rows = conn.execute('''
        SELECT s.id_espece, s.nom_dossier, s.nom_commun, s.nom_latin,
               s.image_ref,
               i.famille, i.utilite_principale, i.description_wikipedia,
               i.avertissement_securite, i.hauteur, i.besoin_eau,
               i.exposition, i.saison_floraison, i.origine
        FROM   species s
        LEFT JOIN species_info i ON s.id_espece = i.id_espece
        ORDER BY s.nom_commun
    ''').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ──────────────────────────────────────────────────────────────
# GET /plants/<id>
# Retourne la fiche complète d'une espèce par son id
# ──────────────────────────────────────────────────────────────
@plants_bp.route('/plants/<int:espece_id>', methods=['GET'])
def get_plant(espece_id):
    conn = get_db()
    row  = conn.execute('''
        SELECT s.id_espece, s.nom_dossier, s.nom_commun, s.nom_latin,
               s.image_ref,
               i.famille, i.utilite_principale, i.description_wikipedia,
               i.avertissement_securite, i.hauteur, i.besoin_eau,
               i.exposition, i.saison_floraison, i.origine
        FROM   species s
        LEFT JOIN species_info i ON s.id_espece = i.id_espece
        WHERE  s.id_espece = ?
    ''', (espece_id,)).fetchone()
    conn.close()

    if row is None:
        return jsonify({'error': 'Espèce non trouvée'}), 404
    return jsonify(dict(row))


# ──────────────────────────────────────────────────────────────
# GET /plants/<id>/image
# Retourne uniquement l'image de référence d'une espèce
# Utilisé par le frontend pour afficher l'image après prédiction
# ──────────────────────────────────────────────────────────────
@plants_bp.route('/plants/<int:espece_id>/image', methods=['GET'])
def get_plant_image(espece_id):
    conn = get_db()
    row  = conn.execute(
        'SELECT image_ref FROM species WHERE id_espece = ?',
        (espece_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return jsonify({'error': 'Espèce non trouvée'}), 404
    if row['image_ref'] is None:
        return jsonify({'error': 'Aucune image de référence disponible'}), 404
    return jsonify({'id_espece': espece_id, 'image_ref': row['image_ref']})


# ──────────────────────────────────────────────────────────────
# GET /plants/search?q=mango
# Recherche par nom_commun ou nom_dossier (insensible à la casse)
# ──────────────────────────────────────────────────────────────
@plants_bp.route('/plants/search', methods=['GET'])
def search_plants():
    q = request.args.get('q', '').strip()

    if not q:
        return jsonify({'error': 'Paramètre q manquant. Ex: /plants/search?q=mango'}), 400

    motif = f'%{q}%'
    conn  = get_db()
    rows  = conn.execute('''
        SELECT s.id_espece, s.nom_dossier, s.nom_commun, s.nom_latin,
               s.image_ref,
               i.famille, i.utilite_principale
        FROM   species s
        LEFT JOIN species_info i ON s.id_espece = i.id_espece
        WHERE  s.nom_commun  LIKE ? COLLATE NOCASE
           OR  s.nom_dossier LIKE ? COLLATE NOCASE
        ORDER BY s.nom_commun
    ''', (motif, motif)).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


# ──────────────────────────────────────────────────────────────
# GET /plants/famille/<famille>
# Filtre toutes les espèces d'une même famille botanique
# ──────────────────────────────────────────────────────────────
@plants_bp.route('/plants/famille/<string:famille>', methods=['GET'])
def get_by_famille(famille):
    conn = get_db()
    rows = conn.execute('''
        SELECT s.id_espece, s.nom_dossier, s.nom_commun, s.nom_latin,
               s.image_ref,
               i.famille, i.utilite_principale, i.origine
        FROM   species s
        LEFT JOIN species_info i ON s.id_espece = i.id_espece
        WHERE  i.famille LIKE ? COLLATE NOCASE
        ORDER BY s.nom_commun
    ''', (f'%{famille}%',)).fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])
