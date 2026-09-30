"""
monitoring.py — Instrumentation Prometheus de l'API PlantID
Responsabilité unique : définir les métriques exposées sur /metrics
et fournir des fonctions utilitaires pour les alimenter.

Importé par predict.py, model_loader.py et app.py.

Pourquoi Prometheus ?
→ C'est le standard de facto pour le monitoring d'applications en
  production. Un serveur Prometheus peut scraper (interroger) l'URL
  /metrics à intervalle régulier, et Grafana peut ensuite construire
  des dashboards à partir de ces données historisées.

Métriques exposées :
  - plantid_predictions_total        : compteur de prédictions par espèce
  - plantid_prediction_errors_total  : compteur d'erreurs par type
  - plantid_prediction_latency_seconds : histogramme du temps de réponse
  - plantid_prediction_confidence    : histogramme des scores de confiance
  - plantid_model_classes            : nombre de classes du modèle chargé
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ── Compteur : nombre de prédictions réussies, par espèce identifiée ──
PREDICTION_COUNT = Counter(
    'plantid_predictions_total',
    'Nombre total de prédictions effectuées, par espèce',
    ['espece']
)

# ── Compteur : erreurs rencontrées sur /predict, par type ─────────────
# Types possibles : 'fichier_manquant', 'format_invalide',
#                   'lecture_image_echouee', 'espece_absente_bdd'
PREDICTION_ERRORS = Counter(
    'plantid_prediction_errors_total',
    "Nombre total d'erreurs lors des prédictions, par type",
    ['type_erreur']
)

# ── Histogramme : temps de réponse de l'endpoint /predict ────────────
# Buckets adaptés à un modèle CPU (quelques centaines de ms à ~2s)
PREDICTION_LATENCY = Histogram(
    'plantid_prediction_latency_seconds',
    "Temps de réponse de l'endpoint /predict",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
)

# ── Histogramme : distribution des scores de confiance (en %) ────────
# Utile pour surveiller la dérive du modèle : si la confiance moyenne
# baisse dans le temps, ça peut signaler des images hors distribution
# d'entraînement (nouvelles espèces, mauvais éclairage, etc.)
PREDICTION_CONFIDENCE = Histogram(
    'plantid_prediction_confidence_percent',
    'Distribution des scores de confiance du modèle (en %)',
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
)

# ── Jauge : nombre de classes chargées (vérifie l'intégrité du modèle) ─
MODEL_CLASSES_LOADED = Gauge(
    'plantid_model_classes',
    'Nombre de classes disponibles dans le modèle chargé'
)


def metrics_response():
    """
    Génère le corps de la réponse HTTP pour l'endpoint /metrics,
    au format texte attendu par un serveur Prometheus.
    Retourne (contenu, content_type) — à utiliser dans une Response Flask.
    """
    return generate_latest(), CONTENT_TYPE_LATEST