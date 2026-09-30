import os
from flask import Flask, Response
from flask_cors import CORS
from routes import plants_bp
from predict import predict_bp
from monitoring import metrics_response

# ── Création de l'application ────────────────────────────────
app = Flask(__name__)

# ── CORS : autorise les requêtes depuis le frontend (phase E3) ─
CORS(app)

# ── Enregistrement des routes définies dans routes.py ─────────
app.register_blueprint(plants_bp)
app.register_blueprint(predict_bp)


# ── GET /metrics — exposition des métriques Prometheus ────────
# À scraper par un serveur Prometheus, ou à consulter directement
# dans un navigateur pour vérifier que les compteurs s'incrémentent.
@app.route('/metrics')
def metrics():
    contenu, content_type = metrics_response()
    return Response(contenu, mimetype=content_type)


@app.route('/health')
def health():
    return {"status": "ok"}, 200


# ── Lancement du serveur ──────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n API démarrée sur http://localhost:{port}")
    print(" Endpoints : /plants | /plants/<id> | /plants/<id>/image")
    print("             /plants/search?q= | /plants/famille/<famille>")
    print("             /predict | /metrics\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)