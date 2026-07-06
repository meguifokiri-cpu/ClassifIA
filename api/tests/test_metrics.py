"""
test_metrics.py — Vérifie que l'endpoint /metrics expose bien
les métriques Prometheus attendues, au bon format.
"""

def test_metrics_endpoint_repond_200(client):
    reponse = client.get('/metrics')
    assert reponse.status_code == 200


def test_metrics_contient_les_compteurs_attendus(client):
    reponse = client.get('/metrics')
    contenu = reponse.get_data(as_text=True)

    # On vérifie la présence des noms de métriques définis dans monitoring.py
    assert "plantid_predictions_total" in contenu
    assert "plantid_prediction_errors_total" in contenu
    assert "plantid_prediction_latency_seconds" in contenu
    assert "plantid_model_classes" in contenu


def test_metrics_content_type_est_prometheus(client):
    reponse = client.get('/metrics')
    assert "text/plain" in reponse.content_type
