import requests

def chercher_taxon(nom):
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": nom, "rank": "species", "per_page": 5, "locale": "fr"}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    print(f"\nRecherche : '{nom}'")
    for taxon in data["results"]:
        nom_commun = taxon.get("preferred_common_name", "—")
        nom_scientifique = taxon.get("name", "—")
        nb_observations = taxon.get("observations_count", 0)
        print(f"  {nom_scientifique:30s} ({nom_commun:20s}) — {nb_observations} observations")

# Adapte cette liste à tes espèces choisies
for nom in ["rose", "tulipe", "lierre", "hortensia", "romarin", "monstera", "echeveria", "fougère"]:
    chercher_taxon(nom)