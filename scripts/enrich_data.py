import urllib.request
import urllib.parse
import json
import time

# 1. Dictionnaire pour cibler les bons articles et éviter les homonymies
targets = {
    # Format : "nom_du_dossier": "Titre_Exact_Wikipedia"
    "orange": "Orange_(fruit)",
    "melon": "Melon_(plante)",
    "ginger": "Gingembre",
    "corn": "Maïs",
    "paddy": "Riz",
    "eggplant": "Aubergine",
    "cucumber": "Concombre",
    "shallot": "Échalote",
    "soybeans": "Soja",
    "spinach": "Épinard",
    "sweet_potatoes": "Patate_douce",
    "waterappel": "Syzygium samarangense", 
    "watermelon": "Pastèque",
    "cassava": "Manioc",
    "pineapple": "Ananas",
    "tobacco": "Tabac",
    "banana": "Banane",
    "guava": "Goyave",
    "mango": "Mangue",
    "coconuts": "Noix_de_coco",
    "curcuma": "Curcuma",
    "bilimbi": "Averrhoa bilimbi",   
    "aloe_vera": "Aloe_vera",
    "cantaloupe": "Melon_de_cantaloup",
    "pomelo": "Pomelo",
    "pepper_chili": "Piment_de_cayenne",
    "kale": "Chou frisé",    
    "galangal": "Alpinia galanga",   
    "longbean": "Haricot_vert",
    "papaya": "Papaye"
}

# 2. Identité obligatoire pour Wikipedia (User-Agent)
headers = {'User-Agent': 'MonProjetBotanique/1.0 (contact@exemple.com)'}

results = []

print("Début de l'extraction...")

for folder_name, wiki_title in targets.items():
    # URL modifiée : prop=extracts & explaintext pour avoir du texte brut
    # exsentences=10 pour avoir plus d'informations par plante
    wiki_title_encoded = urllib.parse.quote(wiki_title)
    url = f"https://fr.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext&exsentences=10&titles={wiki_title_encoded}&formatversion=2"
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            # On récupère la page extraite
            page = data["query"]["pages"]
            results.append(page)
            print(f" {wiki_title} extrait avec succès.")
    except Exception as e:
        print(f" Erreur pour {wiki_title} : {e}")
    
    # Pause courte pour ne pas être bloqué par le serveur
    time.sleep(0.5)

# 3. Sauvegarde finale
with open("wiki_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\nFini ! Le fichier wiki_data.json contient maintenant les textes nettoyés.")