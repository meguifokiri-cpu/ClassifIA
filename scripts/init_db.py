import sqlite3
import os
import json

# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH        = os.path.join(BASE_DIR, 'data', 'flowers.db')
WIKI_JSON_PATH = os.path.join(BASE_DIR, 'wiki_data_clean.json')       
BOT_JSON_PATH  = os.path.join(BASE_DIR, 'botanical_data.json')  


# ============================================================
WIKI_TITLE_MAP = {
    "Orange (fruit)"        : "orange",
    "Melon (plante)"        : "melon",
    "Gingembre"             : "ginger",
    "Maïs"                  : "corn",
    "Riz"                   : "paddy",
    "Aubergine"             : "eggplant",
    "Concombre"             : "cucumber",
    "Échalote"              : "shallot",
    "Soja"                  : "soybeans",
    "Épinard"               : "spinach",
    "Patate douce"          : "sweet_potatoes",
    "Syzygium samarangense" : "waterappel",
    "Pastèque"              : "watermelon",
    "Manioc"                : "cassava",
    "Ananas"                : "pineapple",
    "Tabac"                 : "tobacco",
    "Banane"                : "banana",
    "Goyave"                : "guava",
    "Mangue"                : "mango",
    "Noix de coco"          : "coconuts",
    "Curcuma"               : "curcuma",
    "Bilibili"              : "bilimbi",
    "Aloe vera"             : "aloe_vera",
    "Melon de cantaloup"    : "cantaloupe",
    "Pomelo"                : "pomelo",
    "Piment de cayenne"     : "pepper_chili",
    "Chou frisé"            : "kale",
    "Alpinia galanga"       : "galangal",
    "Haricot vert"          : "longbean",
    "Papaye"                : "papaya",
}


def create_database():
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS species (
            id_espece   INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_dossier TEXT UNIQUE NOT NULL,
            nom_commun  TEXT NOT NULL,
            nom_latin   TEXT,
            image_ref   TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS species_info (
            id_info                 INTEGER PRIMARY KEY AUTOINCREMENT,
            id_espece               INTEGER NOT NULL,
            famille                 TEXT,
            utilite_principale      TEXT,
            description_wikipedia   TEXT,
            avertissement_securite  TEXT,
            hauteur                 TEXT,
            besoin_eau              TEXT,
            exposition              TEXT,
            saison_floraison        TEXT,
            origine                 TEXT,
            FOREIGN KEY (id_espece) REFERENCES species (id_espece)
        )
    ''')

    conn.commit()
    print(" Les 2 tables ont été créées.")
    return conn, cursor


# ============================================================
# ÉTAPE 2 — Chargement des deux fichiers JSON
# ============================================================
def load_botanical_data():
    """Charge botanical_data.json → dict { nom_dossier: {...} }"""
    if not os.path.exists(BOT_JSON_PATH):
        raise FileNotFoundError(f" Fichier introuvable : {BOT_JSON_PATH}")

    with open(BOT_JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)

    print(f" {len(data)} espèces chargées depuis botanical_data.json")
    return data


def load_wiki_descriptions():
    descriptions = {}

    if not os.path.exists(WIKI_JSON_PATH):
        print(f"  Fichier introuvable : {WIKI_JSON_PATH}")
        return descriptions

    with open(WIKI_JSON_PATH, encoding='utf-8') as f:
        wiki_data = json.load(f)

    for entry in wiki_data:
        # ── CORRECTION : entry est maintenant un dict directement ──
        titre       = entry.get("titre", "")
        text        = entry.get("extrait", "")
        nom_dossier = WIKI_TITLE_MAP.get(titre)

        if nom_dossier:
            descriptions[nom_dossier] = text[:500].strip()
        else:
            print(f"    Titre non mappé : '{titre}'")

    print(f" {len(descriptions)} descriptions Wikipedia chargées.")
    return descriptions


# ============================================================
# ÉTAPE 3 — Insertion dans species + species_info
# ============================================================
def insert_all_data(conn, cursor, botanical_data, descriptions):
    inserted_species = 0
    inserted_info    = 0

    for nom_dossier, data in botanical_data.items():

        # ── TABLE 1 : species ──────────────────────────────
        cursor.execute('''
            INSERT OR IGNORE INTO species (nom_dossier, nom_commun, nom_latin)
            VALUES (?, ?, ?)
        ''', (nom_dossier, data["nom_commun"], data["nom_latin"]))

        cursor.execute(
            "SELECT id_espece FROM species WHERE nom_dossier = ?",
            (nom_dossier,)
        )
        id_espece = cursor.fetchone()[0]

        # ── TABLE 2 : species_info ─────────────────────────
        cursor.execute(
            "SELECT COUNT(*) FROM species_info WHERE id_espece = ?",
            (id_espece,)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO species_info (
                    id_espece, famille, utilite_principale,
                    description_wikipedia, avertissement_securite,
                    hauteur, besoin_eau, exposition,
                    saison_floraison, origine
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                id_espece,
                data["famille"],
                data["utilite_principale"],
                descriptions.get(nom_dossier),
                data["avertissement_securite"],
                data["hauteur"],
                data["besoin_eau"],
                data["exposition"],
                data["saison_floraison"],
                data["origine"]
            ))
            inserted_info += 1

        inserted_species += 1

    conn.commit()
    print(f" {inserted_species} espèces insérées dans 'species'.")
    print(f" {inserted_info} lignes insérées dans 'species_info'.")


# ============================================================
# ÉTAPE 4 — Image de référence par espèce (depuis train/)
# ============================================================
def insert_image_ref(conn, cursor):
    train_dir = os.path.join(BASE_DIR, 'data', 'dataset_prepare', 'train')

    if not os.path.exists(train_dir):
        print(f"  Dossier train introuvable : {train_dir}")
        print("  → Exécutez d'abord step2_prepare_data.py")
        return

    updated = 0

    for espece in sorted(os.listdir(train_dir)):
        dossier = os.path.join(train_dir, espece)
        if not os.path.isdir(dossier):
            continue

        images = [
            f for f in os.listdir(dossier)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        if not images:
            continue

        # Première image du dossier train comme référence
        image_ref = os.path.join('data', 'dataset_prepare', 'train', espece, images[0])

        cursor.execute(
            "UPDATE species SET image_ref = ? WHERE nom_dossier = ?",
            (image_ref, espece)
        )
        if cursor.rowcount > 0:
            updated += 1
            print(f"  {espece} → {images[0]}")

    conn.commit()
    print(f" {updated} image(s) de référence insérées dans 'species'.")


# ============================================================
# ÉTAPE 5 — Vérification avec requête JOIN (C2)
# ============================================================
def verify_database(conn):
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("   VÉRIFICATION DE LA BASE DE DONNÉES")
    print("="*60)

    for table in ['species', 'species_info']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"   {table:<20} : {cursor.fetchone()[0]} lignes")

    print("\n   Exemple de requête JOIN (2 tables) :")
    cursor.execute('''
        SELECT s.nom_dossier,
               s.nom_commun,
               si.famille,
               s.image_ref
        FROM   species s
        JOIN   species_info si ON s.id_espece = si.id_espece
        LIMIT 5
    ''')
    print(f"\n   {'Dossier':<18} {'Nom commun':<20} {'Famille':<25} {'Image ref'}")
    print("   " + "-"*80)
    for row in cursor.fetchall():
        img = row[3] if row[3] else "—"
        print(f"   {row[0]:<18} {row[1]:<20} {row[2]:<25} {img}")

    print("="*60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    print("\n Initialisation de la base de données flowers.db")
    print("-"*50)

    conn, cursor   = create_database()
    botanical_data = load_botanical_data()
    descriptions   = load_wiki_descriptions()

    insert_all_data(conn, cursor, botanical_data, descriptions)
    insert_image_ref(conn, cursor)
    verify_database(conn)

    conn.close()
    print(f"\n Base de données prête : {DB_PATH}")