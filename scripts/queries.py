import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ajuste le nombre de dirname selon la profondeur réelle de queries.py
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'dataset_prepare', 'train')
RAW_DIR   = os.path.join(BASE_DIR, 'data', 'raw')

# ancien nom → nouveau nom (celui dans la base)
CORRECTIONS = {
    'coconut'       : 'coconuts',
    'longbeans'     : 'longbean',
    'peper chili'   : 'pepper_chili',
    'sweet potatoes': 'sweet_potatoes',
    'waterapple'    : 'waterappel',
}

for base_dir in (TRAIN_DIR, RAW_DIR):
    print(f"\nDossier : {base_dir}")
    for ancien, nouveau in CORRECTIONS.items():
        ancien_path = os.path.join(base_dir, ancien)
        nouveau_path = os.path.join(base_dir, nouveau)
        if os.path.exists(ancien_path):
            os.rename(ancien_path, nouveau_path)
            print(f"  ✓ '{ancien}' → '{nouveau}'")
        else:
            print(f"  — '{ancien}' introuvable (déjà renommé ?)")