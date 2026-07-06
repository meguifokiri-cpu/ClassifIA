import os

TRAIN_DIR = r"C:\Users\user\Desktop\classif-ia\data\dataset_prepare\train"
RAW_DIR   = r"C:\Users\user\Desktop\classif-ia\data\raw"

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