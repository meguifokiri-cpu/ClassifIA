# check_chemins.py — à lancer depuis la racine du projet
import os
import re

PATTERN_CHEMIN_ABSOLU = re.compile(r'[A-Za-z]:[\\/][^\s"\']*')

for racine, dossiers, fichiers in os.walk('.'):
    if 'venv' in racine or '.git' in racine:
        continue
    for f in fichiers:
        if f.endswith(('.py', '.env', '.yml', '.yaml', '.json')):
            chemin = os.path.join(racine, f)
            try:
                with open(chemin, encoding='utf-8', errors='ignore') as fichier:
                    for i, ligne in enumerate(fichier, 1):
                        if PATTERN_CHEMIN_ABSOLU.search(ligne):
                            print(f"{chemin}:{i} → {ligne.strip()}")
            except Exception:
                pass