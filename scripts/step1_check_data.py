"""
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 1 — Vérification du dossier raw/                      ║
║  Exécuter EN PREMIER avant tout traitement.                   ║
║  Ne modifie aucun fichier.                                    ║
╚═════════════════════════════════════════════════════════════╝

Vérifie que :
  • le dossier data/raw/ existe
  • il contient bien des sous-dossiers (une espèce = un dossier)
  • chaque espèce possède des images dans un format supporté
  • signale les espèces vides ou avec peu d'images (<20)

Usage :
    python step1_check_data.py
"""

import os
import sys

from config import RAW_DIR, EXTENSIONS


def verifier_raw(raw_dir: str) -> list[str]:
    """Parcourt raw_dir et affiche un rapport par espèce."""

    if not os.path.exists(raw_dir):
        print(f"[ERREUR] Dossier introuvable : {raw_dir}")
        print("  → Créez data/raw/ et placez-y vos sous-dossiers d'espèces.")
        sys.exit(1)

    especes = sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    )

    if not especes:
        print(f"[ERREUR] Aucun sous-dossier trouvé dans : {raw_dir}")
        sys.exit(1)

    print(f"\n{'─'*55}")
    print(f"  Dossier raw : {raw_dir}")
    print(f"  Espèces détectées : {len(especes)}")
    print(f"{'─'*55}")

    total = 0
    avertissements = []

    for esp in especes:
        dossier = os.path.join(raw_dir, esp)
        images = [
            f for f in os.listdir(dossier)
            if f.lower().endswith(EXTENSIONS)
        ]
        n = len(images)
        total += n
        flag = ""

        if n == 0:
            flag = "  ← [VIDE]"
            avertissements.append(f"  {esp} : 0 images (espèce ignorée)")
        elif n < 20:
            flag = "  ← [PEU D'IMAGES]"
            avertissements.append(f"  {esp} : seulement {n} images (recommandé ≥ 100)")

        print(f"  {esp:<30} {n:>5} images{flag}")

    print(f"{'─'*55}")
    print(f"  Total                          {total:>5} images")
    print(f"{'─'*55}\n")

    if avertissements:
        print("[AVERTISSEMENTS]")
        for msg in avertissements:
            print(msg)
        print()

    return especes


if __name__ == "__main__":
    especes = verifier_raw(RAW_DIR)
    print(f"[OK] Prêt pour l'étape 2 — {len(especes)} espèces valides.\n")
