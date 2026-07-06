"""
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 2 — Nettoyage, redimensionnement & division           ║
║  Exécuter APRÈS step1_check_data.py                          ║
║  ⚠ Supprime et recrée dataset_prepare/ à chaque exécution.  ║
╚══════════════════════════════════════════════════════════════╝

Ce script :
  • Repart d'une base propre (supprime l'ancien dataset_prepare/)
  • Crée l'arborescence train / val / test pour les 30 espèces
  • Filtre les fichiers non-image et ignore les corrompus
  • Mélange aléatoirement les images (seed fixée pour reproductibilité)
  • Divise en 70 % train / 15 % val / 15 % test
  • Redimensionne chaque image en 224×224 (format MobileNetV2)

Usage :
    python step2_prepare_data.py
"""

import os
import shutil
import random

from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

from config import (
    RAW_DIR, PREPARE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, EXTENSIONS, TRAIN_RATIO, VAL_RATIO,
)

RANDOM_SEED = 42


def creer_arborescence(especes: list[str]) -> None:
    """Supprime l'ancienne structure et recrée train/val/test."""
    if os.path.exists(PREPARE_DIR):
        print("  Suppression de l'ancien dataset_prepare/ ...")
        shutil.rmtree(PREPARE_DIR)

    for esp in especes:
        for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
            os.makedirs(os.path.join(split_dir, esp), exist_ok=True)

    print(f"  Arborescence créée pour {len(especes)} espèces.\n")


def traiter_espece(esp: str) -> dict:
    """
    Pour une espèce :
      1. Filtre les formats valides
      2. Mélange + divise 70/15/15
      3. Redimensionne et sauvegarde chaque image
    Retourne les compteurs par split.
    """
    src_dir = os.path.join(RAW_DIR, esp)
    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(EXTENSIONS)
    ]
    random.shuffle(images)

    n       = len(images)
    n_train = int(TRAIN_RATIO * n)
    n_val   = int(VAL_RATIO * n)

    splits = {
        "train": (images[:n_train],              TRAIN_DIR),
        "val":   (images[n_train:n_train+n_val], VAL_DIR),
        "test":  (images[n_train+n_val:],        TEST_DIR),
    }

    compteurs = {"train": 0, "val": 0, "test": 0, "erreurs": 0}

    for split_name, (fichiers, dest_base) in splits.items():
        for img_name in fichiers:
            src_path  = os.path.join(src_dir, img_name)
            dest_path = os.path.join(dest_base, esp, img_name)
            try:
                img = load_img(src_path, target_size=IMG_SIZE)
                save_img(dest_path, img_to_array(img))
                compteurs[split_name] += 1
            except Exception as e:
                compteurs["erreurs"] += 1
                print(f"    [IGNORÉ] {img_name} : {e}")

    return compteurs


def preparer_dataset() -> None:
    random.seed(RANDOM_SEED)

    especes = sorted(
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    )
    print(f"\n{len(especes)} espèces à traiter...\n")

    creer_arborescence(especes)

    total = {"train": 0, "val": 0, "test": 0, "erreurs": 0}
    col   = max(len(e) for e in especes)

    for esp in especes:
        c = traiter_espece(esp)
        for k in total:
            total[k] += c[k]
        print(
            f"  {esp:<{col}}  "
            f"train={c['train']:>4}  val={c['val']:>3}  test={c['test']:>3}"
            + (f"  erreurs={c['erreurs']}" if c["erreurs"] else "")
        )

    print(f"\n{'─'*55}")
    print(f"  TOTAL  train={total['train']}  val={total['val']}  test={total['test']}")
    if total["erreurs"]:
        print(f"  Images ignorées (corrompues) : {total['erreurs']}")
    print(f"{'─'*55}")
    print("\n[OK] dataset_prepare/ prêt — lancez step3_build_generators.py\n")


if __name__ == "__main__":
    preparer_dataset()
