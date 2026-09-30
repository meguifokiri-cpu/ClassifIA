"""
╔══════════════════════════════════════════════════════════════╗
║  ÉTAPE 3 — Création des générateurs de données               ║
║  Exécuter APRÈS step2_prepare_data.py                        ║
╚══════════════════════════════════════════════════════════════╝

Ce script :
  • Crée le générateur d'entraînement AVEC augmentation
  • Crée les générateurs val et test SANS augmentation
  • Vérifie que les 3 générateurs voient bien les 30 classes
  • Affiche un exemple de batch pour contrôle visuel

Usage :
    python step3_build_generators.py

Pour réutiliser les générateurs dans un autre script :
    from step3_build_generators import build_generators
    train_gen, val_gen, test_gen = build_generators()
"""

import os
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, AUGMENTATION,
)


def build_generators():
    """
    Construit et retourne (train_gen, val_gen, test_gen).
    Lève une erreur si les dossiers sont absents.
    """
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not os.path.exists(split_dir):
            print(f"[ERREUR] Dossier manquant : {split_dir}")
            print("  → Exécutez d'abord step2_prepare_data.py")
            sys.exit(1)

    # ── Générateur train : normalisation + augmentation ──────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        **AUGMENTATION,
    )

    # ── Générateurs val / test : normalisation uniquement ────────
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,   # important : ne pas mélanger pour les métriques
    )

    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,   # important : ne pas mélanger pour les métriques
    )

    return train_gen, val_gen, test_gen


def afficher_rapport(train_gen, val_gen, test_gen) -> None:
    """Affiche un résumé lisible des générateurs."""
    print(f"\n{'─'*55}")
    print("  Générateurs créés")
    print(f"{'─'*55}")
    print(f"  Classes      : {train_gen.num_classes}")
    print(f"  Train        : {train_gen.samples} images  ({len(train_gen)} batchs)")
    print(f"  Val          : {val_gen.samples} images  ({len(val_gen)} batchs)")
    print(f"  Test         : {test_gen.samples} images  ({len(test_gen)} batchs)")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Image size   : {IMG_SIZE}")
    print(f"{'─'*55}")
    print(f"\n  Correspondance classes → indices :")
    for nom, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"    {idx:>2}  {nom}")
    print()

    if train_gen.num_classes != 30:
        print(f"[AVERTISSEMENT] {train_gen.num_classes} classes détectées (attendu : 30)")
        print("  → Vérifiez que data/raw/ contient bien 30 sous-dossiers non vides.\n")


if __name__ == "__main__":
    train_gen, val_gen, test_gen = build_generators()
    afficher_rapport(train_gen, val_gen, test_gen)
    print("[OK] Générateurs prêts — lancez votre script d'entraînement (step4+)\n")
