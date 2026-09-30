"""
╔══════════════════════════════════════════════════════════════════════╗
║  train_model.py — Entraînement, validation, test & sauvegarde       ║
║                                                                      ║
║  Exécuter APRÈS :                                                    ║
║    1. step2_prepare_data.py  (dataset_prepare/ doit exister)         ║
║    2. step3_build_generators.py (vérification des générateurs)       ║
║                                                                      ║
║  Ce script produit :                                                 ║
║    • data/modele_final.h5         ← modèle sauvegardé               ║
║    • data/class_indices.json      ← correspondance classe → indice   ║
║    • data/training_history.png    ← courbes accuracy / loss          ║
║                                                                      ║
║  Usage :                                                             ║
║    python scripts/train_model.py                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # pas d'affichage GUI, sauvegarde directe en fichier
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Import de la config partagée ──────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, AUGMENTATION, BASE_DIR,
)

# ── Chemins de sortie ─────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'modele_final.h5')
INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')
HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'training_history.png')
BEST_PATH    = os.path.join(BASE_DIR, 'model', 'modele_best.h5')

NB_CLASSES = 30
EPOCHS     = 50  

# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Générateurs de données
# ══════════════════════════════════════════════════════════════════════
def build_generators():
    """
    Crée trois générateurs :
      - train : avec augmentation (rotation, zoom, flip...)
      - val   : normalisation uniquement (mesure réelle des performances)
      - test  : normalisation uniquement, shuffle=False (évaluation finale)
    """
    print("\n[1/5] Création des générateurs...")

    train_datagen = ImageDataGenerator(rescale=1./255, **AUGMENTATION)
    eval_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    nb_classes = train_gen.num_classes
    print(f"  Classes détectées : {nb_classes}")
    print(f"  Train  : {train_gen.samples} images")
    print(f"  Val    : {val_gen.samples} images")
    print(f"  Test   : {test_gen.samples} images")

    if nb_classes != NB_CLASSES:
        print(f"\n[AVERTISSEMENT] {nb_classes} classes détectées, {NB_CLASSES} attendues.")
        print("  Vérifiez que data/raw/ contient bien 30 dossiers non vides.\n")

    return train_gen, val_gen, test_gen


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Construction du modèle (Transfer Learning)
# ══════════════════════════════════════════════════════════════════════
def build_model(nb_classes: int) -> Model:
    """
    Construit le modèle en deux parties :
      1. Base MobileNetV2 pré-entraînée sur ImageNet (couches gelées)
      2. Tête de classification personnalisée pour nos 30 espèces
    """
    print("\n[2/5] Construction du modèle...")

    # ── Base pré-entraînée ────────────────────────────────────────────
    base_model = MobileNetV2(
        weights='imagenet',      # poids pré-entraînés sur 1,4M d'images
        include_top=False,       # on retire la tête de classification ImageNet
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # gèle toutes les couches de la base

    # ── Tête de classification personnalisée ──────────────────────────
    x = base_model.output
    x = GlobalAveragePooling2D()(x)   # réduit la sortie en vecteur 1D
    x = Dense(128, activation='relu')(x)  # couche dense d'apprentissage
    x = Dropout(0.5)(x)               # évite le surapprentissage (overfitting)
    predictions = Dense(nb_classes, activation='softmax')(x)  # 30 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    # ── Compilation ───────────────────────────────────────────────────
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params     = model.count_params()
    trainable_params = sum([
        np.prod(v.shape) for v in model.trainable_variables
    ])
    print(f"  Paramètres totaux      : {total_params:,}")
    print(f"  Paramètres entraînables: {trainable_params:,}")
    print(f"  Paramètres gelés       : {total_params - trainable_params:,}")

    return model


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement avec callbacks
# ══════════════════════════════════════════════════════════════════════
def train_model(model, train_gen, val_gen):
    """
    Entraîne le modèle avec trois callbacks :
      - EarlyStopping    : arrête si val_accuracy ne s'améliore plus
      - ModelCheckpoint  : sauvegarde le meilleur modèle automatiquement
      - ReduceLROnPlateau: réduit le taux d'apprentissage si stagnation
    """
    print(f"\n[3/5] Entraînement ({EPOCHS} epochs max)...")

    callbacks = [
        # Arrête l'entraînement si val_accuracy ne progresse plus sur 5 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Sauvegarde automatiquement le meilleur modèle rencontré
        ModelCheckpoint(
            filepath=BEST_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Divise le learning rate par 2 si val_loss stagne sur 3 epochs
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=callbacks,
    )

    return history


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Évaluation finale sur le jeu de test
# ══════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_gen):
    """
    Évalue le modèle sur le jeu de test (données jamais vues).
    C'est la mesure de performance réelle du modèle final.
    """
    print("\n[4/5] Évaluation sur le jeu de test...")

    loss, accuracy = model.evaluate(test_gen, steps=len(test_gen), verbose=1)

    print(f"\n  {'─'*40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  {'─'*40}")

    return loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Sauvegarde du modèle, des indices et des courbes
# ══════════════════════════════════════════════════════════════════════
def save_all(model, history, train_gen):
    """
    Sauvegarde trois fichiers essentiels :
      1. modele_final.h5      → chargé par l'API pour les prédictions
      2. class_indices.json   → traduit l'indice prédit en nom d'espèce
      3. training_history.png → courbes pour le rapport
    """
    print("\n[5/5] Sauvegarde...")

    # ── 1. Modèle complet ─────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"  Modèle sauvegardé       : {MODEL_PATH}")

    # ── 2. Correspondance indice → nom de dossier ─────────────────────
    # Inverse le dict {nom: indice} en {indice: nom}
    # Ex: {'mango': 4} devient {'4': 'mango'}
    # Utilisé par predict.py pour traduire la prédiction en nom d'espèce
    class_indices = train_gen.class_indices          # {'banana': 0, 'mango': 1, ...}
    idx_to_class  = {str(v): k for k, v in class_indices.items()}

    with open(INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"  Indices des classes     : {INDICES_PATH}")

    # ── 3. Courbes d'apprentissage ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    # Courbe Accuracy
    ax1.plot(epochs_range, history.history['accuracy'],     label='Train')
    ax1.plot(epochs_range, history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy par epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Courbe Loss
    ax2.plot(epochs_range, history.history['loss'],     label='Train')
    ax2.plot(epochs_range, history.history['val_loss'], label='Validation')
    ax2.set_title('Loss par epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(HISTORY_PATH, dpi=150)
    plt.close()
    print(f"  Courbes d'apprentissage : {HISTORY_PATH}")


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ENTRAÎNEMENT DU MODÈLE — 30 espèces végétales")
    print("="*60)

    # Vérification des dossiers avant de commencer
    for d, nom in [(TRAIN_DIR, 'train'), (VAL_DIR, 'val'), (TEST_DIR, 'test')]:
        if not os.path.exists(d):
            print(f"\n[ERREUR] Dossier '{nom}' introuvable : {d}")
            print("  → Exécutez d'abord step2_prepare_data.py")
            exit(1)

    train_gen, val_gen, test_gen = build_generators()
    model                        = build_model(nb_classes=train_gen.num_classes)
    history                      = train_model(model, train_gen, val_gen)
    loss, accuracy               = evaluate_model(model, test_gen)
    save_all(model, history, train_gen)

    print("\n" + "="*60)
    print(f"  Entraînement terminé")
    print(f"  Précision finale sur le test : {accuracy*100:.2f}%")
    print(f"  Modèle prêt : {MODEL_PATH}")
    print("="*60 + "\n")