"""
╔══════════════════════════════════════════════════════════════════════╗
║  train_model.py — Entraînement, validation, test & sauvegarde       ║
║                                                                      ║
║  Exécuter APRÈS :                                                    ║
║    1. step2_prepare_data.py  (dataset_prepare/ doit exister)         ║
║    2. step3_build_generators.py (vérification des générateurs)       ║
║                                                                      ║
║  Ce script produit :                                                 ║
║    • model/modele_final.h5       ← modèle sauvegardé                ║
║    • model/modele_best.h5        ← meilleur modèle (checkpoint)     ║
║    • model/class_indices.json    ← correspondance classe → indice   ║
║    • model/training_history.png  ← courbes accuracy / loss          ║
║                                                                      ║
║  Usage :                                                             ║
║    python scripts/train_model.py                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ══════════════════════════════════════════════════════════════════════
# MODIFICATION 1 — Import centralisé depuis config.py
# ──────────────────────────────────────────────────────────────────────
# AVANT : les chemins MODEL_PATH, INDICES_PATH, etc. étaient définis
#         ici dans train_model.py avec os.path.join(BASE_DIR, 'model', ...)
#         → Problème : si BASE_DIR changeait (PC vs Colab), il fallait
#           modifier DEUX fichiers (config.py ET train_model.py).
#
# APRÈS : tous les chemins viennent de config.py qui détecte
#         automatiquement l'environnement (Colab ou PC).
#         → On n'a plus qu'UN seul endroit à modifier.
# ══════════════════════════════════════════════════════════════════════
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, AUGMENTATION,
    MODEL_PATH, BEST_PATH, INDICES_PATH, HISTORY_PATH,
)

# ══════════════════════════════════════════════════════════════════════
# MODIFICATION 2 — Suppression des chemins définis localement
# ──────────────────────────────────────────────────────────────────────
# AVANT : ces 4 lignes existaient ici et pouvaient pointer vers
#         un mauvais dossier si BASE_DIR n'était pas correct :
#
#   MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'modele_final.h5')
#   INDICES_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')
#   HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'training_history.png')
#   BEST_PATH    = os.path.join(BASE_DIR, 'model', 'modele_best.h5')
#
# APRÈS : ces chemins viennent directement de config.py → supprimés ici.
#         De plus, config.py crée automatiquement le dossier model/
#         avec os.makedirs() → class_indices.json sera bien créé.
# ══════════════════════════════════════════════════════════════════════

NB_CLASSES = 30
EPOCHS     = 30

# ══════════════════════════════════════════════════════════════════════
# NOUVEAU — Paramètres de la phase 2 (fine-tuning)
# ──────────────────────────────────────────────────────────────────────
# FT_EPOCHS        : nombre d'epochs pour le fine-tuning (moins que la
#                    phase 1 car on part déjà d'un modèle performant,
#                    on ne fait qu'affiner).
# FT_LEARNING_RATE : learning rate volontairement très bas. Avec le LR
#                    par défaut d'Adam (0.001), les gradients détruiraient
#                    en quelques batches les poids pré-entraînés d'ImageNet.
#                    1e-5 permet un ajustement fin, pas une redécouverte.
# FT_UNFREEZE_LAST : nombre de couches finales de MobileNetV2 à dégeler.
#                    On ne dégèle jamais TOUT le réseau : les premières
#                    couches détectent des motifs génériques (bords,
#                    textures) valables pour n'importe quelle image et
#                    n'ont pas besoin d'être réentraînées.
# ══════════════════════════════════════════════════════════════════════
FT_EPOCHS         = 15
FT_LEARNING_RATE  = 1e-5
FT_UNFREEZE_LAST  = 30

# Rempli au moment de l'exécution avec le nombre réel d'epochs de la
# phase 1 (peut être < EPOCHS si EarlyStopping s'est déclenché avant).
# Sert uniquement à tracer la ligne verticale de transition sur le graphique.
PHASE1_EPOCH_COUNT = None


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Générateurs de données (inchangé)
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
# ÉTAPE 2 — Construction du modèle (inchangé)
# ══════════════════════════════════════════════════════════════════════
def build_model(nb_classes: int) -> Model:
    print("\n[2/5] Construction du modèle...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params     = model.count_params()
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"  Paramètres totaux      : {total_params:,}")
    print(f"  Paramètres entraînables: {trainable_params:,}")
    print(f"  Paramètres gelés       : {total_params - trainable_params:,}")

    # NOUVEAU : on retourne aussi base_model, nécessaire pour le dégeler
    # explicitement lors de la phase 2 (fine_tune_model en a besoin pour
    # accéder à sa liste de couches internes).
    return model, base_model


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement avec callbacks
# ══════════════════════════════════════════════════════════════════════
def build_callbacks():
    """
    Construit une liste de callbacks FRAÎCHE.

    Pourquoi une fonction et pas une liste globale réutilisée ?
    EarlyStopping et ModelCheckpoint sont des objets à état interne
    (ex: "meilleur score vu jusqu'ici", "nombre d'epochs sans
    amélioration"). Si on réutilisait les MÊMES instances entre la
    phase 1 et la phase 2, le fine-tuning hériterait de l'état de la
    phase 1 (ex: patience déjà à moitié consommée) — comportement
    difficile à prévoir. On recrée donc des callbacks neufs à chaque
    phase.
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # ── MODIFICATION 3 — Chemin BEST_PATH vient de config.py ──────
        # AVANT : BEST_PATH était défini localement dans ce fichier.
        # APRÈS : il vient de config.py → automatiquement correct
        #         sur PC et sur Colab sans rien changer ici.
        ModelCheckpoint(
            filepath=BEST_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def train_model(model, train_gen, val_gen):
    print(f"\n[3/5] Entraînement — Phase 1 : Feature Extraction ({EPOCHS} epochs max)...")
    print("  (base MobileNetV2 gelée — seule la tête de classification apprend)")

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=build_callbacks(),
    )

    return history


# ══════════════════════════════════════════════════════════════════════
# NOUVEAU — ÉTAPE 3bis — Fine-tuning (Phase 2)
# ──────────────────────────────────────────────────────────────────────
# Utilise EXACTEMENT les mêmes générateurs (train_gen, val_gen) que la
# phase 1 — donc les mêmes images. On ne charge rien de nouveau : on
# continue simplement l'entraînement du même objet `model`, en
# dégelant une partie de sa base.
# ══════════════════════════════════════════════════════════════════════
def fine_tune_model(model, base_model, train_gen, val_gen):
    print(f"\n[3bis/5] Entraînement — Phase 2 : Fine-tuning ({FT_EPOCHS} epochs max)...")

    # 1. Dégeler la base MobileNetV2 dans son ensemble...
    base_model.trainable = True

    # 2. ...puis re-geler tout sauf les FT_UNFREEZE_LAST dernières couches.
    #    Les couches basses (proches de l'entrée) codent des motifs très
    #    génériques (contours, textures, dégradés) valables pour
    #    n'importe quel type d'image : on ne les touche pas.
    #    Les couches hautes (proches de la sortie) codent des motifs plus
    #    abstraits et spécifiques : ce sont elles qu'on ajuste à nos
    #    30 espèces de plantes.
    fine_tune_at = len(base_model.layers) - FT_UNFREEZE_LAST
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 3. Recompilation OBLIGATOIRE après tout changement de `trainable`.
    #    Keras ne prend en compte les changements de trainable qu'au
    #    moment du compile() — les oublier est une source de bugs
    #    silencieux fréquente (le modèle continue de tourner mais
    #    n'entraîne pas ce qu'on croit).
    #    Learning rate volontairement très faible : voir commentaire
    #    sur FT_LEARNING_RATE plus haut.
    model.compile(
        optimizer=Adam(learning_rate=FT_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params     = model.count_params()
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"  Couches dégelées        : {FT_UNFREEZE_LAST} dernières couches de MobileNetV2")
    print(f"  Paramètres totaux       : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}  (avant fine-tuning: 167,838)")
    print(f"  Paramètres gelés        : {total_params - trainable_params:,}")

    # 4. On repart des générateurs de la phase 1 — mêmes images, aucun
    #    rechargement de données nécessaire. Le modèle continue depuis
    #    les poids déjà appris en phase 1 (on ne réinitialise rien).
    history_ft = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=FT_EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=build_callbacks(),
    )

    return history_ft


def merge_histories(history1, history2):
    """
    Fusionne les historiques des deux phases en un seul objet pour
    que les courbes finales (accuracy/loss) affichent une continuité
    de bout en bout.
    """
    merged = {}
    for key in history1.history:
        merged[key] = history1.history[key] + history2.history.get(key, [])
    return merged


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Évaluation finale (inchangé)
# ══════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_gen):
    print("\n[4/5] Évaluation sur le jeu de test...")

    loss, accuracy = model.evaluate(test_gen, steps=len(test_gen), verbose=1)

    print(f"\n  {'─'*40}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  {'─'*40}")

    return loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Sauvegarde
# ══════════════════════════════════════════════════════════════════════
def save_all(model, history_dict, train_gen):
    """
    Sauvegarde trois fichiers essentiels :
      1. modele_final.h5      → chargé par l'API pour les prédictions
      2. class_indices.json   → traduit l'indice prédit en nom d'espèce
      3. training_history.png → courbes pour le rapport

    NOUVEAU : `history_dict` est maintenant un simple dict (issu de
    merge_histories) et non plus l'objet `history` renvoyé par
    model.fit() — car on fusionne ici les courbes des phases 1 et 2.
    """
    print("\n[5/5] Sauvegarde...")

    # ── MODIFICATION 4 — Vérification explicite du dossier model/ ─────
    # AVANT : si model/ n'existait pas, model.save() crashait sans
    #         message clair → class_indices.json n'était jamais créé.
    # APRÈS : on vérifie et crée le dossier ici aussi (double sécurité).
    #         config.py le crée déjà au démarrage, mais cette ligne
    #         protège si quelqu'un importe train_model.py sans config.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Modèle complet
    model.save(MODEL_PATH)
    print(f"  Modèle sauvegardé       : {MODEL_PATH}")

    # 2. Correspondance indice → nom d'espèce
    # ── MODIFICATION 5 — Ajout d'un print de vérification ─────────────
    # AVANT : si class_indices.json n'était pas créé, on ne savait pas
    #         pourquoi (pas de message d'erreur clair).
    # APRÈS : on affiche le contenu des 3 premières classes pour
    #         confirmer visuellement que le fichier est bien écrit.
    class_indices = train_gen.class_indices
    idx_to_class  = {str(v): k for k, v in class_indices.items()}

    with open(INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"  Indices des classes     : {INDICES_PATH}")
    print(f"  Aperçu : { {k: idx_to_class[k] for k in list(idx_to_class)[:3]} }")

    # 3. Courbes d'apprentissage (phase 1 + phase 2 fusionnées)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history_dict['accuracy']) + 1)

    ax1.plot(epochs_range, history_dict['accuracy'],     label='Train')
    ax1.plot(epochs_range, history_dict['val_accuracy'], label='Validation')
    # Ligne verticale marquant la transition phase 1 -> phase 2
    if PHASE1_EPOCH_COUNT is not None:
        ax1.axvline(x=PHASE1_EPOCH_COUNT, color='gray', linestyle='--', label='Début fine-tuning')
    ax1.set_title('Accuracy par epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history_dict['loss'],     label='Train')
    ax2.plot(epochs_range, history_dict['val_loss'], label='Validation')
    if PHASE1_EPOCH_COUNT is not None:
        ax2.axvline(x=PHASE1_EPOCH_COUNT, color='gray', linestyle='--', label='Début fine-tuning')
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

    # ── MODIFICATION 6 — Affichage des chemins au démarrage ────────────
    # AVANT : on ne savait pas vers quel dossier le script pointait.
    #         Sur Colab, si le chemin était mauvais, l'erreur arrivait
    #         tard (après plusieurs minutes d'attente).
    # APRÈS : on affiche les chemins dès le début → on voit tout de suite
    #         si on pointe vers Drive (Colab) ou vers le PC local.
    print(f"\n  Chemins utilisés :")
    print(f"  Train  : {TRAIN_DIR}")
    print(f"  Modèle : {MODEL_PATH}")
    print(f"  Indices: {INDICES_PATH}")

    for d, nom in [(TRAIN_DIR, 'train'), (VAL_DIR, 'val'), (TEST_DIR, 'test')]:
        if not os.path.exists(d):
            print(f"\n[ERREUR] Dossier '{nom}' introuvable : {d}")
            print("  → Exécutez d'abord step2_prepare_data.py")
            exit(1)

    train_gen, val_gen, test_gen = build_generators()
    model, base_model            = build_model(nb_classes=train_gen.num_classes)

    # ── Phase 1 : Feature extraction (base gelée) ──────────────────────
    history_p1 = train_model(model, train_gen, val_gen)
    PHASE1_EPOCH_COUNT = len(history_p1.history['accuracy'])

    # ── Phase 2 : Fine-tuning (dernières couches dégelées) ─────────────
    # Utilise train_gen/val_gen inchangés : mêmes images qu'en phase 1,
    # on continue simplement à entraîner le même objet `model`.
    history_p2 = fine_tune_model(model, base_model, train_gen, val_gen)

    # ── Fusion des deux historiques pour un graphique continu ──────────
    history_merged = merge_histories(history_p1, history_p2)

    # ── Évaluation finale (après fine-tuning, donc sur le meilleur modèle) ──
    loss, accuracy = evaluate_model(model, test_gen)
    save_all(model, history_merged, train_gen)

    print("\n" + "="*60)
    print(f"  Entraînement terminé (Phase 1 + Phase 2 fine-tuning)")
    print(f"  Précision finale sur le test : {accuracy*100:.2f}%")
    print(f"  Modèle prêt : {MODEL_PATH}")
    print("="*60 + "\n")