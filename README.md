# classifia — Classification de 30 espèces avec MobileNetV2

## 1. Installation

Ouvrez un terminal dans VS Code (`Ctrl + ù`) et exécutez :

```bash
pip install -r requirements.txt
```

> Si vous utilisez un environnement virtuel (recommandé) :
> ```bash
> python -m venv .venv
> .venv\Scripts\activate       # Windows
> source .venv/bin/activate    # Mac / Linux
> pip install -r requirements.txt
> ```

---

## 2. Préparer vos données

Placez vos images brutes dans `data/raw/`, **une espèce = un sous-dossier** :

```
classifia/
└── data/
    └── raw/
        ├── espece_01/
        ├── espece_02/
        └── ... (30 dossiers)
```

---

## 3. Ordre d'exécution

| Ordre | Fichier | Rôle |
|-------|---------|------|
| 1 | `step1_check_data.py` | Vérifie raw/ sans rien modifier |
| 2 | `step2_prepare_data.py` | Nettoie, redimensionne, divise en train/val/test |
| 3 | `step3_build_generators.py` | Crée les générateurs (augmentation sur train) |
| 4 | *(votre script d'entraînement)* | Utilise les générateurs pour entraîner MobileNetV2 |

Lancer chaque étape depuis le terminal :

```bash
python step1_check_data.py
python step2_prepare_data.py
python step3_build_generators.py
```

---

## 4. Structure du projet

```
classifia/
├── config.py                  ← chemins & constantes (à adapter si besoin)
├── requirements.txt           ← bibliothèques à installer
├── step1_check_data.py        ← vérification
├── step2_prepare_data.py      ← préparation du dataset
├── step3_build_generators.py  ← générateurs de données
└── data/
    ├── raw/                   ← vos images brutes (ne pas modifier)
    └── dataset_prepare/       ← généré automatiquement par step2
        ├── train/
        ├── val/
        └── test/
```

---

## 5. Paramètres modifiables

Tout est centralisé dans `config.py` :

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `IMG_SIZE` | `(224, 224)` | Taille d'entrée MobileNetV2 |
| `BATCH_SIZE` | `32` | Images par batch |
| `TRAIN_RATIO` | `0.70` | 70 % pour l'entraînement |
| `VAL_RATIO` | `0.15` | 15 % pour la validation |
| Reste | `0.15` | 15 % pour le test final |

---

## 6. Bibliothèques installées

| Bibliothèque | Rôle |
|---|---|
| `tensorflow` | Keras, MobileNetV2, ImageDataGenerator |
| `Pillow` | Lecture et redimensionnement des images |
| `numpy` | Manipulation des tableaux d'images |
