# Rapport de Projet ClassifIA

## Introduction

Ce projet, nommé **ClassifIA**, consiste en une application de classification d'images de plantes et d'animaux utilisant l'intelligence artificielle. L'objectif principal est de développer un système capable de reconnaître différentes espèces végétales à partir d'images.

## Objectifs du Projet

- **E1** : Collecter et préparer les données
- **E3** : Déployer un modèle IA
- **E4** : Créer une application complète

## Stack Technique

- **Langage** : Python
- **Backend** : FastAPI
- **IA** : TensorFlow / PyTorch
- **Interface** : Streamlit

## Structure du Projet

Le projet est organisé comme suit :

- `data/` : Contient les données brutes et traitées, y compris les dossiers d'images pour chaque plante.
- `model/` : Dossier pour le modèle IA.
- `api/` : Code du backend FastAPI.
- `app/` : Interface utilisateur.
- `scripts/` : Scripts Python pour diverses tâches.
- `docs/` : Documentation, incluant ce README et requirements.txt.
- `logs/` : Fichiers de logs.
- `tests/` : Tests unitaires.

## Tâches Réalisées

### 1. Initialisation de la Base de Données (`init_db.py`)

- Création d'une base de données SQLite (`flowers.db`) avec deux tables principales :
  - `species_info` : Informations sur les espèces (nom scientifique, famille, utilité, etc.)
  - `images_train` : Chemins des images d'entraînement liées aux espèces.
- Cette étape établit la structure de données pour stocker les informations collectées.

### 2. Collecte de Données Wikipedia (`test_data.py`)

- Extraction d'informations depuis Wikipedia pour chaque plante ciblée.
- Utilisation d'un dictionnaire `targets` pour mapper les noms de dossiers aux titres exacts des articles Wikipedia, évitant les homonymies.
- Récupération d'extraits de texte (descriptions) pour enrichir les données.
- Sauvegarde des données extraites dans `wiki_data.json`.

### 3. Enrichissement des Données (`enrich_data.py`)

- Chargement des données depuis `wiki_data.json`.
- Utilisation de l'API Perenual pour obtenir des détails techniques (besoins en eau, exposition au soleil, cycle de vie, etc.).
- Traduction automatique des textes en français à l'aide de GoogleTranslator.
- Insertion des données enrichies dans la base de données SQLite.
- Gestion des erreurs et des données manquantes.

### 4. Mise à Jour des Descriptions (`to_db.py`)

- Lecture de `wiki_data.json` pour récupérer les descriptions utilité.
- Mise à jour sélective de la colonne `description_utilite` dans la table `species_info`.
- Ce script est complémentaire à `enrich_data.py` et peut être relancé indépendamment pour rafraîchir les descriptions.

### 5. Vérification du Dataset (`check_dataset.py`)

- Comptage du nombre d'images par classe dans le dossier `data/processed`.
- Aide à vérifier la distribution des données avant l'entraînement du modèle.

## Données

- **Données brutes** : Images organisées par plante dans `data/raw/`.
- **Dataset divisé** : `data/split_ttv_dataset_type_of_plants/` avec dossiers Train, Test et Validation pour chaque plante.
- **Base de données** : `data/flowers.db` contenant les métadonnées des espèces.

## Pipeline de Données

Le pipeline de traitement des données suit cette séquence :

1. `init_db.py` → Création de la DB
2. `test_data.py` → Collecte Wikipedia → `wiki_data.json`
3. `enrich_data.py` → Enrichissement avec API → Insertion en DB
4. `to_db.py` (optionnel) → Mise à jour descriptions

## Prochaines Étapes

- Développement du modèle IA pour la classification.
- Création de l'API FastAPI pour servir le modèle.
- Développement de l'interface Streamlit pour l'utilisateur.
- Tests et déploiement.

## Conclusion

Ce projet a permis de mettre en place une base solide pour la classification d'images de plantes, avec une collecte et un enrichissement des données structurés. Les scripts développés assurent une pipeline automatisée et maintenable.</content>
<parameter name="filePath">c:\Users\user\Desktop\classif-ia\report.md