
"""
Calcule les statistiques nécessaires à la distance de Mahalanobis :
- une moyenne d'embedding par classe (30 vecteurs de 1280 dims)
- une SEULE matrice de covariance partagée entre toutes les classes
  (nécessaire car 1000 images/classe < 1280 dimensions : une covariance
  par classe serait sous-déterminée et instable)

À exécuter UNE FOIS après l'entraînement. Durée estimée : 10-20 min sur CPU
pour 30 000 images (dépend de ta machine).
"""
import numpy as np
import pickle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_DIR, IMG_SIZE, BATCH_SIZE

print("Chargement du modèle...")
model = load_model("model/modele_best.h5")
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
# Générateur SANS augmentation, SANS mélange — on veut des images stables
# et un ordre reproductible pour associer chaque embedding à sa vraie classe
eval_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_gen_stable = eval_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

nb_classes = train_gen_stable.num_classes
nb_batches = len(train_gen_stable)
print(f"Extraction des embeddings : {nb_classes} classes, {nb_batches} batchs...")

embeddings_par_classe = {i: [] for i in range(nb_classes)}

for i in range(nb_batches):
    batch_x, batch_y = next(train_gen_stable)
    embeddings = feature_extractor.predict(batch_x, verbose=0)
    labels = np.argmax(batch_y, axis=1)
    for emb, label in zip(embeddings, labels):
        embeddings_par_classe[int(label)].append(emb)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{nb_batches} batchs traités")

print("Extraction terminée. Calcul des statistiques...")

# ── Moyenne par classe ──────────────────────────────────────────
moyennes = {}
tous_les_ecarts = []  # pour la covariance partagée

for classe, embs in embeddings_par_classe.items():
    embs = np.array(embs)
    moyenne = embs.mean(axis=0)
    moyennes[classe] = moyenne
    tous_les_ecarts.append(embs - moyenne)  # écarts à la moyenne de leur classe

# ── Covariance partagée (pooled), calculée sur TOUS les écarts ────
tous_les_ecarts = np.vstack(tous_les_ecarts)  # shape (30000, 1280)
covariance_partagee = np.cov(tous_les_ecarts, rowvar=False)
covariance_partagee += np.eye(covariance_partagee.shape[0]) * 1e-6  # régularisation
cov_inv = np.linalg.inv(covariance_partagee)

stats = {
    "moyennes": moyennes,      # dict {classe: vecteur 1280-dim}
    "cov_inv": cov_inv,        # matrice partagée (1280, 1280)
    "idx_to_class": train_gen_stable.class_indices,
}

with open("model/ood_stats.pkl", "wb") as f:
    pickle.dump(stats, f)

print(f"[OK] Statistiques sauvegardées dans model/ood_stats.pkl")
print(f"     {nb_classes} classes, dimension embedding : {tous_les_ecarts.shape[1]}")