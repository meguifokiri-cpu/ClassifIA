# test_diagnostics_complet.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

import numpy as np
from PIL import Image
from model_loader import predict_species

images_test = {
    "banane_vraie":     r"C:\Users\HP\Desktop\classif-IA\data\test\images.webp",
    "epinard_vrai":     r"C:\Users\HP\Desktop\classif-IA\data\test\epinard.jpg",
    "aloe_vera":        r"C:\Users\HP\Desktop\classif-IA\data\test\aloe-vera.webp",
    "roses_bouquet":    r"C:\Users\HP\Desktop\classif-IA\data\test\téléchargé.webp",
    "avion":            r"C:\Users\HP\Desktop\classif-IA\data\test\avion.webp",
   "Fleur_deglantier":  r"C:\Users\HP\Desktop\classif-IA\data\test\medium.jpeg",
   "Avion_Air_France":  r"C:\Users\HP\Desktop\classif-IA\data\test\images.jfif",
   "personne":          r"C:\Users\HP\Desktop\classif-IA\data\test\Sans titre.png",

}

print(f"{'Image':20s} {'Prédit':12s} {'Conf%':>7s} {'Entropie':>9s} {'Marge':>7s} {'Mahal':>7s}")
print("-" * 70)

for nom, chemin in images_test.items():
    img = Image.open(chemin).convert('RGB').resize((224, 224), Image.NEAREST)
    nom_dossier, confiance, diag = predict_species(np.array(img))
    print(f"{nom:20s} {nom_dossier:12s} {confiance*100:7.1f} {diag['entropie']:9.4f} {diag['ecart_top1_top2']:7.4f} {diag['distance_mahalanobis']:7.2f}")