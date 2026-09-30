# generer_matrice_confusion.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TEST_DIR, IMG_SIZE, BATCH_SIZE

model = load_model('model/modele_best.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

predictions = model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
noms_classes = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(len(noms_classes)))
ax.set_yticks(range(len(noms_classes)))
ax.set_xticklabels(noms_classes, rotation=90, fontsize=7)
ax.set_yticklabels(noms_classes, fontsize=7)
ax.set_xlabel('Classe prédite')
ax.set_ylabel('Classe réelle')
ax.set_title('Matrice de confusion — 31 classes')

plt.colorbar(im, ax=ax, label='Nombre de prédictions')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150)
print("[OK] Matrice sauvegardée dans model/confusion_matrix.png")