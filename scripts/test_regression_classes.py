# test_regression_classes.py — évalue le nouveau modèle sur TOUT le jeu de test
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TEST_DIR, IMG_SIZE, BATCH_SIZE
import numpy as np

model = load_model('model/modele_best.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

predictions = model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

idx_to_class = {v: k for k, v in test_gen.class_indices.items()}

# Accuracy par classe
print(f"\n{'Classe':25s} {'Accuracy':>10s}")
print("-" * 37)
for idx in range(len(idx_to_class)):
    mask = (y_true == idx)
    if mask.sum() == 0:
        continue
    acc = (y_pred[mask] == idx).mean()
    print(f"{idx_to_class[idx]:25s} {acc*100:9.1f}%")