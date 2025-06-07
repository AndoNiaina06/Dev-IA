
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import models

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier.h5')

#    3 echantillons, juste changer le nom de l'images en bas
img = cv2.imread('bird.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (32, 32))
img = img / 255.0

plt.imshow(img)
plt.axis('off')

# décommenter ce ligne de code en bas pour voir l'image

# plt.show()

# Prediction en ligne de commande
img_array = np.expand_dims(img, axis=0)  # (1, 32, 32, 3)
prediction = model.predict(img_array)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
