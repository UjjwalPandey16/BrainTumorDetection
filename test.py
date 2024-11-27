import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model('models/brainTumor-4category-b64e50-categorical.h5')

image = cv2.imread('Testing/meningioma/Te-me_0010.jpg')

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)

print(result)

print(result[0][0])

TumorState = bool(result[0][0])

print(TumorState)
