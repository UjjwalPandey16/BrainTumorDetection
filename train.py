import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
from keras.utils import normalize
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical

os.environ['TF_DML_VISIBLE_DEVICES'] = 'auto'


image_directory = 'datasets_new/'


glioma_images = os.listdir(image_directory+'glioma/')
meningioma_images = os.listdir(image_directory+'meningioma/')
notumor_images = os.listdir(image_directory+'notumor/')
pituitary_images = os.listdir(image_directory+'pituitary/')

dataset = []
label = []

INPUT_SIZE = 64

# print(no_tumor_images)
#
# path = 'n0.jpg'
#
# print(path.split('.')[1])

categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
label_map = {category: i for i, category in enumerate(categories)}

# Update data loading and labeling
for category in categories:
    category_images = os.listdir(image_directory + category + '/')
    for image_name in category_images:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + category + '/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(label_map[category])


print(len(dataset))
print(len(label))

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# print(x_train.shape)
# print(y_train.shape)
#
# print(x_test.shape)
# print(y_test.shape)

num_classes = len(categories)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# One-hot encoding
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Update output layer of the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))  # Change the units to the number of classes
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, verbose=True, epochs=50, validation_data=(x_test, y_test), shuffle=False)

model.save('models/brainTumor-4category-b64e50-categorical.h5')
