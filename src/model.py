
# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import os
import numpy as np
from PIL import Image
import random
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from skimage import transform
from torchvision import transforms
from torchvision.transforms import Compose as C
from PIL import Image, ImageOps
import cv2

def one_hot_labeling(class_name):
    if class_name == "seviye1":
        return(np.array([1,0,0]))
    elif class_name == "seviye2":
        return(np.array([0,1,0]))
    elif class_name == "seviye3":
        return(np.array([0,0,1]))

def read_dataset(path):
    IMG_SIZE = 200
    classes = os.listdir(path)
    dataset = []

    for cls in classes:
        images = os.listdir(path+"/"+cls)
        for img in images:
            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            dataset.append([np.array(image),one_hot_labeling(cls)])

            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = image.rotate(90)
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            dataset.append([np.array(image),one_hot_labeling(cls)])

            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = image.rotate(270)
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            dataset.append([np.array(image),one_hot_labeling(cls)])

            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = ImageOps.flip(image)
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            dataset.append([np.array(image),one_hot_labeling(cls)])

            image = Image.open(path+"/"+"/"+cls+"/"+img)
            image = image.convert('L')
            image = ImageOps.mirror(image)
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            dataset.append([np.array(image),one_hot_labeling(cls)])

    random.shuffle(dataset)
    return dataset
dataset = read_dataset('C:\\Users\\Elif\\Desktop\\bitirmeProjesi\\3classdataset\\Dataset')

train_dataset = dataset[int(len(dataset)/5):]
test_dataset = dataset[:int(len(dataset)/5)]


print(len(dataset))
print(len(train_dataset))
print(len(test_dataset))


IMG_SIZE = 200
trainImages = np.array([i[0] for i in train_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_dataset])


IMG_SIZE = 200
testImages = np.array([i[0] for i in test_dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_dataset])


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(trainImages, trainLabels, epochs=20,
                    validation_data=(testImages, testLabels))

model.save("model1.h5")


'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit(trainImages,trainLabels, batch_size = 32, epochs = 8, verbose = 1)
'''
'''
test_loss, test_acc = model.evaluate(testImages, testLabels)

print('Test accuracy:', test_acc)

predictions = model.predict(testImages)
count = 0
for i in range(len(predictions)):
    if np.argmax(predictions[i]) != np.argmax(testLabels[i]):
        count = count + 1
print(count)
'''
