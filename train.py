import pandas as pd
import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
import math
import datetime
import time
from os import path

# Default dimensions we found online
img_width, img_height = 600, 400

# Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'  # loading up our datasets || model weights
train_data_dir = './finalDataset/train/'
validation_data_dir = './finalDataset/val/'
# test_data_dir = ‘data / test’

# number of epochs to train top model
num_epochs = 10  # this has been changed after multiple model run
# batch size used by flow_from_directory and predict_generator
batch_size = 16

# Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1./255)
# needed to create the bottleneck .npy files

# __this can take an hour and half to run so only run it once.
# once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__

start = datetime.datetime.now()
print("Starting vgg predictions :")

generator = datagen.flow_from_directory(train_data_dir,
                                        target_size=(img_width, img_height),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

generator_val = datagen.flow_from_directory(validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)

nb_train_samples = len(generator.filenames)
nb_val_samples = len(generator_val.filenames)

num_classes = len(generator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_val = int(math.ceil(nb_val_samples/batch_size))

bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)
bottleneck_features_val = vgg16.predict_generator(generator_val, predict_size_val)

np.save('bottleneck_features_train.npy', bottleneck_features_train)
np.save('bottleneck_features_val.npy', bottleneck_features_val)

print("Saving vgg predictions! ")
end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)

# training data
generator_top = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='binary',
                                            shuffle=False)

generator_val_top = datagen.flow_from_directory(validation_data_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                shuffle=False)


nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)


# print("Found {} classes".format(num_classes))
# load the bottleneck features saved earlier
# train_data = bottleneck_features_train
train_data = np.load('bottleneck_features_train.npy')
validation_data = np.load('bottleneck_features_val.npy')

# get the class labels for the training data, in the original order
train_labels = generator_top.classes
val_labels = generator_val_top.classes

# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)
validation_labels = to_categorical(val_labels, num_classes=num_classes)

# This is the best model we found. For additional models, check out I_notebook.ipynb
print(train_data.shape)
print(train_data.shape[1:])
start = datetime.datetime.now()
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.3))
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='sigmoid'))

if path.exists(top_model_weights_path):
    model.load_weights(top_model_weights_path)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(train_data, train_labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)

(eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

prediction = model.predict(validation_data)
# print(validation_data.shape)
# print(train_data.shape)

# p = np.zeros((2))           # p[0]=tp, p[1]=fp, p[2]=fn, p[3]=tn

# for i in range(len(validation_data)):
#     if validation_labels==1 and prediction[i][0]>prediction[i][1] :
#         p[0]+=1
#     elif validation_labels == 0 and prediction[i][0]<prediction[i][1]:
#         p[1]+=1

    # print(validation_labels[i], prediction[i])

# sensitivity = p[0]/prediction.shape[0]
# specificity = p[1]/prediction.shape[0]

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))
end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)

# Graphing our training and validation
# acc = history.history['acc']
# # val_acc = history.history['val_acc']
# loss = history.history['loss']
# # val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'r', label='Training acc')
# # plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')e
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training loss')
# # plt.plot(epochs, val_loss, 'b, label='Validation loss')
# plt.title('Training and validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
