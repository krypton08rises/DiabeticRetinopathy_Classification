import keras as K
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

tr_data = ImageDataGenerator(rescale=1. / 255)
batchsize=4
train_data_generator = tr_data.flow_from_directory(directory='./finalDataset/train',
                                                   target_size=(600, 400),
                                                   batch_size=batchsize,
                                                   class_mode=None,
                                                   shuffle=True)

ts_data = ImageDataGenerator(rescale=1. / 255)
test_data_generator = ts_data.flow_from_directory(directory='./finalDataset/val',
                                                  target_size=(200, 135),
                                                  batch_size=batchsize,
                                                  class_mode=None,
                                                  shuffle=True)

# print(train_data_generator.image_shape)
# print(train_data_generator[0])
print("next image:")
# print(train_data_generator[1])

num_classes = len(train_data_generator.class_indices)

model = Sequential()

model.add(Conv2D(input_shape=(200, 135, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# keras.losses.categorical_crossentropy for multiclass!

model.summary()
# model.load_weights('model_vgg16.h5')
checkpoint = ModelCheckpoint("vgg16_1.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=len(train_data_generator.filenames)/batchsize,
                           generator=train_data_generator,
                           validation_data=test_data_generator,
                           validation_steps=10,
                           epochs=0,
                           callbacks=[checkpoint])      #, early])

model.save_weights('model_vgg16.h5')

# print(len(test_data_generator.filenames))
# print(test_data_generator.filenames)
prediction = model.predict_generator(test_data_generator, steps=len(test_data_generator.filenames))
print(prediction)
