import keras as K
import tensorflow as tf
from os import path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import math

batch_size = 4
num_classes = 2
num_epochs = 0
train_in = './finalDataset/train'
test_in = './finalDataset/val'


# def auc_roc(generator):
#     # any tensorflow metric
#     value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
#
#     # find all variables created for this metric
#     metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
#
#     # Add metric variables to GLOBAL_VARIABLES collection.
#     # They will be initialized for new session.
#     for v in metric_vars:
#         tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
#
#     # force to update metric values
#     with tf.control_dependencies([update_op]):
#         value = tf.identity(value)
#         return value


class RocCallback(Callback):
    def __init__(self, validation_data):
        # self.x = training_data[0]
        # self.y = training_data.classes
        self.x_val = validation_data
        self.y_val = to_categorical(validation_data.classes, num_classes=2)


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # y_pred_train = self.model.predict_proba(self.x)
        # roc_train = roc_auc_score(self.y, y_pred_train)
        # y_pred_val = self.model.predict_proba(self.x_val)
        y_pred_val = self.model.predict_generator(self.x_val, steps=len(self.x_val.filenames)//batch_size)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        # print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        print('\rroc-auc_val: %s'%(str(round(roc_val, 4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def setup_generators(train_in, test_in):
    tr_data = ImageDataGenerator(rescale=1. / 255)
    train_data_generator = tr_data.flow_from_directory(directory=train_in,
                                                       target_size=(175, 175),
                                                       batch_size=batch_size,
                                                       class_mode='categorical',
                                                       shuffle=True)

    ts_data = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = ts_data.flow_from_directory(directory=test_in,
                                                      target_size=(175, 175),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True)
    return train_data_generator, test_data_generator

# np.save('./cropped_features_train.npy', )


# gen = test_data_generator.next()
# print(gen)
# print(train_data_generator.filenames)
# sys.exit(0)

# num_classes = len(train_data_generator.class_indices)
# print(train_data_generator.filenames)
# nb_train_samples=len(train_data_generator)
# print(nb_train_samples)


# train_labels = to_categorical(train_data_generator.classes, num_classes=num_classes)
# validation_labels = to_categorical(test_data_generator.classes, num_classes=num_classes)
# print(train_labels)

def def_model():
    model = Sequential()

    model.add(Conv2D(input_shape=(175, 175, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
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
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="sigmoid"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model

# keras.losses.categorical_crossentropy for multiclass!


model = def_model()

model.summary()


train_data_generator,test_data_generator = setup_generators(train_in, test_in)

model_weights = './untrained_vgg16.h5'
if path.exists(model_weights):
    model.load_weights(model_weights)

checkpoint = ModelCheckpoint("utvgg_norm.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)
# quit()
# early = EarlyStopping(monitor='val_acc',
#                       min_delta=0,
#                       patience=20,
#                       verbose=1,
#                       mode='auto')

history = model.fit_generator(steps_per_epoch=len(train_data_generator.filenames) // batch_size,
                              generator=train_data_generator,
                              validation_data=test_data_generator,
                              validation_steps=10,
                              epochs=num_epochs,
                              callbacks=[checkpoint, RocCallback(test_data_generator)])

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'r')
# plt.plot(epochs, acc, 'r')
# plt.title('Training and validation accuracy')
# plt.figure()
#
# plt.plot(epochs, loss, 'r')
# plt.plot(epochs, val_loss, 'r')

model.save_weights('utvgg_norm.h5.h5')

# print(len(test_data_generator.filenames))
# print(test_data_generator.filenames)
prediction = model.predict_generator(test_data_generator,
                                     steps=len(test_data_generator.filenames))
# print(prediction)
# np.save('predictions.txt', prediction)



