import functools
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import keras as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow as tf

from keras import __version__
from metrics import metric
from keras import applications
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, cohen_kappa_score
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input

IM_WIDTH, IM_HEIGHT = 300, 300  # fixed size for InceptionV3
NB_EPOCHS_TL = 1
NB_EPOCHS_FT = 2
BAT_SIZE = 16
FC_SIZE = 1024
NB_VGG_LAYERS_TO_FREEZE = 10
NB_RESNET_LAYERS_TO_FREEZE = 50


def get_nb_files(directory):
    """
    Get number of files by searching directory recursively
    """
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))

    print("Number of files in", directory, " is ", cnt)
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print(layer.trainable)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """
    Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model):
    """
    Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """

    count = NB_VGG_LAYERS_TO_FREEZE
    for layer in model.layers:
        if count>=0 or layer.name[-4:]=='pool':
            layer.trainable=False
        else :
            layer.trainable=True
        count-=1
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    #     print(layer.trainable)
    # for layer in model.layers[:NB_VGG_LAYERS_TO_FREEZE]:
    #     layer.trainable = False
    # for layer in model.layers[NB_VGG_LAYERS_TO_FREEZE:]:
    #     layer.trainable = True

    opt = Adam(lr=0.0001)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])



def train(args):

    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """

    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(args.train_dir,
                                                        target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(args.val_dir,
                                                            target_size=(IM_WIDTH, IM_HEIGHT),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=True)
    val_labels = validation_generator.classes
    # validation_labels = to_categorical(val_labels, num_classes=2)


    # setting up class weights for imbalanced dataset ...
    # The sum of the weights of all examples stays the same.


    neg = 27216 + 1015
    pos = 5920 + 953 + 7905
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # setup model
    # model = K.models.load_model('./vgg_ft_ci20.model')
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    print("Number of vgg layers :", len(base_model.layers))

    # adding fully connected layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning...


    setup_to_transfer_learn(model, base_model)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print(layer.trainable)

    # checkpoint = ModelCheckpoint("ft_vgg16.h5",
    #                              monitor=RocCallback(validation_generator),    # not working with custom callbacks
    #                              verbose=1,
    #                              save_best_only=True,
    #                              save_weights_only=False,
    #                              mode='auto',
    #                              period=1)

    print('Transfer Learning is starting...')
    history_tl = model.fit_generator(train_generator,
                                     nb_epoch=NB_EPOCHS_TL,
                                     samples_per_epoch=nb_train_samples,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight=class_weight,
                                     callbacks=[RocCallback(validation_generator)])

    # fine-tuning ...
    setup_to_finetune(model)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print(layer.trainable)

    # model.summary()

    print('Fine tuning is starting...')
    history_ft = model.fit_generator(train_generator,
                                     samples_per_epoch=nb_train_samples,
                                     nb_epoch=NB_EPOCHS_FT,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight=class_weight,
                                     callbacks=[RocCallback(validation_generator)])

    # making predictions ...3
    model.save('./experiment5/vgg_ft_ni20.model')
    pred = model.predict_generator(validation_generator,
                                   steps=len(validation_generator.filenames) // batch_size)
    pred_Y_cat = np.argmax(pred, -1)


    # F1Score ...
    predictions = np.argmax(pred, axis=1)
    f1score = f1_score(predictions, val_labels)
    print("F1 Score is", f1score)


    # Storing Predictions as CSV ...
    filenames = validation_generator.filenames
    labels = validation_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predictions]
    results = pd.DataFrame({"Filename": filenames,
                            "true":pred[:, 1],
                            "Predictions": predictions,
                            "Label": val_labels})
    results.to_csv("./experiment5/ft_ni_results30.csv", index=False)
    metric("./experiment5/ft_ni_results20.csv")

    # plotting data...
    if args.plot:
        plot_training(pred, val_labels)


def kappa_score(y_pred, y_true):
    skl_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return skl_score


# Callback to area under the curve ...
class RocCallback(Callback):
    def __init__(self, validation_data):
        # self.x = training_data[0]
        # self.y = training_data.classes
        self.x_val = validation_data
        self.y_val = validation_data.classes
        self.best = 0

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
        y_pred_val = self.model.predict_generator(self.x_val, steps=len(self.x_val.filenames) // BAT_SIZE)
        roc_val = roc_auc_score(self.y_val, y_pred_val[:, 1])

        # print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*'
        # '+'\n')
        print('\rroc-auc_val: %s' % (str(round(roc_val, 4))), end=100 * ' ' + '\n')
        if roc_val > self.best:
            print('roc score increased from %s to %s' % (str(round(self.best, 4)), str(round(roc_val, 4))),
                  end=100 * ' ' + '\n')
            print('Saving model ...')
            self.best = roc_val
            self.model.save('./experiment5/vgg_ft_ni30.model')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def plot_training(prediction, labels):
    lr_probs = prediction[:, 1]
    lr_auc = roc_auc_score(labels, lr_probs)
    lr_fpr, lr_tpr, _ = roc_curve(labels, lr_probs)
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def f1_score(predictions, labels):
    fs = np.zeros((4))  # 0-tp, 1-fp, 2-fn, 3-tn
    for i in range(len(predictions)):
        if labels[i] == 1 and predictions[i] == 1:
            fs[0] += 1
        elif labels[i] == 1 and predictions[i] == 0:
            fs[2] += 1
        elif labels[i] == 0 and predictions[i] == 1:
            fs[1] += 1
        else:
            fs[3] += 1
    precision = fs[0] / (fs[0] + fs[1])
    recall = fs[0] / (fs[0] + fs[2])

    f1score = 2 * precision * recall / (precision + recall)
    return f1score


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='./finalDataset/train/')
    a.add_argument("--val_dir", default='./finalDataset/val/')
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--plot", action="store_true", default=True)

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
