import functools
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import tensorflow as tf

from keras import __version__
from keras import applications
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
from metrics import metric

IM_WIDTH, IM_HEIGHT = 300, 300
NB_EPOCHS_TL = 0
NB_EPOCHS_FT = 5
BAT_SIZE = 16
FC_SIZE = 1024
NB_VGG_LAYERS_TO_FREEZE = 3


def f1_score(y_pred, y_true):
    """
    returns f1 score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


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
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy', f1_score])


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

    # count = NB_VGG_LAYERS_TO_FREEZE
    # for layer in model.layers:
    #     if count>=0 or layer.name[-4:]=='pool':
    #         layer.trainable=False
    #     else :
    #         layer.trainable=True
    #     count-=1
    for layer in model.layers[:NB_VGG_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_VGG_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    # opt = Adam(lr=0.0001)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', f1_score])



def train(args):

    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """

    nb_classes = 2
    neg_class_label  = '0'
    pos_class_label  = '1'
    nb_train_samples = get_nb_files(args.train_dir)
    nb_val_samples = get_nb_files(args.val_dir)
    batch_size = int(args.batch_size)

    nb_train_samples_neg = len([name for name in os.listdir(os.path.join(args.train_dir, neg_class_label))])
    nb_train_samples_pos = len([name for name in os.listdir(os.path.join(args.train_dir, pos_class_label))])
    print("Number of neg training examples is ",nb_train_samples_neg )
    print("Number of pos training examples is ",nb_train_samples_pos )


    # data prep
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(args.train_dir,
                                                        target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(args.val_dir,
                                                            target_size=(IM_WIDTH, IM_HEIGHT),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle=False)
    val_labels = validation_generator.classes
    # validation_labels = to_categorical(val_labels, num_classes=2)


    # setting up class weights for imbalanced dataset ...
    # The sum of the weights of all examples stays the same.


    total = nb_train_samples_neg + nb_train_samples_pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, nb_train_samples_pos, 100 * nb_train_samples_pos/ total))


    weight_for_0 =  total / nb_train_samples_neg 
    weight_for_1 =  total / nb_train_samples_pos

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # setup model
    # model = keras.models.load_model('./experiment/vgg_ft_ni10.model')
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    print("Number of vgg layers :", len(base_model.layers))

    # adding fully connected layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning...
    setup_to_transfer_learn(model, base_model)

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    #     print(layer.trainable)

    # checkpoint = ModelCheckpoint("ft_vgg16.h5",
    #                              monitor=RocCallback(validation_generator),    # not working with custom callbacks
    #                              verbose=1,
    #                              save_best_only=True,
    #                              save_weights_only=False,
    #                              mode='auto',
    #                              period=1)

    # print('Transfer Learning is starting...')
    # history_tl = model.fit_generator(train_generator,
    #                                  nb_epoch=NB_EPOCHS_TL,
    #                                  samples_per_epoch=nb_train_samples,
    #                                  validation_data=validation_generator,
    #                                  nb_val_samples=nb_val_samples,
    #                                  class_weight=class_weight,
    #                                  callbacks=[RocCallback(validation_generator)])

    # fine-tuning ...
    setup_to_finetune(model)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        print(layer.trainable)
    model.load_weights('./experiment/vgg_ft_ni25.h5')
    # model.summary()

    # print('Fine tuning is starting...')
    # history_ft = model.fit_generator(train_generator,
    #                                  samples_per_epoch=nb_train_samples,
    #                                  nb_epoch=NB_EPOCHS_FT,
    #                                  validation_data=validation_generator,
    #                                  nb_val_samples=nb_val_samples,
    #                                  class_weight=class_weight,
    #                                  callbacks=[RocCallback(validation_generator)])

    # making predictions ...3
    # model.save_weights('./experiment/vgg_ft_ni25.h5')
    # model.save('./experiment/vgg_ft_ni10.model')
    pred = model.predict_generator(validation_generator, steps=nb_val_samples//batch_size)

    # confusion_matrix(val_labels, pred)

    # F1Score ...
    predictions = np.argmax(pred, axis=1)


    # Storing Predictions as CSV ...
    filenames = validation_generator.filenames
    labels = validation_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predictions]
    results = pd.DataFrame({"Filename": filenames,
                            "true" : pred[:, 1],
                            "Predictions": predictions,
                            "Label": val_labels})
    # results.to_csv("./experiment/ft_ni_results15.csv", index=False)
    # metric("./experiment/ft_ni_results15.csv")

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
        print('\nroc-auc_val: %s' % (str(round(roc_val, 4))), end=100 * ' ' + '\n')
        if roc_val > self.best:
            print('roc score increased from %s to %s' % (str(round(self.best, 4)), str(round(roc_val, 4))),
                  end=100 * ' ' + '\n')
            print('Saving model ...')
            self.best = roc_val
            self.model.save('./experiment/vgg_ft_ni25.model')
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




if __name__ == "__main__":
    a = argparse.ArgumentParser()
    # a.add_argument("--train_dir", default='/pstore/home/maunza/kaggle_DR/lumin_norm/trainLabelsTwoClass2/train')
    a.add_argument("--train_dir", default='./finalDataset/train')
    a.add_argument("--val_dir", default='./finalDataset/val')
    # a.add_argument("--val_dir", default='/pstore/home/maunza/kaggle_DR/lumin_norm/trainLabelsTwoClass2/validate')
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
