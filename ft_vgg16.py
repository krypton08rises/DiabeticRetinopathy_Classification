import functools
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import keras as K
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import __version__
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import applications
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.metrics import precision, recall, auc

IM_WIDTH, IM_HEIGHT = 224, 224  # fixed size for InceptionV3
NB_EPOCHS = 0
BAT_SIZE = 16
FC_SIZE = 1024
NB_VGG_LAYERS_TO_FREEZE = 10


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
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
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
    for layer in model.layers[:NB_VGG_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_VGG_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    opt = Adam(lr=0.0001)
    # model.compile(optimizer=opt,
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', 'auc'])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])



def train(args):
    """
    Use transfer learning and fine-tuning to train a network on a new dataset
    """
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    # print(nb_val_samples // batch_size)

    # data prep
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # train_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input,
    #     rotation_range=30,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True
    # )
    # test_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input,
    #     rotation_range=30,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True
    # )

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
    # print("validation_labels : ", validation_labels.shape)
    validation_labels = to_categorical(val_labels, num_classes=2)
    # print("validation_labels now : ", validation_labels.shape)

    # setup model
    # base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    # print("NUmber of vgg layers :", len(base_model.layers))
    # number of layers in network

    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    checkpoint = ModelCheckpoint("utvgg_norm.h5",
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)

    history_tl = model.fit_generator(train_generator,
                                     nb_epoch=nb_epoch,
                                     samples_per_epoch=nb_train_samples,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight='auto',
                                    callbacks=[RocCallback(validation_generator), checkpoint])

    # fine-tuning


    # ival = IntervalEvaluation(validation_generator=validation_generator, interval=1)
    # roc = RocCallback(validation_generator)

    # model = K.models.load_model(args.output_model_file)
    setup_to_finetune(model)


    history_ft = model.fit_generator(train_generator,
                                     samples_per_epoch=nb_train_samples,
                                     nb_epoch=nb_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples,
                                     class_weight='auto',
                                     callbacks=[RocCallback(validation_generator)])



    pred = model.predict_generator(validation_generator,
                                   steps=len(validation_generator.filenames)//batch_size)
    predictions = np.argmax(pred, axis=1)
    f1score = f1_score(predictions, val_labels)

    print("F1 Score is", f1score)

        # filenames = validation_generator.filenames
    # labels = validation_generator.class_indices
    # labels = dict((v, k) for k, v in labels.items())
    # print(labels)
    # predictions = [labels[k] for k in predictions]
    # results = pd.DataFrame({"Filename": filenames,
    #                         "Predictions": predictions,
    #                         "Label": val_labels})
    # results.to_csv("normalised_ft_results10.csv", index=False)
    # print(pred)
    # print(pred.shape, validation_labels.shape)
    model.save(args.output_model_file)

    # print(pred.shape, validation_labels.shape)
    if args.plot:
        plot_training(predictions, val_labels)



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
        y_pred_val = self.model.predict_generator(self.x_val, steps=len(self.x_val.filenames)//BAT_SIZE)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        # print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        print('\rroc-auc_val: %s'%(str(round(roc_val, 4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return




def plot_training(prediction, labels):

    test_y = np.array(labels)

    lr_probs = prediction[:, 1]
    lab = labels[:, 1]
    lr_auc = roc_auc_score(labels, lr_probs)
    lr_fpr, lr_tpr, _ = roc_curve(lab, lr_probs)
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(len(acc))
    # plt.plot(epochs, acc, 'r.')
    # plt.plot(epochs, val_acc, 'r')
    # plt.title('Training and validation accuracy')
    #
    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    # plt.show()


def f1_score(predictions, labels):

    fs = np.zeros((4))          # 0-tp, 1-fp, 2-fn, 3-tn
    for i  in range(len(predictions)):
        if labels[i]==1 and predictions[i]==1:
            fs[0]+=1
        elif labels[i]==1 and predictions[i]==0:
            fs[2]+=1
        elif labels[i]==0 and predictions[i]==1:
            fs[1]+=1
        else:
            fs[3]+=1
    precision = fs[0]/(fs[0]+fs[1])
    recall = fs[0]/(fs[0]+fs[2])

    f1score = 2*precision*recall/(precision+recall)
    return f1score
        # print(predictions[i], labels[i])



class IntervalEvaluation(Callback):
    def __init__(self, validation_generator, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.validation_generator = validation_generator
        # self.X_val
        # self.y_val

    # def on_epoch_end(self, epoch, logs={}):
    #     if epoch % self.interval == 0:
    #         y_pred = self.model.predict_proba(self.X_val, verbose=0)
    #         score = roc_auc_score(self.y_val, y_pred)
    #         print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
    def custom_callback(self, epoch, logs):
        if epoch%self.interval==0:
            prediction = self.model.predict_generator(self.validation_generator, verbose=0)
            score = roc_auc_score(self.validation_generator.classes, prediction)
            print("interval evaluation - epoch: {:d} - score : {:6f}".format(epoch, score))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='./finalDataset/train/')
    a.add_argument("--val_dir", default='./finalDataset/val/')
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="vgg-ft4.model")
    a.add_argument("--plot", action="store_true", default=True)

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
