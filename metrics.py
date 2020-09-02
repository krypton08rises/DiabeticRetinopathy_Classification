import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import os

path = r'./ft_results30.csv'

def metric(path):
    df = pd.read_csv(path)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    test_y = []
    lr_probs = []
    for index, row in df.iterrows():
        if row['Predictions'] == 1 and row['Label'] == 1:
            tp += 1
        elif row['Predictions'] == 1 and row['Label'] == 0:
            fp += 1
        elif row['Predictions'] == 0 and row['Label'] == 1:
            fn += 1
        else:
            tn += 1
        test_y.append(row['Label'])
        lr_probs.append(row['Predictions'])

    print(len(test_y), len(lr_probs))
    lr_probs = np.array(lr_probs)
    lr_auc = roc_auc_score(test_y, lr_probs)
    print('Logistic: ROC AUC=%.3f' % lr_auc)
    lr_fpr, lr_tpr, _ = roc_curve(test_y, lr_probs)
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    pyplot.legend()
    pyplot.show()
    # print(tp,fp,fn,tn)

    specificity = tn / (fp + tn)
    sensitivity = tp / (tp + fn)
    print(specificity, sensitivity)

metric(path)


# AUC for a binary classifier
# def auc(y_true, y_pred):
#     ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
#     pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
#     pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
#     binSizes = -(pfas[1:]-pfas[:-1])
#     s = ptas*binSizes
#     return K.sum(s, axis=0)


# PFA, prob false alert for binary classifier
# def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
#     y_pred = K.cast(y_pred >= threshold, 'float32')
#     # N = total number of negative labels
#     N = K.sum(1 - y_true)
#     # FP = total number of false alerts, alerts from the negative class labels
#     FP = K.sum(y_pred - y_pred * y_true)
#     return FP/N


# P_TA prob true alerts for binary classifier
# def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
#     y_pred = K.cast(y_pred >= threshold, 'float32')
#     # P = total number of positive labels
#     P = K.sum(y_true)
#     # TP = total number of correct alerts, alerts from the positive class labels
#     TP = K.sum(y_pred * y_true)
#     return TP/P