import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import os

path = r'./ft_results30.csv'        # any csv file

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

# metric(path)