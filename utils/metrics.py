from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from easydict import EasyDict
import numpy as np 


def scoring(true_label, predicted_label):

    acc = np.round(accuracy_score(true_label,predicted_label),2)
    f1 = np.round(f1_score(true_label, predicted_label, average= 'macro',zero_division = 0),2)
    precision = np.round(precision_score(true_label, predicted_label, average= 'macro',zero_division = 0),2)
    recall = np.round(recall_score(true_label, predicted_label, average= 'macro',zero_division = 0),2) 
    c = np.round(confusion_matrix(true_label,predicted_label,normalize = 'pred'),2)
    
    return {'scores': EasyDict({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}), 'confusion_matrix': c}