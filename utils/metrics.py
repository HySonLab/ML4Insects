from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from easydict import EasyDict
import numpy as np 


def top_k_accuracy(pred_proba, true_label, k = 2):
    n = len(true_label)
    n_class = len(pred_proba[0]) - 1
    n_correct = 0
    for i in range(n):
        top_k = np.argsort(pred_proba[i])[:n_class-k:-1]
        if true_label[i] in top_k:
            n_correct +=1 
    return n_correct/n 

def scoring(true_label, predicted_label):

    acc = np.round(accuracy_score(true_label,predicted_label),4)
    f1 = np.round(f1_score(true_label, predicted_label, average= 'macro',zero_division = 0),4)
    precision = np.round(precision_score(true_label, predicted_label, average= 'macro',zero_division = 0),4)
    recall = np.round(recall_score(true_label, predicted_label, average= 'macro',zero_division = 0),4) 
    c = np.round(confusion_matrix(true_label,predicted_label, normalize='pred'),4)
    
    return {'scores': EasyDict({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}), 'confusion_matrix': c}
