from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score,recall_score, confusion_matrix,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
import preprocessingV1

# ======================= MODEL ===========================
def train_test_validation_split(X,y,train_size = 0.7,validation_size = 0.1,test_size = 0.2,shuffle = True,stratify = False):
    t = train_size + validation_size
    r = validation_size/t
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = test_size,stratify=y)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = r,stratify=y_train)
    return x_train, x_val, x_test, y_train, y_val, y_test

def train(clf,x_train,x_val,y_train,y_val,result_list=[],cm_list = [],verbose = True):
        if verbose == True:
            print('Call: ',clf)
            print('Training ...')
        clf.fit(x_train,y_train)

        if verbose == True:
            print('Validating')
        y_pred = clf.predict(x_val)
        cf = confusion_matrix(y_pred,y_val)

        
        acc = round(accuracy_score(y_pred,y_val),3)
        balanced_acc = round(balanced_accuracy_score(y_pred,y_val),3)
        f1 = round(f1_score(y_pred,y_val,average='weighted',zero_division = 0),3)
        precision = round(precision_score(y_pred,y_val,average='weighted',zero_division = 0),3)
        recall = round(recall_score(y_pred,y_val,average='weighted',zero_division = 0),3)
        
        result_list.append([acc,balanced_acc,f1,precision,recall])
        cm_list.append(cf) 
        if verbose == True:
            print('Finished training')

def test(clf,x_test,y_test,result_list=[],cm_list = [],verbose = True):
        if verbose == True:
            print('Call: ',clf)
            print('Testing ...')
        y_pred = clf.predict(x_test)
        cf = confusion_matrix(y_pred,y_test)

        acc = round(accuracy_score(y_pred,y_test),3)
        balanced_acc = round(balanced_accuracy_score(y_pred,y_test),3)
        f1 = round(f1_score(y_pred,y_test,average='weighted',zero_division = 0),3)
        precision = round(precision_score(y_pred,y_test,average='weighted',zero_division = 0),3)
        recall = round(recall_score(y_pred,y_test,average='weighted',zero_division = 0),3)
        
        result_list.append([acc,balanced_acc,f1,precision,recall])
        cm_list.append(cf) 
        if verbose == True:
            print('Testing done')    

def plot_result(result_dict,scale = [None]):
    _,((ax1,ax2),(ax3,ax4)) = plt.subplots(1,4,figsize = (16,4),sharex = True,sharey = True)

    model_name = ['DecTree','LR','SVC','RF','GB','Ada','XGB']
    n_model = len(model_name)

    xtick = np.arange(0,n_model)
    w = 0.1
    i=0

    for s in scale:
        r = result_dict[s] 
        ax1.bar(xtick+i*w*np.ones(n_model),r['Balanced Accuracy'],width = w, label = f'{s}')
        ax2.bar(xtick+i*w*np.ones(n_model),r['f1'],width = w, label = f'{s}')
        ax3.bar(xtick+i*w*np.ones(n_model),r['Precision'],width = w, label = f'{s}')
        ax4.bar(xtick+i*w*np.ones(n_model),r['Recall'],width = w, label = f'{s}')

        i+=1 

    ax1.set_title('Balanced Accuracy')
    ax2.set_title('f1')
    ax3.set_title('Precision')
    ax4.set_title('Recall')
    ax3.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
    ax4.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
    ax1.legend()
    plt.suptitle('Model training scores w.r.t 4 scaling strategy on short data')
    plt.show()

def save(model,path):
    # save the model to disk
    pickle.dump(model, open(path, 'wb'))

# ======================= VISUALIZATION ===========================
def onevarplot(feature_matrix: np.ndarray, label: np.ndarray, var: int, n_obs: int = None, xlab: str = None, ylab: str = None):
    '''
        Input:
            feature_matrix: a matrix with 11 features calculated previously
            label: corresponding label to each row
            var: which variable (column to plot)
            xlab: label of axis x
            ylab: label of axis y
        Output:
            A colored plot of a column of feature_matrix
    '''
    plt.figure(figsize=(5,5))

    f1 = feature_matrix[:,var]
    split = []

    for i in range(len(classes)):
        split.append(f1[label == classes[i]])

    if not isinstance(n_obs,int):
        classes,counts = np.unique(label,return_counts=True)
        n_obs = min(counts)

    for i in range(len(classes)):
        plt.scatter(np.arange(0,n_obs),split[i][0:n_obs],label = classes[i])

    if ( isinstance(xlab,str) ):
        plt.xlabel(xlab)
    if ( isinstance(ylab,str) ):
        plt.ylabel(ylab)
        
    plt.legend(ncols = 3)

def twovarplot(feature_matrix: np.ndarray, label: np.ndarray, var1: int, var2: int, n_obs: int):
    plt.figure(figsize=(5,5))
    reverse_labels_dict = {0:'np',1:'c',2:'e1',3:'e2',4:'f',5:'pd',6:'g'}
    var = ['mean', 'sd', 'sk', 'zcr', 'hurst', 'energy', 'sample_entropy', 
                'permutation_entropy', 'spectral_entropy', 'spectral_centroid', 'spectral_flatness']
    classes = np.unique(label)

    f1 = feature_matrix[:,var1]; f2 = feature_matrix[:,var2]
    split1 = []; split2 = []

    for i in range(len(classes)):
        split1.append(f1[label == classes[i]])
        split2.append(f2[label == classes[i]])

    for i in range(len(classes)):
        plt.scatter(split1[i][0:n_obs],split2[i][0:n_obs],label = reverse_labels_dict[classes[i]])

    plt.xlabel(var[var1])
    plt.ylabel(var[var2])
    plt.legend()

def twovarplot_multiple(feature_matrix: np.ndarray, label: np.ndarray, n_obs: int = None, n_pairs: int = 6):

    reverse_labels_dict = {0:'np',1:'c',2:'e1',3:'e2',4:'f',5:'pd',6:'g'}
    
    var = ['mean', 'sd', 'sk', 'zcr', 'hurst', 'energy', 'sample_entropy', 
                'permutation_entropy', 'spectral_entropy', 'spectral_centroid', 'spectral_flatness']
    classes = np.unique(label)

    v = []
    for n in range(n_pairs):
        v.append(np.random.randint(0,11,2))

    fig, ax = plt.subplots(2,3,figsize = (5*3,5*2))

    n_ax = 0
    for var1, var2 in v:
        f1 = feature_matrix[:,var1]; f2 = feature_matrix[:,var2]
        split1 = []; split2 = []

        for i in range(len(classes)):
            split1.append(f1[label == classes[i]])
            split2.append(f2[label == classes[i]])

        for i in range(len(classes)):
            ax[n_ax//3,n_ax%3].scatter(split1[i][0:n_obs],split2[i][0:n_obs],label = reverse_labels_dict[classes[i]])

        ax[n_ax//3,n_ax%3].set_xlabel(var[var1])
        ax[n_ax//3,n_ax%3].set_ylabel(var[var2])
        ax[n_ax//3,n_ax%3].legend()
        n_ax+=1 

