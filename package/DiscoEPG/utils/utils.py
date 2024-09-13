import numpy as np 
import pandas as pd
import datetime
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
wd = os.getcwd()

def write_training_log(model, config, result):
    
    os.makedirs(f'./log/{model.__arch__}', exist_ok= True)
    date = str(datetime.datetime.now())[:-7]
    columns = ['Date','Description', 'Version', 'Optimizer', 'Device', '#Epochs', 'Learning_rate', 'Batch_size'] \
                        + ['t_train', 't_train_epoch', 't_data'] \
                        + ['Train_loss', 'Train_accuracy', 'Val_loss', 'Val_acc', 'Test_acc', 'Test_f1', 'Test_precision', 'Test_recall'] \
                        

    # Write scoring results in a .csv file
    session_result_path = f'./log/{model.__arch__}/session_result.csv'
    if os.path.exists(session_result_path):
        f = pd.read_csv(session_result_path, index_col = [0])
    else:
        f = pd.DataFrame(columns=columns)
    
    
    infos = [date, config.exp_name + '_' + config.method, model.__version__, config.optimizer, config.device, config.n_epochs, config.lr, config.batch_size]
    t = [result['training_time'], np.mean(result['per_epoch_training_time']), result['data_processing_time']]
    train_loss = result['training_loss'][-1]
    train_acc = result['training_accuracy'][-1]
    val_loss = result['validation_loss'][-1]
    val_acc = result['validation_accuracy'][-1]
    results = [train_loss, train_acc, val_loss, val_acc] + list(result['test_score'].values())

    test_label = list(result['test_class_accuracy'].keys())
    class_acc_cols = [f'{label}_acc' for label in test_label]
    columns += class_acc_cols
    class_acc = list(result['test_class_accuracy'].values())

    # print([infos + t + results + class_acc])
    # print(columns)
    f = pd.concat([f, pd.DataFrame([infos + t + results + class_acc],columns = columns)])
    f.columns = columns
    f.to_csv(session_result_path)

    # Write training log in a .txt file
    with open(f'./log/{model.__arch__}/session_log.txt','a') as f:

        f.writelines([
                    f'======================================================================================\n',
                    f'Date: {date} | Description: {config.exp_name} | Model version: {model.__version__}\n',
                    f'Optimizer: {config.optimizer} | Device: {config.device} | Epochs: {config.n_epochs} | Learning rate: {config.lr} | Batch size: {config.batch_size}\n',
                    f"Early stopping: {result['early_stopping_epoch']}\n",
                    f"Training loss: {' '.join([str(num) for num in np.round(result['training_loss'],2)])}\n",
                    f"Training accuracy: {' '.join([str(num) for num in np.round(result['training_accuracy'],2)])}\n",
                    f"Validation loss: {' '.join([str(num) for num in np.round(result['validation_loss'],2)])}\n",
                    f"Validation accuracy: {' '.join([str(num) for num in np.round(result['validation_accuracy'],2)])}\n",
                    f"Flatten cf: {' '.join([str(num) for num in result['test_confusion_matrix'].flatten()])}\n",
                    ])  


def plot_training_result(model, config, result, savefig = True):
    plt.rcParams.update({'font.size': 12})
    # Learning curves
    train_loss = result['training_loss']
    train_acc = result['training_accuracy']
    val_loss = result['validation_loss']
    val_acc = result['validation_accuracy']

    # test scores
    test_score = result['test_score']
    scores = list(result['test_score'].keys())
    
    # Confusion matrices
    cf = result['test_confusion_matrix']
    
    f, ax = plt.subplots(1,3,figsize = (14,5))
    ax[0].plot(train_loss,'r',label = 'Tr.loss')
    ax[0].plot(train_acc,'r--',label = 'Tr.acc')
    ax[0].plot(val_loss,'b', label = 'Val.loss')
    ax[0].plot(val_acc,'b--', label = 'Val.acc')
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Accuracy curve and loss curve.')
    ax[0].grid()
    ax[0].legend()

    w = 0.3
    h = [test_score[k] for k in scores]
    plt.bar
    ax[1].bar(np.arange(len(scores)), h, width = w)
    ax[1].set_xticks(np.arange(len(scores)),scores)
    ax[1].set_title('Test accuracy and weighted scores')
    ax[1].set_ylim(0,100)
    
    test_label = result['test_class_accuracy'].keys()
    sns.heatmap(np.round(cf*100,2), ax = ax[2], annot= True, cbar = False, xticklabels=test_label, yticklabels =test_label)
    ax[2].set_title('Confusion matrix')     
    ax[2].set_xlabel('Predicted label')
    ax[2].set_ylabel('True label')
    plt.suptitle(config.exp_name)
    plt.tight_layout()

    if savefig == True:
        os.makedirs(f'./log/{model.__arch__}', exist_ok= True)
        
        d = str(datetime.date.today())
        p = os.path.join(f'./log/{model.__arch__}/{config.exp_name}_{model.__arch__}_{d}.png')
        plt.savefig(p)      

def format_filenames(folder):
    if os.path.exists(f'./data/{folder}'):
        dir = os.listdir(f'./data/{folder}')
        for n in dir:
            if not n.startswith(folder):
                newname = folder + '_' + n
                os.rename(f'./data/{folder}/{n}', f'./data/{folder}/{newname}')
    else:
        raise RuntimeError(f"Path './data/{folder}' does not exist.")