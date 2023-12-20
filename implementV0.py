from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np
import preprocessingV1

def train_one_epoch(model,device,optimizer,criterion,train_loader,train_loss):
    model.train()
    trainingloss = 0
    for i, (x_batch,y_batch) in enumerate(train_loader):    
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)

        loss = criterion(output,y_batch.ravel())
        trainingloss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainingloss = trainingloss/(i+1)
    train_loss.append(trainingloss)

def validate_one_epoch(model,device,validation_loader,val_acc):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for x_batch,y_batch in validation_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            _,predicted = torch.max(outputs,1)

            n_samples += y_batch.size(0)
            n_correct += (predicted == y_batch.ravel()).sum().item()
        val_acc.append(n_correct/n_samples)

def test(model,device,true_label,test_loader,test_score,class_accuracy,test_cf):
    model.eval()
    with torch.no_grad():
        predicted_label = []
        for x_batch,y_batch in test_loader:
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            _,predicted = torch.max(outputs,1)

            predicted_label.append(predicted.cpu().numpy())
    
    predicted_label = np.concatenate([p for p in predicted_label])

    acc = round(accuracy_score(true_label,predicted_label),4)
    b_acc = round(balanced_accuracy_score(true_label,predicted_label),4)
    
    print(f'Accuracy : {acc}')
    print(f'Balanced accuracy: {b_acc}')
    
    print(f'Predicted labels: {np.unique(predicted_label,return_counts=True)}')
    print(f'True labels: {np.unique(true_label,return_counts=True)}')
    print('')

    c = confusion_matrix(true_label,predicted_label)
    n_class_predictions = np.sum(c,axis = 1)
    
    test_score.append([acc,b_acc])
    class_accuracy.append([round(c[i,i]/n_class_predictions[i],2) for i in range(len(n_class_predictions))])
    test_cf.append(c)
    
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay

def plot_test_result(scale,scale_trainloss,scale_validationacc,scale_classaccuracy,scale_testscore,scale_testcf):
    _,((ax1,ax2,ax3,ax4)) = plt.subplots(1,4,figsize = (16,4))
    w = 0.2
    i = 0

    color = ['r','g','b','y']
    ax1_custom_legends = []

    for s in scale:
        
        ax1.plot(scale_trainloss[s], color[i], label = f'train_loss - {s}' )
        ax1.plot(scale_validationacc[s], f'{color[i]}--', label = f'val_acc - {s}')
        ax1_custom_legends.append(Line2D([0], [0], color=color[i], lw=4))

        h = scale_classaccuracy[s][0]
        ax2.bar(np.arange(0,7)+i*w*np.ones(7),h,w,label = f'{s}')

        h = scale_testscore[s][0]
        ax3.bar(np.arange(0,2)+i*w*np.ones(2),h,w,label = f'{s}',)
        i+=1
    ax1_custom_legends.append(Line2D([0], [0], color='black', linestyle = '--', lw=4))
    ax1_custom_legends.append(Line2D([0], [0], color='black', linestyle = '-', lw=4))
    ax1.set_xlabel('Epochs')
    ax1.set_title('Training loss and Validation accuracy')
    ax2.set_title('Class Accuracy')
    ax2.set_xticks(np.arange(0,7),preprocessingV1.labels_dict.keys())
    ax3.set_title('Test score')

    x = ConfusionMatrixDisplay(scale_testcf[s][0],display_labels=preprocessingV1.labels_dict.keys())
    x.plot(ax=ax4,colorbar = False)
    ax4.set_title('Confusion Matrix')

    ax1.legend()
    ax2.legend()
    ax3.set_xticks(np.arange(0,2),labels = ['Accuracy','Balanced Accuracy'])

    plt.suptitle('Model performance w.r.t 4 scaling strategy',y = 1.1)


import torch.nn as nn
import pandas as pd

def make_prediction(wave_array,ana,model,device,window_size=1024,hop_length=256,mode = 'mlp',scope = 5):

    # Prepare data
    input, true_label = preprocessingV1.generate_data(wave_array,ana,window_size=window_size,hop_length=hop_length,mode=mode)

    # Make dense prediction
    predicted_label = []
    for i in range(input.shape[0]):
        x = torch.from_numpy(input[i,:]).unsqueeze(0).float().to(device)
        prediction = torch.argmax(model(x),dim=1).cpu().item()
        predicted_label.append(prediction)
    
    print('Accuracy: ', round(accuracy_score(true_label,predicted_label),2))
    print('Balanced accuracy: ', round(balanced_accuracy_score(true_label,predicted_label),2))
    # return predicted_label    

    # ================ IN DEVELOPMENT ==================
    # # Inspect surrounding signals and follows major votes 
    # aggregation = predicted_label[0:3]
    # for i in range(3,len(predicted_label)):
    #     l, c = np.unique(predicted_label[i-3:i+1],return_counts=True)
    #     aggregation.append(l[np.argmax(c)])
    # predicted_label = aggregation
        
    # # Inspect surrounding signals and follows major votes
    # scope = 5    
    # for i in range(3,len(predicted_label)):
    #     left, right = int(np.floor(scope/2)), int(np.ceil(scope/2))
    #     l, c = np.unique(predicted_label[left:right],return_counts=True)
    #     corrected_label = l[np.argmax(c)]
    #     predicted_label[i] = corrected_label
    # # ===================================================
        
    # Write result in form of analysis files
    n_windows = input.shape[0]
    time = [] # Make time marks
    for i in range(n_windows):
        time.append((window_size+i*hop_length)/100)
    time = [0] + time 
    predicted_label = [predicted_label[0]] + predicted_label # merge the predicted labels
    
    predicted_label = pd.Series(predicted_label).map({0:1,1:2,2:4,3:5,4:6,5:8,6:7}).tolist()

    ana_label = [] # analysis file
    ana_time = [time[0]]

    pin = 0 # Merge consecutive commonly labeled intervals into one
    for i in range(n_windows):
        if predicted_label[i] != predicted_label[pin]:
            ana_label.append(predicted_label[pin])
            ana_time.append(time[i])
            pin = i
    ana_label.append(predicted_label[n_windows])

    ana_time.append(time[i])
    ana_label += [12]

    predicted_analysis = pd.DataFrame({'label':ana_label,'time':ana_time})
    
    return predicted_analysis
    



