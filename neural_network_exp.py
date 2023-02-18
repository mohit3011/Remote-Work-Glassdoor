import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os,sys,re
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

import statistics
import time
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import random
from data_augmentation import *

#arg parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_augmentation', default='majority_oversampling')




class Dataset(torch.utils.data.Dataset):
  # 'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return X, y

class NN_Model(nn.Module):
  
    def __init__(self, num_labels, config=None, device=torch.device("cuda:0")):
        super(NN_Model, self).__init__()
        self.dense1 = nn.Linear(in_features=41, out_features=10) #Add ReLu in forward loop
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(in_features=10, out_features=num_labels) #Add softmax in forward loop
        self.device = device
        
    def forward(self, inputs, attention_mask=None, labels=None):

        X = inputs.to(self.device)
        X = F.relu(self.dense1(X.float()))
        X = self.dropout(X)
        X = F.log_softmax(self.dense2(X))
        return X

def save_models(epochs, model):
    torch.save(model.state_dict(), "bert_model_fold_{}.h5".format(epochs))
    print("Checkpoint Saved")


def train_loop(dataloaders, dataset_sizes,  num_classes, epochs=1):
    model = NN_Model(num_labels=num_classes)
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-08) # clipnorm=1.0, add later
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    model.to(device)
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
#                 scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(labels.long())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print(labels.long())
                    actual_labels = torch.max(labels.long(), 1)[1]
                    loss = criterion(outputs, actual_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
#                 running_loss += loss.item() * inputs.size(0)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == actual_labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_loss < best_loss:
#                 save_models(epoch,model)
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

  skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
  data_list = np.load('data_file.npy')
  label_list = np.load('label_file.npy')
  data_augmentation = parser.parse_args().data_augmentation

  print(data_list.shape)

  round_acc_arr = []
  round_rec_arr = []
  round_pre_arr = []
  round_f1_arr = []
  round_f1_micro_arr = []
  round_corr_arr_train = []
  round_corr_arr_test = []


  round_acc_cum = 0
  round_rec_cum = 0
  round_pre_cum = 0
  round_f1_cum = 0
  round_f1_micro_cum = 0

  round_range = 1000

  #-------------------------------------------


  for round_num in range(round_range):

    print("Round Number: ", round_num)
    acc_cum = 0
    rec_cum = 0
    pre_cum = 0
    f1_cum = 0
    f1_micro_cum = 0
    acc_arr = []
    rec_arr = []
    pre_arr = []
    f1_arr = []
    f1_micro_arr = []
    predicted_label_arr = []
    test_label_arr = []
    error_analysis = []
    fold_number = 1

    # Encode the Labels -------------------------------------------------------
    encoder = LabelEncoder()
    encoder.fit(label_list)
    encoded_labels = encoder.transform(label_list)

    class_weights_labels = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(encoded_labels),
                                        y = encoded_labels                                                    
                                    )
    num_classes = len(list(encoder.classes_))
    print("num_classes: ", num_classes)
    print(encoder.classes_)

    #-------------------------------------------------------
    
    # Transfer the labels to device
    encoded_labels = np.asarray(encoded_labels, dtype='int32')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #class_weights_labels = torch.tensor(class_weights_labels, dtype=torch.float, device=device)

    for train_index, test_index in skf.split(data_list, encoded_labels):
        print("Running fold #", fold_number)
        X_train, X_test = data_list[train_index], data_list[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
        
        # Apply SMOTE to Train, Test and Validation -----------------------------------------------
        X_train, y_train, corr_train = apply_smote(X_train, y_train, data_augmentation)
        X_validation, y_validation, corr_validation = apply_smote(X_validation, y_validation, data_augmentation)
        X_test, y_test, corr_test = apply_smote(X_test, y_test, data_augmentation)
        #------------------------------------------------------------------------------------
        
        y_train = to_categorical(y_train)
        y_validation = to_categorical(y_validation)
        metric_test = np.copy(y_test)
        y_test = to_categorical(y_test)

        training_set = Dataset(X_train, y_train)
        validation_set = Dataset(X_validation, y_validation)
        test_set = Dataset(X_test, y_test)

        dataloaders = {
            'train' : torch.utils.data.DataLoader(training_set, batch_size=4,
                                                 shuffle=True, num_workers=2, drop_last=True),
            'validation' : torch.utils.data.DataLoader(validation_set, batch_size=4,
                                                 shuffle=True, num_workers=2, drop_last=True)
        }

        dataset_sizes = {
            'train': len(training_set),
            'validation': len(validation_set),
        }

        print(len(training_set))
        print(len(validation_set))
        print(len(test_set))


        model = train_loop(dataloaders, dataset_sizes, num_classes, epochs=35)
        

        y_pred = np.array([])

        for i in tqdm(range(len(test_set))):
            inputs = torch.Tensor([test_set[i][0]]).to(device)
            model.eval()
            outputs = model(inputs)
            preds = torch.max(outputs, 1)[1]
            y_pred = np.append(y_pred, preds.cpu().numpy())

        acc_arr.append(accuracy_score(metric_test, y_pred))
        acc_cum += acc_arr[fold_number-1]
        rec_arr.append(recall_score(metric_test, y_pred, average='macro'))
        rec_cum += rec_arr[fold_number-1]
        pre_arr.append(precision_score(metric_test, y_pred, average='macro'))
        pre_cum += pre_arr[fold_number-1]
        f1_arr.append(f1_score(metric_test, y_pred, average='macro'))
        f1_cum  += f1_arr[fold_number-1]
        f1_micro_arr.append(f1_score(metric_test, y_pred, average='micro'))
        f1_micro_cum  += f1_micro_arr[fold_number-1]
        fold_number+=1



    round_acc_cum += acc_cum/5
    round_acc_arr.append(acc_cum/5)

    round_rec_cum += rec_cum/5
    round_rec_arr.append(rec_cum/5)
    
    round_pre_cum += pre_cum/5
    round_pre_arr.append(pre_cum/5)
    
    round_f1_cum += f1_cum/5
    round_f1_arr.append(f1_cum/5)
    
    round_f1_micro_cum += f1_micro_cum/5
    round_f1_micro_arr.append(f1_micro_cum/5)

    #print(corr_train)
    round_corr_arr_train.append(corr_train)
    round_corr_arr_test.append(corr_test)



  round_corr_arr_train = np.array(round_corr_arr_train)
  round_corr_arr_test = np.array(round_corr_arr_test)

  print("Train Correlation: ", np.mean(round_corr_arr_train,axis=0))
  print("Test Correlation: ", np.mean(round_corr_arr_test,axis=0))

  print("Accuracy: ", round_acc_cum/round_range)
  print("Recall: ", round_rec_cum/round_range)
  print("Precision: ", round_pre_cum/round_range)
  print("F1 score(macro): ", round_f1_cum/round_range)
  print("F1 score(micro): ", round_f1_micro_cum/round_range)

  print("Accuracy_stdev: ", statistics.stdev(round_acc_arr))
  print("Recall_stdev: ", statistics.stdev(round_rec_arr))
  print("Precision_stdev: ", statistics.stdev(round_pre_arr))
  print("F1(macro) score_stdev: ", statistics.stdev(round_f1_arr))
  print("F1(micro) score_stdev: ", statistics.stdev(round_f1_micro_arr))