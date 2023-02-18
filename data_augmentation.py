import numpy as np 
import sys, os, re, copy
import random
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr

def apply_gaussian_noise(data_list, label_list):
  original_dataset = np.copy(data_list)

  index_list = []
  for i in range(len(label_list)):
    if int(label_list[i])==1:
      index_list.append(i)
  
  for i in range(len(index_list)-1):
    for j in range(i+1, len(index_list)):
      
      rand_el = random.choice([i, j])
      max_num = np.max(data_list[index_list[rand_el]])
      min_num = np.min(data_list[index_list[rand_el]])
      if abs(min_num) >= max_num:
        gauss = np.random.normal(0,(float(max_num)/float(2)),data_list[index_list[i]].shape)
      else:
        gauss = np.random.normal(0,(float(abs(min_num))/float(2)),data_list[index_list[i]].shape)
      #print(gauss)

      new_data_row = data_list[index_list[rand_el]] + gauss

      new_data_row = np.reshape(new_data_row, (1, new_data_row.shape[0]))

      if i==0 and j==1:
        fake_dataset = np.copy(new_data_row)
      else:
        fake_dataset = np.append(fake_dataset, new_data_row , axis=0)  
        

      data_list = np.append(data_list, new_data_row , axis=0)
      label_list = np.append(label_list, 1)        
    
    return data_list, label_list, fake_dataset


def apply_majority_oversampling(data_list, label_list):

    original_dataset = np.copy(data_list)

    index_list = []
    for i in range(len(label_list)):
      if int(label_list[i])==1:
        index_list.append(i)
    
    for i in range(len(index_list)-1):
      for j in range(i+1, len(index_list)):
        ratio = random.random()
        while ratio==0 or ratio==1:
          ratio = random.random()

        new_data_row = ratio*data_list[index_list[i]] + (1-ratio)*data_list[index_list[j]]

        new_data_row = np.reshape(new_data_row, (1, new_data_row.shape[0]))

        if i==0 and j==1:
          fake_dataset = np.copy(new_data_row)
        else:
          fake_dataset = np.append(fake_dataset, new_data_row , axis=0)  
        

        data_list = np.append(data_list, new_data_row , axis=0)
        label_list = np.append(label_list, 1)        

    
    return data_list, label_list, fake_dataset


def compute_correlation(original_data, label_list, fake_data):

  corr_list = []

  majority_original = []

  for i in range(len(label_list)):
    if int(label_list[i])==1:
      majority_original.append(original_data[i])
  
  majority_original = np.array(majority_original)
  mean_majority_vec = np.mean(majority_original, axis=0)

  sum_corr = 0
  for i in range(fake_data.shape[0]):
    corr, _ = spearmanr(mean_majority_vec, fake_data[i])
    sum_corr += corr
  return sum_corr/float(fake_data.shape[0])

def apply_smote(data_list, label_list, oversampling_type='majority_oversampling'):

    #print("Before Count: ", Counter(label_list))
    
    # Apply majority oversampling or gaussina noise
    original_data_list = np.copy(data_list)
    original_label_list = copy.deepcopy(label_list)

    if oversampling_type=='majority_oversampling':
        data_list, label_list, fake_list = apply_majority_oversampling(data_list, label_list)
    else:
        data_list, label_list, fake_list = apply_gaussian_noise(data_list, label_list)
    
    #print(data_list.shape)
    # Apply SMOTE
    transformed_data_list = np.copy(data_list)
    #print("After Majority Oversampling Count: ", Counter(label_list))
    orig_shape = transformed_data_list.shape
    transformed_label_list = []
    '''for i in range(0,transformed_data_list.shape[0]):
            for j in range(12):
                transformed_label_list.append(int(label_list[i]))'''
    
    for i in range(0,transformed_data_list.shape[0]):
          transformed_label_list.append(int(label_list[i]))    

    #print("Original Shape: ", orig_shape)
    


    oversample = SMOTE(k_neighbors=1)
    transformed_data_list, transformed_label_list = oversample.fit_resample(transformed_data_list, transformed_label_list)
    #print(len(transformed_label_list))
    added_num = int(transformed_data_list.shape[0]) - int(data_list.shape[0]) 


    label_list = np.append(label_list, np.zeros(added_num))

    corr_num = compute_correlation(original_data_list, original_label_list, fake_list)

    #print("After Count: ", Counter(label_list))

    return transformed_data_list, label_list, corr_num