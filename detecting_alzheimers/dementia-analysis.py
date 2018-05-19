"""
This code was developed by Axo Sal and Taras Kucherenko

It is using the dataset [MRI and alzheimer's] (https://www.kaggle.com/jboysen/mri-and-alzheimers)
"""

import csv
import pandas as pd
import numpy as np

def test_random(df):
    """
    The most silly solution - return random label for each patient
    :param df: the Pandas DataFrame with the dataset
    :return:   accuracy on this dataset
    """
    
    # Split the dataset into the test and train
    rand = np.random.rand(len(df))
    msk = rand < 0.9

    train = df[msk]
    test = df[~msk]

    true_labels = test[['CDR']].values
    length = len(test)

    possible_values = [0.0, 0.5, 1.0, 2.0]
    # 0 = Normal, 0.5 = Very Mild Dementia, 1 = Mild Dementia, 2 = Moderate Dementia, 3 = Severe Dementia

    random_labels = np.random.choice(possible_values, length)

    # Count how many times we were correct
    accurate = 0

    for i in range(length):
       if random_labels[i] == true_labels[i]:
           accurate += 1

    accuracy = accurate*1.0/length

    return accuracy

def test_mmse_stats(df):
    """
    Another primitive baseline - using pure statistics and rely on the MMSE score only (memory screening test)
    :param df: the Pandas DataFrame with the dataset
    :return:   accuracy on this dataset
    """

    rand = np.random.rand(len(df))
    msk = rand < 0.9

    train = df[msk]
    test = df[~msk]

    true_labels = test[['CDR']].values
    mmse_all = test[['MMSE']].values

    length = len(test)

    mmse_from_train_data = train[['MMSE']].values

    # Count how many time we classified correcylt
    accurate = 0

    for i in range(length):
        # Get mmse score of the current patient
        mmse = mmse_all[i]

        # Find all the patient in the training set with the same score
        mmse_mask = mmse_from_train_data == mmse
        similar_people = train[mmse_mask]

        # If there are no people with exactly the same score - use slightly higher score
        while(len(similar_people)==0):
            mmse+=1
            mmse_mask = mmse_from_train_data == mmse
            similar_people = train[mmse_mask]
            # print('For person'+str(i)+' we increased MMSE')

        # Get dimentia classes for those people
        cdr_query = similar_people[['CDR']].values
        np_query = np.array(cdr_query)
        np_query = cdr_query.reshape(len(np_query))

        # Classify based on the majority vote
        (values,counts) = np.unique(np_query,return_counts=True)
        ind=np.argmax(counts)
        label = values[ind]

        if label == true_labels[i]:
            accurate += 1

    accuracy = accurate*1.0/length

    return accuracy

def __main__():
    
    # Read the dataset
    with open('oasis_cross-sectional.csv') as csvfile:
       df = pd.read_csv(csvfile)

       # Filter out useless points
       df = df[df.CDR.notnull()]

    # Test random labels
    all_results = [test_random(df) for i in range(50)] * 100

    error_mean = np.mean(all_results)
    error_std = np.std(all_results)

    print("Using random labels gives error rate "+ str(error_mean)+ " +/-" +str(error_std))

    # Test statistical classification just based on MMSE score
    all_results = [test_mmse_stats(df) for i in range(50)] * 100

    error_mean = np.mean(all_results)
    error_std = np.std(all_results)

    print("Just using MMSE gives error rate "+ str(error_mean) + "+/-" + str(error_std))
  
__main__()
