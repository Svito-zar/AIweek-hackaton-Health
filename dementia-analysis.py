import csv
import pandas as pd
import numpy as np
from sklearn import metrics, cross_validation
from sklearn.model_selection import cross_val_score

# Read the dataset
with open('oasis_cross-sectional.csv') as csvfile:
    #  reader = csv.reader(csvfile, delimiter=',')
    #  for row in reader:
    #      print(row)

     df = pd.read_csv(csvfile)

# Filter out useless points
df = df[df.CDR.notnull()]
targets = df[['CDR']].values

# print(len(df))

# Split the dataset



def test_random(train, test):
    
    # labels = test

    labels = test[['CDR']]
    length = len(test)

    possible_value = [0.0, 0.5, 1.0, 2.0]

    random_labels = np.random.choice(possible_value, length)

    # print(labels)
    # print(random_labels)

    cdr_array = labels.values

    accurate = 0

    for i in range(length):
       if random_labels[i] == cdr_array[i]:
           accurate += 1

    accuracy = accurate/length

def test_mmse_stats(df):

    rand = np.random.rand(len(df))
    msk = rand < 0.9

    train = df[msk]
    test = df[~msk]

    # labels = test

    labels = test[['CDR']].values
    mmse_all = test[['MMSE']].values


    length = len(test)

    mmse_from_train_data = train[['MMSE']].values

    # print(labels)
    # print(random_labels)

    accurate = 0

    mmse = []

    for i in range(length):
        mmse = mmse_all[i]

        mmse_mask = mmse_from_train_data == mmse

        NaN_mask = df['CDR'] == None

        # print(df['CDR'])
        print(NaN_mask)

        similar_people = train[mmse_mask]


        while(len(similar_people)==0):
            mmse+=1
            mmse_mask = mmse_from_train_data == mmse
            similar_people = train[mmse_mask]
            # print('For person'+str(i)+' we increased MMSE')

        cdr_query = similar_people[['CDR']].values 

        np_query = np.array( cdr_query)
        np_query = cdr_query.reshape(len(np_query))
        #print(np_query)

        (values,counts) = np.unique(np_query,return_counts=True)
        #print(counts)
        ind=np.argmax(counts)
        output = values[ind]
        

        if output == labels[i]:
            accurate += 1

    accuracy = accurate/length 

    return accuracy

# mmse_from_train_data = train[['MMSE']].values
# mmse_array = np.array(mmse_from_train_data)
# values = np.unique(mmse_array,return_counts=False)

# print(values)
    

# test_random(train, test)
all_results = [test_mmse_stats(df) for i in range(50)] * 100

# print(all_results)

error_mean = np.mean(all_results)
error_std = np.std(all_results)

# print(error_mean, "+/-", error_std)
  
# print(df)

# random_clas = RandomClassifier([0.0, 0.5, 1.0, 2.0])

# scores = cross_val_score(random_clas, df, targets, cv=10)