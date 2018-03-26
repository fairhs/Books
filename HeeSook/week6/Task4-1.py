# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:49:30 2018

@author: us1cf
"""

import pandas as pd
import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
# for Task 5
from sklearn.neural_network import MLPClassifier

#from pdm_tools import preprocess_data, plot_skewed_columns, write_dict_to_csv

dict_data = {}
log_dict_list =[]    

def preprocess_data(df):

    df['AGE'].fillna(round(df['AGE'].mean()), inplace=True)
    df['ORGANICS'].fillna(round(df['ORGANICS'].mean()), inplace=True)
    df['REGION']=df['REGION'].replace([' ', 'Midlands', 'North' ,'South East' ,'Scottish' ,'South West'], [0,1,2,3,4,5]).astype(float)
    df['REGION'].fillna(round(df['REGION'].mean()), inplace=True) 
    
    
    df['TV_REG']=df['TV_REG'].replace([' ','Wales & West', 'Midlands', 'N West', 'East', 'N East', 'London' ,'S & S East' ,'C Scotland', 'Ulster' ,'S West', 'Yorkshire' ,'Border', 'N Scot'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13]).astype(float)
    df['TV_REG'].fillna(round(df['TV_REG'].mean()), inplace=True) 

    
    df['AGEGRP2'] = df['AGEGRP2'].replace([' ','70-80', '40-50', '60-70','50-60', '30-40', '10-20', '20-30'], [0,1,2,3,4,5,6,7]).astype(float)
    df['AGEGRP2'].fillna(round(df['AGEGRP2'].mean()), inplace=True) 

    #['U' 'F' 'M' nan]    
    #df['REGION'].replace([' ', 'U' 'F' 'M'], [0,1,2,3]).astype(float)
    df['GENDER'] = df['GENDER'].replace([' ', 'U', 'F', 'M'], [0,1,2,3]).astype(float)
    df['GENDER'].fillna(1, inplace=True)
    
    #['Gold' 'Silver' 'Tin' 'Platinum']    
    df['CLASS'] =df['CLASS'].replace(['Gold', 'Silver', 'Tin' ,'Platinum'], [3,2,1,4]).astype(float)

    #NEIGHBORHOOD   #################################
    df['NEIGHBORHOOD'].fillna(round(df['NEIGHBORHOOD'].mean()), inplace=True)


    #AGE   #################################
    df['AGE'].fillna(round(df['AGE'].median()), inplace=True)

    #[0 1 2 3]
    df['ORGANICS'] =df['ORGANICS'].astype(float)
    #[0 1 ]
    df['ORGYN'] =df['ORGYN'].astype(float)
    
    bill_median = round(df['BILL'].median())


    df['BILL'] = np.where( (df['BILL'] > bill_median), 1, 0 )
#    print(df['BILL'] .unique())

    #AFFL
    df['AFFL'].fillna(round(df['AFFL'].mean(),2), inplace=True)
    df['AFFL'] = round((df['AFFL']) / 10)+1   
    

    
    df.drop(['DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' , 'AGEGRP1'], axis=1, inplace=True)
    #df.drop(['NEIGHBORHOOD', 'DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' , 'AGEGRP1'], axis=1, inplace=True)
    
    df = pd.get_dummies(df)
    
    return df

def plot_skewed_columns(df , attr ,task):
    # setting up subplots for easier visualisation
    fig =plt.figure()
    fig, axes = plt.subplots(2,4, figsize=(10,10), sharex=False)

    # gift avg plots
    sns.distplot(df['BILL'].dropna(), hist=False, ax=axes[0,0])
    sns.distplot(df['AGEGRP2'].dropna(), hist=False, ax=axes[0,1])
    sns.distplot(df['ORGYN'].dropna(), hist=False, ax=axes[1,0])
    sns.distplot(df['REGION'].dropna(), hist=False, ax=axes[1,1])

    # gift cnt plots
    sns.distplot(df['AFFL'].dropna(), hist=False, ax=axes[0,2])
    sns.distplot(df['GENDER'].dropna(), hist=False, ax=axes[0,3])
    sns.distplot(df['ORGANICS'].dropna(), hist=False, ax=axes[1,2])
    sns.distplot(df['NEIGHBORHOOD'].dropna(), hist=False, ax=axes[1,3])

    fig.savefig(str(task) + "_" + str(attr) +".png")
    plt.show()
task = "Task4-1"

sv_file = str(task)+".csv"
#    top5_file = str(task)+"top5_.csv"
#import csv
# Test3-3
#import csv
#    dict_data[0] = {key1 : attr, key2 : a1, key3: a3, key5 : a2, key6: a4,  key8 : b1, key9 : b2}
#    
#    dict_data[0].update({ key4 : a5, key7 : a6, key10: b3, key11 :max_iter_opt })
key1 = 'attr'  # attributes
key2 = 'a1'     # Training Accuracy
key3 = 'a3' 
key4 = 'a5' 
key5 = 'a7' 
key6 ='a9'    
key7 = 'a2' # Test Accuracy 
key8 = 'a4'
key9 = 'a6'
key10 = 'a8'
key11 = 'a10' 
key12 ='test_param'   # Best parameter
def write_dict_to_csv(csv_file, dict_list):
    try:
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = [key1,key2,key3,key4,key5,key6,key7,key8,key9,key10,key11,key12]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(dict_list) 
    except IOError:
        print("I/O error", csv_file)
    return       


def NN_TrainData(attr, max_iter_opt):
    
    max_iteration= max_iter_opt
    print(max_iter_opt)

      
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
       
    # preprocessing step
    df = preprocess_data(Loaded_df)
    
    # random state
    rs = 10
    
    # train test split
    y = df[attr]
    X = df.drop([attr], axis=1)
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)
    
 ###################################################
    model = MLPClassifier(max_iter=max_iteration, random_state=rs)
    model.fit(X_train, y_train)
    
    print("[1] Train accuracy: MLPClassifier")
    a1 = model.score(X_train, y_train)
    a2=  model.score(X_test, y_test)
    
    print("Train accuracy:", a1) # = model.score(X_train, y_train))
    print("Test accuracy:", a2 ) # = model.score(X_test, y_test))    
    
    
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print(model)
############################################################    
    print(X_train.shape) 
    b1= X_train.shape
    #log_dict_list.append(dict_data)     

############################################################
    params = {'hidden_layer_sizes': [(3,), (5,), (7,),(9,)] , 'alpha' : [0.01, 0.001, 0.001, 0.0001]}
    
    cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=max_iteration, random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)

    
    print("[2]Train accuracy: GSCV+MLPClassifier")
    a3= cv.score(X_train, y_train)
    a4=cv.score(X_test, y_test)
    print("Train accuracy:", a3) 
    print("Test accuracy:", a4 )     
    
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print(cv.best_params_)
    b2=cv.best_params_
    
    
    #key 2,3,4  - key 5,6,7 - 
    dict_data[0] = {key1 : attr, key2 : a1, key3: a3, key5 : a2, key6: a4, key8 : b1, key9 : b2}

#################################################################


def NN_TrainData_LogTrans(attr, max_iter_opt) :  
    
    max_iteration = max_iter_opt
    print(max_iteration)
     
    log_dict_list =[]
#    sv_file = str(task)+".csv"
#    top5_file = str(task)+"top5_.csv"
           
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
       
    # preprocessing step
    df = preprocess_data(Loaded_df)
    
    # random state
    rs = 10

    print(attr)
    if attr == 'BILL' :
        columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'NEIGHBORHOOD',
                        'ORGANICS', 'CLASS', 'REGION', 'AFFL']
    elif attr == 'GENDER' :
    #print(attr)
        columns_to_transform = ['BILL', 'AGEGRP2', 'TV_REG', 'NEIGHBORHOOD',
                        'ORGANICS', 'CLASS', 'REGION', 'AFFL']    
    elif attr == 'AGEGRP2' :
    #print(attr)
        columns_to_transform = ['GENDER', 'BILL', 'TV_REG', 'NEIGHBORHOOD',
                        'ORGANICS', 'CLASS', 'REGION', 'AFFL']    
    elif attr == 'TV_REG' :
    #print(attr)
        columns_to_transform = ['GENDER', 'AGEGRP2', 'BILL', 'NEIGHBORHOOD',
                        'ORGANICS', 'CLASS', 'REGION', 'AFFL']    
    elif attr == 'NEIGHBORHOOD' :
    #print(attr)
        columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'ORGANICS', 'CLASS', 'REGION', 'AFFL']    
    elif attr == 'ORGANICS' :
    #print(attr)
        columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'BILL', 'CLASS', 'REGION', 'AFFL']    
    elif attr == 'REGION' :
    #print(attr)
        columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'ORGANICS', 'CLASS', 'NEIGHBORHOOD', 'AFFL']    
    elif attr == 'AFFL' :
    #print(attr)
        columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'ORGANICS', 'CLASS', 'REGION', 'NEIGHBORHOOD']    
    elif attr == 'ORGYN' :
         #print(attr)
         columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'ORGANICS', 'CLASS', 'REGION', 'NEIGHBORHOOD']   
    elif attr == 'CLASS' :
         #print(attr)
         columns_to_transform = ['GENDER', 'AGEGRP2', 'TV_REG', 'BILL',
                        'ORGANICS', 'ORGYN', 'REGION', 'NEIGHBORHOOD']   
    # copy the dataframe
    df_log = df.copy()
    
    # transform the columns with np.log
    for col in columns_to_transform:
        df_log[col] = df_log[col].apply(lambda x: x+1)
        df_log[col] = df_log[col].apply(np.log)
        
    # create X, y and train test data partitions
    y_log = df_log[attr]
    X_log = df_log.drop([attr], axis=1)
    X_mat_log = X_log.as_matrix()
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                        random_state=rs)
    
    # standardise them again
    scaler_log = StandardScaler()
    X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
    X_test_log = scaler_log.transform(X_test_log)


############################################################
    params = {'hidden_layer_sizes': [(3,), (5,), (7,),(9,)] , 'alpha' : [0.01, 0.001, 0.001, 0.0001]}
    
#    print(1)
    cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=max_iteration, random_state=rs), cv=10, n_jobs=-1)
#    print(1-1)
    cv.fit(X_train_log, y_train_log)
    

    
    print("[3]Train accuracy: GSCV+MLPClassifier")
    a5=  cv.score(X_train_log, y_train_log)
    a6= cv.score(X_test_log, y_test_log)
    print("Train accuracy:", a5) 
    print("Test accuracy:", a6 )    
    
    y_pred = cv.predict(X_test_log)
    print(classification_report(y_test_log, y_pred))
    
    print(cv.best_params_)
    b3=cv.best_params_
    
    
    #key 2,3,4  - key 5,6,7 - 
    #dict_data[0] = {key1 : attr, key2 : a1, key3: a3, key5 : a2, key6: a4,  key8 : b1, key9 : b2}
    dict_data[0].update({ key4 : a5, key7 : a6, key10: b3, key11 :max_iter_opt })
    print(dict_data[0])
    log_dict_list.append(dict_data[0])     
    write_dict_to_csv( sv_file, log_dict_list)
#################################################################

#def a():
#    dict_data[0] = { key1: 33 , key2: 44}
#    #log_dict_list.append(dict_data)
#    #print(dict_data)
#
#def b():
#    dict_data[0].update({key3:55})    
#    log_dict_list.append(dict_data[0])
##############################################################
#################################################################
#    
if __name__=='__main__':

    maxiter = [50,200] #,400,600,800,1000,3000,5000,7000,10000]
    #maxiter = [11] # for test     
    attr_list=['BILL','AGEGRP2','AFFL','ORGANICS']
    for i in range(len(attr_list)):
        log_dict_list=[]
        print(log_dict_list)
        for j in range(len(maxiter)):
            print(maxiter[j]) 
            NN_TrainData(attr_list[i],maxiter[j])
            NN_TrainData_LogTrans(attr_list[i],maxiter[j])

