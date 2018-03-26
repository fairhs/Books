# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:49:30 2018

@author: us1cf
"""

import pandas as pd
import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# for Task 5
from sklearn.neural_network import MLPClassifier
# for Task 5-3
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

     # import the model
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
   
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

task = "Task5-EnsembleModel"

sv_file = str(task)+".csv"
#    top5_file = str(task)+"top5_.csv"
#import csv
# Test3-3
key1 = 'attr'  # attributes
key2 = 'a1'     # Training accuracy
key3 = 'a2' # Test accuracy
key4 = 'a3' # Training accuracy
key5 = 'a4'# Test accuracy
key6 ='best_param'  # Best parameter
key7 = 'max_iter' #max_iteration
key8 = 'test_param'   # Best parameter



def write_dict_to_csv(csv_file, dict_list):
    try:
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = [key1,key2,key3,key4,key5,key6,key7,key8]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(dict_list) 
    except IOError:
        print("I/O error", csv_file)
    return     
  


def EnsembleModelling(attr, max_iter_opt) :  
    
    max_iteration = max_iter_opt
    print(max_iteration)

    log_dict_list =[]   
           
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
       
    # preprocessing step
    df = preprocess_data(Loaded_df)
    
    # random state
    rs = 10
    
###################################################
    # target/input split
    y = df[attr]
    X = df.drop([attr], axis=1)

    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)
  
    dt_model = None
    log_reg_model=None
    nn_model=None
#################################################
    # grid search CV for decision tree
    params_dt = {'criterion': ['gini'],
              'max_depth': range(2, 5),
              'min_samples_leaf': range(40, 61, 5)}
    
    cv = GridSearchCV(param_grid=params_dt, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
    cv.fit(X_train, y_train)
    
    dt_model = cv.best_estimator_
    print(dt_model)
    
    # grid search CV for logistic regression
    params_log_reg = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    cv = GridSearchCV(param_grid=params_log_reg, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    log_reg_model = cv.best_estimator_
    print(log_reg_model)
    
    # grid search CV for NN
    params_nn = {'hidden_layer_sizes': [(3,), (5,), (7,), (9,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}
    
    cv = GridSearchCV(param_grid=params_nn, estimator=MLPClassifier(max_iter=max_iteration, random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    nn_model = cv.best_estimator_
    print(nn_model)   
#################################################
 
    # initialise the classifier with 3 different estimators
    voting = VotingClassifier(estimators=[('dt', dt_model), ('lr', log_reg_model), ('nn', nn_model)], voting='soft')
###########################################################################

    # fit the voting classifier to training data
    voting.fit(X_train, y_train)
    
    # evaluate train and test accuracy
    a1= voting.score(X_train, y_train)
    a2= voting.score(X_test, y_test)
    print("Ensemble train accuracy:", a1)
    print("Ensemble test accuracy:", a2)
    
    # evaluate ROC auc score
    y_pred_proba_ensemble = voting.predict_proba(X_test)
    roc_index_ensemble = 0 
    roc_index_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble[:, 1])
    print("ROC score of voting classifier:", roc_index_ensemble)        
    
    dict_data = {key1 : attr, key2 :a1 , key3: a2, key4 :'', key5 : ' ', key6 : roc_index_ensemble , key7 :max_iteration , key8 :' '}
    log_dict_list.append(dict_data)
############################################################
       
    print(sv_file)
    write_dict_to_csv(sv_file, log_dict_list)
#################################################################


##############################################################
#################################################################
#    
if __name__=='__main__':
 
    #maxiter = [50,200,400,600,800,1000,3000,5000,7000,10000]
    #maxiter = [200,400,600,800]
    maxiter=[300]
    attr_list = ['BILL', 'AGEGRP2', 'CLASS','AFFL', 'ORGANIC' ] 
    for j in range(len(attr_list)):            
        for i in range(len(maxiter)) :      
            print(maxiter[i] , attr_list[j])
            EnsembleModelling(attr_list[j], maxiter[i])        
            



