# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:27:31 2018

@author: us1cf
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:54:46 2018

@author: us1cf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

import seaborn as sns

from sklearn.feature_selection import RFECV
#from pdm_tools import preprocess_data


'''
#import csv
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

'''

#from dm_tools import visualize_decision_tree, analyse_feature_importance 
task = "Task3-3-RFECV"


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
key12 ='b1'    # Best parameter
key13 ='b2'
key14 = 'c1' #  X_train_log.shape[1]
key15 = 'c3'
key16 = 'c2' #    c4 =rfe.n_features_
key17 = 'c4'

def write_dict_to_csv(csv_file, dict_list):
    try:
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = [key1,key2,key3,key4,key5,key6,key7,key8,key9,key10,key11,key12,key13,key14,key15,key16,key17]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(dict_list) 
    except IOError:
        print("I/O error", csv_file)
    return 



def TrainData(attr) :   
    
    
    log_dict_list =[]
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
    
    prepared_df=preprocess_data(Loaded_df)   
    prepared_df.info()
    
    sv_file = str(task)+".csv"
    top5_file = str(task)+"top5_.csv"
        
    rs =10
######################################
    # train test split
    y = prepared_df[attr]
    X = prepared_df.drop([attr], axis=1)
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

#############################################

   
    
    # initialise a standard scaler object
    scaler = StandardScaler()
    
    # visualise min, max, mean and standard dev of data before scaling
    print(str(attr)+"Before scaling\n-------------")
    for i in range(5):
        col = X_train[:,i]
        print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
              format(i, min(col), max(col), np.mean(col), np.std(col)))
    
    # learn the mean and std.dev of variables from training data
    # then use the learned values to transform training data
    X_train = scaler.fit_transform(X_train, y_train)
    
    print(str(attr)+"After scaling\n-------------")
    for i in range(5):
        col = X_train[:,i]
        print("Variable #{}: min {}, max {}, mean {:.2f} and std dev {:.2f}".
              format(i, min(col), max(col), np.mean(col), np.std(col)))
    
    # use the statistic that you learned from training to transform test data
    # NEVER learn from test data, this is supposed to be a set of dataset
    # that the model has never seen before
    X_test = scaler.transform(X_test)
    
#####################################################
    model = LogisticRegression(random_state=rs)    
    # fit it to training data
    model.fit(X_train, y_train)    
        
    print(str(attr)+" [1] LogisticRegression============================",)   
    a1 = model.score(X_train, y_train)
    a2= model.score(X_test, y_test)
    print("Train accuracy:", a1) # = model.score(X_train, y_train))
    print("Test accuracy:", a2 ) # = model.score(X_test, y_test))

    
    # classification report on test data
    y_pred = model.predict(X_test)
 #   print(d=classification_report(y_test, y_pred))    
    d=classification_report(y_test, y_pred) 
    print(d)

    ##############################################
    # grab feature importances from the model and feature name from the original X
    coef = model.coef_[0]
    feature_names = X.columns
    
    # sort them out in descending order
    indices = np.argsort(np.absolute(coef))
    indices = np.flip(indices, axis=0)
    
    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:20]
    top5_list=[]
    for i in indices:
        top5 = str(feature_names[i]) +" : " + str(coef[i])    
#        dict_data = {key1 : attr, key2 : '', key3: str(feature_names[i]), key4 : str(coef[i]) , key5 : '',key6 : '' , key7 :' ' , key8 :' '}
#        log_dict_list.append(dict_data)
        top5_data = {key1 : attr, key2 :feature_names[i] , key3:coef[i]}
        top5_list.append(top5_data)
        #top5_important
        print(top5)
    
        #print(feature_names[i], ':', coef[i])    
############################################ 
    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    # use all cores to tune logistic regression with C parameter
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    # test the best model
    print(str(attr)+" [2] GridSearchCV + LogisticRegression============================",)   
    a3=cv.score(X_train, y_train)
    a4=cv.score(X_test, y_test)
    print("Train accuracy:", a3) #cv.score(X_train, y_train))
    print("Test accuracy:", a4 ) # cv.score(X_test, y_test)) 
    
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print(cv.best_params_)

# print parameters of the        
#############################################   
    #transformation 
    # list columns to be transformed
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
    df_log = prepared_df.copy()
# 
    
#############################################    
    #transform the columns with np.log
    for col in columns_to_transform:
        df_log[col] = df_log[col].apply(lambda x: x+1)
        df_log[col] = df_log[col].apply(np.log)
    
    # plot them again to show the distribution
    plot_skewed_columns(df_log,attr,task)     
    
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



###########################
    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_log, y_train_log)
    
    # test the best model
    print(str(attr)+"[3] Train accuracy: Grid search CV + LogisticRegression",)
    a5 =cv.score(X_train_log, y_train_log)
    a6 = cv.score(X_test_log, y_test_log)
    print("Train accuracy:", a5)  
    print("Test accuracy:",  a6 ) 
 
    
    y_pred = cv.predict(X_test_log)
    print(classification_report(y_test_log, y_pred))
    
    # print parameters of the best model
    print(cv.best_params_)
    
###########################
###########################


    rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train, y_train) # run the RFECV
    
    # comparing how many variables before and after
    print(str(attr)+" [3] RFECV + LogisticRegression============================",)   
    c1 = X_train.shape[1]
    c2 = rfe.n_features_
    
    print("Original feature set", c1)
    print("Number of features after elimination", c2) 
 
    

##############################
    X_train_sel = rfe.transform(X_train)
    X_test_sel = rfe.transform(X_test)
    
    #########################
    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel, y_train)
    
    # test the best model

    print(str(attr)+" [4] Train accuracy: Grid search CV",)
    a7=cv.score(X_train_sel, y_train)
    a8 = cv.score(X_test_sel, y_test)
    print("Train accuracy:", a7)  
    print("Test accuracy:",  a8 )   
    
    
    y_pred = cv.predict(X_test_sel)
    print(classification_report(y_test, y_pred))
    
    # print parameters of the best model
    print(cv.best_params_) 
    b1=cv.best_params_
    #dict_data = {key1 : attr, key2 : 4, key3: str("Grid search CV"), key4 : a1, key5 : a2 ,key6 : '' , key7 :' ' , key8  : cv.best_params_}
   #write_dict_to_csv( sv_file, dict_data)
 #   log_dict_list.append(dict_data)
 
    
    
##########################
# running RFE + log transformation
    rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train_log, y_train_log) # run the RFECV on log transformed dataset
    
    # comparing how many variables before and after

    print(str(attr)+" [4] RFECV + LogisticRegression============================",)    
    c3= X_train_log.shape[1]
    c4 =rfe.n_features_
    print("Original feature set", c3)
    print("Number of features after elimination", c4)

 
    
    # select features from log transformed dataset
    X_train_sel_log = rfe.transform(X_train_log)
    X_test_sel_log = rfe.transform(X_test_log)
    
    # init grid search CV on transformed dataset
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel_log, y_train_log)
    
    # test the best model
    print(str(attr)+" [5] GridSearchCV + LogisticRegression ============================",)    
    a9= cv.score(X_train_sel_log, y_train_log)
    a10 =cv.score(X_test_sel_log, y_test_log)
    print("Train accuracy:", a9)  
    print("Test accuracy:",  a10 ) 
    
    y_pred_log = cv.predict(X_test_sel_log)
    print(classification_report(y_test_log, y_pred_log))
    
    # print parameters of the best model
    print(cv.best_params_)
    b2 =cv.best_params_
    dict_data = {key1 : attr, key2 : a1, key3: a3, key4 : a5, key5 : a7 ,key6 : a9 , key7 :a2 , key8  :a4, key9: a6, key10:a8, key11: a10 , key12: b1, key13 : b2 , key14: c1, key15 : c3 , key16 : c2 , key17 : c4}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
    write_dict_to_csv( sv_file, log_dict_list)
    
    
       
##########################   
    
    
    
    
######################################  
    
    
if __name__=='__main__':

    
# C:\prj\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
#  'precision', 'predicted', average, warn_for)   
    
    attr_list=['BILL','AGEGRP2','AFFL','ORGANICS']
    for i in range(len(attr_list)):
        TrainData(attr_list[i])