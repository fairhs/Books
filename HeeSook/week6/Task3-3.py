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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

import seaborn as sns

from sklearn.feature_selection import RFECV


'''
#import csv
key1 = 'attr'  # attributes
key2 = 'no'     # Number of training
key3 = 'des' # description of training 
key4 = 'TrainAcc' # Training accuracy
key5 = 'TestAcc'# Test accuracy

key6 ='originf'  # orignial feature set  
key7 = 'afterEliminate' #'after elimination 
ket8 = 'test_param'   # Best parameter

'''

#from dm_tools import visualize_decision_tree, analyse_feature_importance 
task = "Task3-3"

def TrainData(attr) :   
    
    
    log_dict_list =[]
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
    
    prepared_df=preprocess_data(Loaded_df)   
    prepared_df.info()
    
    sv_file = str(task)+".csv"
        
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
    dict_data = {key1 : attr, key2 : 1, key3: str("LogisticRegression"), key4 : a1, key5 : a2,key6 : '' , key7 :' ' , key8 :' '}
    #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
    
    # classification report on test data
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))    
    ##############################################
    # grab feature importances from the model and feature name from the original X
    coef = model.coef_[0]
    feature_names = X.columns
    
    # sort them out in descending order
    indices = np.argsort(np.absolute(coef))
    indices = np.flip(indices, axis=0)
    
    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:20]
    
    for i in indices:
        print(feature_names[i], ':', coef[i])    
############################################ 
    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    # use all cores to tune logistic regression with C parameter
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    # test the best model
    print(str(attr)+" [2] GridSearchCV + LogisticRegression============================",)   
    a1=cv.score(X_train, y_train)
    a2=cv.score(X_test, y_test)
    print("Train accuracy:", a1) #cv.score(X_train, y_train))
    print("Test accuracy:", a2 ) # cv.score(X_test, y_test))
    dict_data = {key1 : attr, key2 : 2, key3: str("GridSearchCV + LogisticRegression"), key4 : a1, key5 : a2, key6 : '' , key7 :' ' , key8 :' '}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
 
    
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
    a1 =cv.score(X_train_log, y_train_log)
    a2 = cv.score(X_test_log, y_test_log)
    print("Train accuracy:", a1)  
    print("Test accuracy:",  a2 ) 
 
    
    y_pred = cv.predict(X_test_log)
    print(classification_report(y_test_log, y_pred))
    
    # print parameters of the best model
    print(cv.best_params_)
    dict_data = {key1 : attr, key2 : 3, key3: str("Grid search CV"), key4 : a1, key5 : a2,key6 : '' , key7 :' ' , key8  : cv.best_params_}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
 
    
###########################
###########################


    rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train, y_train) # run the RFECV
    
    # comparing how many variables before and after
    print(str(attr)+" [4] RFECV + LogisticRegression============================",)   
    print("Original feature set", X_train.shape[1])
    print("Number of features after elimination", rfe.n_features_) 
    dict_data = {key1 : attr, key2 : 4, key3: str("RFECV + LogisticRegression"), key4 : '', key5 : '',key6 : X_train.shape[1] , key7 :rfe.n_features_ , key8 :' '}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
 
    

##############################
    X_train_sel = rfe.transform(X_train)
    X_test_sel = rfe.transform(X_test)
    
    #########################
    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel, y_train)
    
    # test the best model

    print(str(attr)+" [5] Train accuracy: Grid search CV",)
    a1=cv.score(X_train_sel, y_train)
    a2 = cv.score(X_test_sel, y_test)
    print("Train accuracy:", a1)  
    print("Test accuracy:",  a2 )   
    
    
    y_pred = cv.predict(X_test_sel)
    print(classification_report(y_test, y_pred))
    
    # print parameters of the best model
    print(cv.best_params_) 
    dict_data = {key1 : attr, key2 : 5, key3: str("Grid search CV"), key4 : a1, key5 : a2 ,key6 : '' , key7 :' ' , key8  : cv.best_params_}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
 
    
    
##########################
# running RFE + log transformation
    rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train_log, y_train_log) # run the RFECV on log transformed dataset
    
    # comparing how many variables before and after

    print(str(attr)+" [6] RFECV + LogisticRegression============================",)    
    print("Original feature set", X_train_log.shape[1])
    print("Number of features after elimination", rfe.n_features_)
    dict_data = {key1 : attr, key2 : 6, key3: str("RFECV + LogisticRegression (Log Trans)"), key4 : '', key5 : '',key6 : X_train_log.shape[1], key7 : rfe.n_features_ , key8 :' '}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
 
    
    # select features from log transformed dataset
    X_train_sel_log = rfe.transform(X_train_log)
    X_test_sel_log = rfe.transform(X_test_log)
    
    # init grid search CV on transformed dataset
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel_log, y_train_log)
    
    # test the best model
    print(str(attr)+" [7] GridSearchCV + LogisticRegression ============================",)    
    a1= cv.score(X_train_sel_log, y_train_log)
    a2 =cv.score(X_test_sel_log, y_test_log)
    print("Train accuracy:", a1)  
    print("Test accuracy:",  a2 ) 
    
    y_pred_log = cv.predict(X_test_sel_log)
    print(classification_report(y_test_log, y_pred_log))
    
    # print parameters of the best model
    print(cv.best_params_)
    dict_data = {key1 : attr, key2 : 7, key3: str("GridSearchCV + LogisticRegression( log trans)"), key4 : a1, key5 : a2 ,key6 : '' , key7 :' ' , key8  : cv.best_params_}
   #write_dict_to_csv( sv_file, dict_data)
    log_dict_list.append(dict_data)
    write_dict_to_csv( sv_file, log_dict_list)
 
    
       
##########################   
    
    
    
    
######################################  
    
    
if __name__=='__main__':
 
 #    # list columns to be transformed
#    columns_to_transform = ['BILL', 'AGEGRP2' ,'ORGYN', 'REGION', 'AFFL' , 'GENDER' , 
#                            'ORGANICS' ,'TV_REG'   
#    
#    TrainData('BILL')
#    TrainData('REGION')
#    TrainData('AGEGRP2')
#    TrainData('GENDER')
#    TrainData('TV_REG')   
#    TrainData('ORGANICS')   
#    TrainData('CLASS')         
    TrainData('AFFL')       
#    TrainData('NEIGHBORHOOD')       
#    TrainData('ORGYN')