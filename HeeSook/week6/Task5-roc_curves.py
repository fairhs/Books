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
# for Task 5-3
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#from pdm_tools import preprocess_data, plot_skewed_columns, write_dict_to_csv

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

task = "Task5_ROC_CURVES"

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
  


def CompareingModel(attr, max_iter_opt) :  
    log_dict_list =[]   
    max_iteration = max_iter_opt
    print(max_iteration)

           
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
       
    # preprocessing step
    df = preprocess_data(Loaded_df)
    
    # random state
    rs = 10
    dt_model = None
    log_reg_model=None
    nn_model=None
    X_train = None
    X_test= None
    y_train = None
    y_test = None
    X_mat =None;
###################################################
    # target/input split
    y = df[attr]
    X = df.drop([attr], axis=1)

    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)
  

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
    
    #max_iteration = 500
    cv = GridSearchCV(param_grid=params_nn, estimator=MLPClassifier(max_iter=max_iteration, random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    nn_model = cv.best_estimator_
    print(nn_model)
    
#######################################################################################
    y_pred_dt = dt_model.predict(X_test)
    y_pred_log_reg = log_reg_model.predict(X_test)
    y_pred_nn = nn_model.predict(X_test)
    
    a1 =accuracy_score(y_test, y_pred_dt)
    a2 = accuracy_score(y_test, y_pred_log_reg)
    a3 = accuracy_score(y_test, y_pred_nn)
    print("Accuracy score on test for DT:", a1)
    print("Accuracy score on test for logistic regression:",a2 )
    print("Accuracy score on test for NN:", a3)
 
    
########################################################################################
    # typical prediction
    y_pred = dt_model.predict(X_test)
    
    # probability prediction from decision tree
    y_pred_proba_dt=None
    y_pred_proba_log_reg = None
    y_pred_proba_nn = None
    y_pred_proba_dt = dt_model.predict_proba(X_test)
    
    print("Probability produced by decision tree for each class vs actual prediction on TargetB (0 = non-donor, 1 = donor). You should be able to see the default threshold of 0.5.")
    print("(Probs on zero)\t(probs on one)\t(prediction made)")
    # print top 10
    for i in range(20):
        print(y_pred_proba_dt[i][0], '\t', y_pred_proba_dt[i][1], '\t', y_pred[i])   
 
    
##############################################################
    y_pred_proba_dt = dt_model.predict_proba(X_test)
    y_pred_proba_log_reg = log_reg_model.predict_proba(X_test)
    y_pred_proba_nn = nn_model.predict_proba(X_test)
    roc_index_dt=None
    roc_index_log_reg = None
    roc_index_nn=  None
    print(type(y_pred_proba_dt))
    print(type(roc_index_dt))
    roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])
    roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])
    roc_index_nn = roc_auc_score(y_test, y_pred_proba_nn[:, 1])
    
    a4=0
    a5=0
    a6=0
    a4=roc_index_dt
    a5=roc_index_log_reg
    a6= roc_index_nn

    print("ROC index on test for DT:", a4)
    print("ROC index on test for logistic regression:", a5)
    print("ROC index on test for NN:", a6)    
#    
##############################################################
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])
    fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])
    fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_proba_nn[:,1])
##########################
    
    plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
    plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
    plt.plot(fpr_nn, tpr_nn, label='ROC Curve for NN {:.3f}'.format(roc_index_nn), color='darkorange', lw=0.5)
    
    # plt.plot(fpr[2], tpr[2], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig (str(task) + "_" + str(attr) +"_" +str(max_iteration)+".png")
    plt.show()

    dict_data = {key1 : attr, key2 :a1 , key3: a2, key4 :a3, key5 : a4, key6 : a5 , key7 :a6 , key8 :max_iteration}
    log_dict_list.append(dict_data)
   # return a1, a2, cv.best_params_
############################################################
    print(sv_file)
    write_dict_to_csv(sv_file, log_dict_list)


#################################################################


##############################################################
#################################################################
#    
if __name__=='__main__':
#AGEGRP2, CLASS, AFFL , ORGANICS, CLASS , AGE, NEIGHBORHOOD, GENDER, TV_REG
#
#  File "C:\prj\anaconda3\lib\site-packages\sklearn\metrics\base.py", line 72, in _average_binary_score
#    raise ValueError("{0} format is not supported".format(y_type))
#
#The minimum number of members in any class cannot be less than n_splits=10.  AGE


#    ValueError: multiclass format is not supported
    maxiter = [50,100]#,200,400,600,800,1000]
    attr_list = ['BILL']
    for j in range(len(attr_list)):            
        for i in range(len(maxiter)) :      
            print(maxiter[i] , attr_list[j])
            CompareingModel(attr_list[j], maxiter[i])      
    
    