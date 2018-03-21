# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:54:46 2018

@author: us1cf
"""

import pydot
from io import StringIO
from sklearn.tree import export_graphviz

from sklearn.model_selection import GridSearchCV

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

#from dm_tools import visualize_decision_tree, analyse_feature_importance 

'''
CUSTID          22223 non-null int64
GENDER          19711 non-null object
DOB             22223 non-null object
EDATE           22223 non-null object
AGE             20715 non-null float64
AGEGRP1         20715 non-null object
AGEGRP2         20715 non-null object
TV_REG          21758 non-null object
NGROUP          21549 non-null object
NEIGHBORHOOD    21549 non-null float64
LCDATE          21942 non-null object
ORGANICS        22223 non-null int64
BILL            22223 non-null float64
REGION          21758 non-null object
CLASS           22223 non-null object
ORGYN           22223 non-null int64
AFFL            21138 non-null float64
LTIME           21942 non-null float64
'''
def preprocess_data(df):
    # Q1.4 and Q6.2
       
    
    # Q1.1
    #cols_miss_drop =['CUSTID', 'TV_REG','NEIGHBORHOOD', 'LCDATE','LTIME']
    #mask = pd.isnull(df['Distance'])

    #for col in cols_miss_drop:
    #    mask = mask | pd.isnull(df[col])

  #  df = df[~mask]
  # Q1.2
    df['AGE'].fillna(round(df['AGE'].mean()), inplace=True)
    df['ORGANICS'].fillna(round(df['ORGANICS'].mean()), inplace=True)
    #df['ORGANICS'].fillna(df['ORGANICS'].mean(), inplace=True)
    #df['ORGANICS'].fillna(df['ORGANICS'].mean(), inplace=True)
    df['REGION']=df['REGION'].replace([' ', 'Midlands', 'North' ,'South East' ,'Scottish' ,'South West'], [0,1,2,3,4,5]).astype(float)
    df['REGION'].fillna(round(df['REGION'].mean()), inplace=True) 
    
    
    df['TV_REG']=df['TV_REG'].replace([' ','Wales & West', 'Midlands', 'N West', 'East', 'N East', 'London' ,'S & S East' ,'C Scotland', 'Ulster' ,'S West', 'Yorkshire' ,'Border', 'N Scot'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13]).astype(float)
    df['TV_REG'].fillna(round(df['TV_REG'].mean()), inplace=True) 

    #['Wales & West', 'Midlands', 'N West', 'East', 'N East', London' ,'S & S East' ,'C Scotland', 'Ulster' ,'S West', 'Yorkshire' ,'Border', 'N Scot']
    #'Midlands' 'North' nan 'South East' 'Scottish' 'South West']
    #'REGION',
    #df['AGEGRP1'] = pd.isnull(df['AGEGRP2'])    
    df['AGEGRP2'] = df['AGEGRP2'].replace([' ','70-80', '40-50', '60-70','50-60', '30-40', '10-20', '20-30'], [0,1,2,3,4,5,6,7]).astype(float)
    df['AGEGRP2'].fillna(round(df['AGEGRP2'].mean()), inplace=True) 

    #['U' 'F' 'M' nan]    
    #df['REGION'].replace([' ', 'U' 'F' 'M'], [0,1,2,3]).astype(float)
    df['GENDER'] = df['GENDER'].replace([' ', 'U', 'F', 'M'], [0,1,2,3]).astype(float)
    df['GENDER'].fillna(round(df['GENDER'].mean()), inplace=True)
    
    #['Gold' 'Silver' 'Tin' 'Platinum']    
    df['CLASS'] =df['CLASS'].replace(['Gold', 'Silver', 'Tin' ,'Platinum'], [3,2,1,4]).astype(float)

    #[0 1 2 3]
    df['ORGANICS'] =df['ORGANICS'].astype(float)
    #[0 1 ]
    df['ORGYN'] =df['ORGYN'].astype(float)
    
    
    #AFFL
    df['AFFL'].fillna(round(df['AFFL'].mean(),2), inplace=True)
    
    #df['Longtitude_nan'] = pd.isnull(df['Longtitude'])
    #df['Lattitude'].fillna(0, inplace=True)
    #df['Longtitude'].fillna(0, inplace=True)
    
    # Q6.1. Change date into weeks and months
    #df['Sales_week'] = pd.to_datetime(df['Date']).dt.week
    #df['Sales_month'] = pd.to_datetime(df['Date']).dt.month
    #df = df.drop(['AGE'], axis=1)  # drop the date, not required anymore
    
    # Q4
    
    df.drop(['NEIGHBORHOOD','DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' ,'AGEGRP1'], axis=1, inplace=True)
    #df = df.drop(['NEIGHBORHOOD', 'LCDATE','LTIME'], axis=1, inplace=True)
    df = pd.get_dummies(df)
    
    return df

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_

    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
       print(feature_names[i], ':', importances[i])


def visualize_decision_tree(dm_model, feature_names, save_name):
#    import pydot
#    from io import StringIO
#    from sklearn.tree import export_graphviz
    
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue()).write_png(save_name)
    #pydotplus
    #graph.create_png(save_name)
  #  graph.save_png(save_name)
    #graph.draw(save_name) # saved in the following file
    #pydot.Graph.
   
    #graph = pydot.graph_from_dot_data(dotfile.getvalue())
    #from IPython.display import Image 
    #Image(graph.create_png())  
    
    
    #export_graph(save_name)

def checkResult(attr):
    # target/input split
    y = Loaded_df[attr]
    X = Loaded_df.drop([attr], axis=1)
    
    # setting random state
    rs = 0
    
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)
    
    
    # simple decision tree training
    model = DecisionTreeClassifier(random_state=rs)
    model.fit(X_train, y_train)
    
    print(attr)
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))
    analyse_feature_importance(model, X.columns, 20)
    
    
    # grid search CV
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(2, 7),
              'min_samples_leaf': range(200, 600, 100)}
 

    cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    
    print("Train accuracy:", cv.score(X_train, y_train))
    print("Test accuracy:", cv.score(X_test, y_test))
    
    # test the best model
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # print parameters of the best model
    print(cv.best_params_)  
#    y_pred = model.predict(X_test)
#    print(classification_report(y_test, y_pred))
    visualize_decision_tree(cv.best_estimator_, X.columns, 'Tree_'+str(attr)+'.png')
   
    
if __name__=='__main__':
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    #print(Loaded_df.info())
    prepared_df=preprocess_data(Loaded_df)
    #print(prepared_df.info())


    checkResult('CLASS')
    checkResult('GENDER')
    checkResult('AGEGRP2')
    checkResult('REGION')
