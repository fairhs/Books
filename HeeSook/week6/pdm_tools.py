# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:24:28 2018

@author: us1cf
"""

import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    print(df['BILL'] .unique())

    #AFFL
    df['AFFL'].fillna(round(df['AFFL'].mean(),2), inplace=True)
    df['AFFL'] = round((df['AFFL']) / 10)+1   
    

    
    df.drop(['DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' , 'AGEGRP1'], axis=1, inplace=True)
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


#import csv
# Test3-3
key1 = 'attr'  # attributes
key2 = 'no'     # Number of training
key3 = 'des' # description of training 
key4 = 'TrainAcc' # Training accuracy
key5 = 'TestAcc'# Test accuracy
key6 ='originf'  # orignial feature set  
key7 = 'afterEliminate' #'after elimination 
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


    
