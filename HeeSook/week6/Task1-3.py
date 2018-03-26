# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:50:33 2018

@author: us1cf
"""

import numpy as np

import pandas as pd
import plotly.plotly as py
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
    

    
    dfdrop = df.drop(['DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' , 'AGEGRP1'], axis=1, inplace=True)
    print("Drop list===================================")
    print ("'DOB', 'LCDATE', 'LTIME', 'EDATE',  'NGROUP' , 'AGEGRP1'")
    print("============================================")   
    df = pd.get_dummies(df)
    
    return df


if __name__=='__main__':
    
    
    Loaded_df = pd.read_csv('datasets/organics.csv',index_col=0)
    print("Original DataSet=============================")
    print(Loaded_df.info())
       
    # preprocessing step
    df = preprocess_data(Loaded_df) 
    
    print("Prepared Data Set===========================")
    print(Loaded_df.info())
    

    
    