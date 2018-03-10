# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:17:08 2018

@author: us1cf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#read the veteran dataset 

df = pd.read_csv('datasets/organics.csv')

#show all columns information

#print(df.info())
#print(df['DOB'].describe())
#print(df['DOB'].value_counts())
#Week3 Prac

import matplotlib.pyplot as plt

def makeplot(x, y):
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Scatter ' + x + ' v ' + y)
    plt.show()

    
##makeplot('BILL', 'AFFL')

#makeplot('GENDER', 'AFFL')
#makeplot('AGE', 'ORGANICS')


#makeplot('AGE', 'BILL')

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


temp = df.fillna(0)
temp.head()
sns.pairplot(temp)

#plt.savefig('week2correlation.pdf')
plt.show()

