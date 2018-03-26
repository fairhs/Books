# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:50:33 2018

@author: us1cf
"""

import numpy as np

import pandas as pd
df = pd.read_csv('datasets/organics.csv')

g = sns.countplot(data = df, x = 'GENDER')
plt.show()

print(df['CLASS'].unique())
dg = sns.countplot(data=df, x='CLASS')
plt.show()

grouped = df.groupby(['CLASS','AGEGRP2'])['NGROUP','NEIGHBORHOOD'].count()
b= grouped.plot.bar()
plt.show()

dg = sns.countplot(data=df, x='NGROUP')
plt.show()

dg = sns.countplot(data=df, x='NEIGHBORHOOD')
plt.show()

g = sns.distplot(df['AGE'].dropna())
plt.show()

g = sns.distplot(df['AFFL'].dropna())
plt.show()

g = sns.countplot(data = df, x = 'AFFL')
plt.show()

dg = sns.countplot(data=df, x='TV_REG')
plt.show()

dump_ngroup = df.groupby(['NGROUP'])['NEIGHBORHOOD'].count()

print(dump_ngroup)