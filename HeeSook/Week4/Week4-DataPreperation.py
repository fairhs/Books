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

print()
#show all columns information


from datetime import date

def calculate_age(born):
    today = date.today()
    return today.year - int(y) - ((today.month, today.day) < (int(m), int(d))
                    
                            
bod  =pd.to_datetime(df['DOB'][1])
                            
calculate_age(bod)
today = date.today()
age = today - pd.to_datetime(df['DOB'])
print (age)

#print('age')
#print(df.info())
'''
print(df['DOB'].describe())
#print(df['DOB'].value_counts())




#dg = sns.distplot(df['DemAge'].dropna())
#dg = sns.countplot(data=df , x='DemGender')
plt.show()
'''
#Week3 Prac

import matplotlib.pyplot as plt
# df.info() # See all field in the data

#plt.xlabel('GENDER')
#plt.ylabel('NGROUP')
#plt.title('Scatter GENDER v NGROUP')
#plt.show()

def makeplot(x, y):
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Scatter ' + x + ' v ' + y)
    plt.show()
    
makeplot('CUSTID', 'AFFL')

# print(df.groupby(['CUSTID'])['NEIGHBORHOOD'].mean())

#
#dg = sns.distplot(df['NEIGHBORHOOD'].dropna())
#plt.show()
    
#makeplot('ORGANICS', 'ORGYN')
#dg = sns.countplot(data=df, x='GENDER')
#plt.show()

#flagOrgnics = {'U':0, 'H': 1}
#df['ORGYN'] = df['ORGYN'].map(flagOrgnics)

#
#import plotly.plotly as py
#import plotly.graph_objs as go
#
#import numpy as np
#
#N = 500
#
#
#custid=df['CUSTID']
#norganic=df['ORGANICS']
#borganic=df['ORGYN']
#
#print(df['CUSTID'].value)
#trace0 = go.Scatter(
#    x = np.random.randn(N),
#    y = np.random.randn(N)-2,
#    name = 'Above',
#    mode = 'markers',
#    marker = dict(
#        size = 10,
#        color = 'rgba(152, 0, 0, .8)',
#        line = dict(
#            width = 2,
#            color = 'rgb(0, 0, 0)'
#        )
#    )
#)
#
#trace1 = go.Scatter(
#    x= np.random.randn(N),
#    y = np.random.randn(N)-2,
#    name = 'Below',
#    mode = 'markers',
#    marker = dict(
#        size = 10,
#        color = 'rgba(255, 182, 193, .9)',
#        line = dict(
#         width = 2,
#        )
#    )
#)
#
#data = [trace0, trace1]
#
#layout = dict(title = 'Styled Scatter',
#              yaxis = dict(zeroline = False),
#              xaxis = dict(zeroline = False)
#             )
#
#fig = dict(data=data, layout=layout)
#py.iplot(fig, filename='styled-scatter')

#
#custid=df['CUSTID']
#numberororganic=df['ORGANICS']
#organic=df['ORGYN']
#
#plt.plot(custid,numberororganic, color='g')
#plt.plot(custid, organic, color='orange')
#plt.xlabel('Number of organic products purchased')
#plt.ylabel('Organic purchased')
#plt.title('organics Vs orgyn')
#plt.show()


# makeplot('AGE', 'ORGANICS')


#makeplot('AGE', 'BILL')

'''
temp = df.fillna(0)
temp.head()
sns.pairplot(temp)
plt.show()
'''

#print(df['GENDER'].describe())
#print(df['GENDER'].str.strip(['b'',''']).describe())

