import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('datasets/organics.csv')
'''
CUSTID          22223 non-null object
GENDER          19711 non-null object
DOB             22223 non-null float64
EDATE           22223 non-null float64
AGE             20715 non-null float64
AGEGRP1         20715 non-null object
AGEGRP2         20715 non-null object
TV_REG          21758 non-null object
NGROUP          21549 non-null object
NEIGHBORHOOD    21549 non-null object
LCDATE          21942 non-null float64
ORGANICS        22223 non-null float64
BILL            22223 non-null float64
REGION          21758 non-null object
CLASS           22223 non-null object
ORGYN           22223 non-null float64
AFFL            21138 non-null float64
LTIME           21942 non-null float64
'''
#print(df.info())
#print(df['GENDER'].unique())
'''
fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['AGE','CLASS']).count()['NEIGHBORHOOD'].unstack().plot(ax=ax)

fig, ax = plt.subplots(figsize=(15,7))
obj= df.groupby(['AGE','CLASS']).count()['NEIGHBORHOOD'].plot(ax=ax)


fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['NGROUP','CLASS']).count()['NEIGHBORHOOD'].unstack().plot(ax=ax)

fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['ORGYN','CLASS']).count()['ORGANICS'].unstack().plot(ax=ax)


fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['ORGYN','NGROUP','CLASS']).count()['ORGANICS'].unstack().plot(ax=ax)
'''

#grouped = df.groupby(['TV_REG','ORGYN'])['CLASS'].count()
#print(grouped)
#
#plt.scatter(x=grouped.index.get_level_values(0), y=grouped.index.get_level_values(1))
#plt.show()


#print(Cs)
grouped = df.groupby(['CLASS','ORGYN'])['AFFL','BILL'].count()
b= grouped.plot()


grouped = df.groupby(['CLASS','ORGYN'])['AFFL','BILL','GENDER'].count()
b= grouped.plot()
plt.show()


grouped = df.groupby(['CLASS','ORGYN']).count()
b= grouped.plot()
plt.show()

grouped = df.groupby(['AGEGRP1','ORGYN']).count()
b= grouped.plot()
plt.show()
#f, a = plt.subplots(1,3)
#for n in range(len(grouped)):
#    grouped.xs(Cs[n],ORGYNs[n]).plot(kind ='bar')
#grouped.xs(Cs[0],ORGYNs[0]).plot(kind ='bar')
#grouped = df.groupby(['CLASS','TV_REG']).count()

#fig, ax = plt.subplots(figsize=(10,5))
#pos = list(range(len(grouped))) 
#width = 0.25
#dg = sns.countplot(data=df, x=grouped)
#plt.show() 
#ax = sns.countplot(x=grouped, hue=Cs, data=df)
#print(df['CLASS'].unique())
#plt.bar(pos, grouped['Gold'], width, color='blue', label='Gold')


#plt.bar(Cs+width, grouped['Gold'].index.get_level_values(1),width,color='blue', label='Gold')

#plt.bar(Cs, grouped['Silver'],height, bottom=None, color='pink', label='Silver')
#plt.bar(Cs, grouped['Tin'],height, bottom=None, color='green', label='Tin')
#plt.bar(Cs, grouped['Platinum'],height, bottom=None, color='yellow', label='Platinum')
#plt.legend()
#plt.xlabel('Class')
#plt.ylabel('Count')

#groups = df.groupby(['TV_REG','ORGYN']).count()
#print (groups)
'''

fig, ax1 = plt.subplot(2,1,1);
x = linspace(0,10,50);
y1 = sin(2*x);
plot(ax1,x,y1)
title(ax1,'Subplot 1')
ylabel(ax1,'Values from -1 to 1')

ax2 = plyt.subplot(2,1,2);
y2 = rand(50,1);
scatter(ax2,x,y2)
title(ax2,'Subplot 2')
ylabel(ax2,'Values from 0 to 1')
'''

'''
xlabel = 'ORGYN'
ylabel = 'TV_REG'
x = df[xlabel]
y = df[ylabel]
colors = 'red'
area = 15

plt.title( xlabel+' & ' + ylabel)
plt.ylabel(ylabel)
plt.xlabel(xlabel)

#plt.scatter(x, y, s=area, c='red', alpha=0.5)
plt.scatter(x, y, s=area, c=colors)
plt.show()
'''