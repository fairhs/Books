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
xlabel = 'AGE'
ylabel = 'ORGANICS'

x = df[xlabel]
y = df[ylabel]
colors = 'red'
area = 15

plt.title( xlabel+' & ' + ylabel)
plt.ylabel(ylabel)
plt.xlabel(xlabel)

plt.bar(x,y)
#plt.scatter(x, y, s=area, c='red', alpha=0.5)
#plt.scatter(x, y, s=area, c=colors)

plt.show()
#plt.savefig('AGE_ORGANIC.png')
#