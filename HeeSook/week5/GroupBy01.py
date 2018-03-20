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
grouped = df.groupby(['CLASS','AGEGRP2'])['NGROUP','NEIGHBORHOOD'].count()
b= grouped.plot.bar()
plt.show()


grouped = df.groupby(['CLASS','AGEGRP2'])['GENDER','AGE','AFFL','BILL'].count()
b= grouped.plot.bar()
plt.show()


grouped = df.groupby(['CLASS','AGEGRP2'])['AFFL','BILL'].count()
b= grouped.plot.bar()
plt.show()


grouped = df.groupby(['CLASS','AGEGRP2'])['ORGANICS', 'ORGYN'].count()
b= grouped.plot.bar()
plt.show()

grouped = df.groupby(['CLASS','AGEGRP2'])['ORGANICS'].count()
b= grouped.plot.bar()
plt.show()


grouped = df.groupby(['CLASS','AGEGRP2']).count()
b= grouped.plot.bar()
plt.show()