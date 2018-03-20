import numpy as np
import pandas as pd

def preprocess_data(df):
    # Q1.4 and Q6.2
       
    df = df.drop(['TV_REG','NEIGHBORHOOD', 'LCDATE','LTIME'], axis=1)
    
    # Q1.1
    #cols_miss_drop =['CUSTID', 'TV_REG','NEIGHBORHOOD', 'LCDATE','LTIME']
    #mask = pd.isnull(df['Distance'])

    #for col in cols_miss_drop:
    #    mask = mask | pd.isnull(df[col])

  #  df = df[~mask]
    
    # Q1.2
    df['AGE'].fillna(df['AGE'].mean(), inplace=True)
    df['ORGANICS'].fillna(df['ORGANICS'].mean(), inplace=True)
    
    df['AGEGRP1'] = pd.isnull(df['AGEGRP2'])    
    df['AGEGRP2'] = pd.isnull(df['AGEGRP1'])
    #df['Longtitude_nan'] = pd.isnull(df['Longtitude'])
    #df['Lattitude'].fillna(0, inplace=True)
    #df['Longtitude'].fillna(0, inplace=True)
    
    # Q6.1. Change date into weeks and months
    #df['Sales_week'] = pd.to_datetime(df['Date']).dt.week
    #df['Sales_month'] = pd.to_datetime(df['Date']).dt.month
    #df = df.drop(['AGE'], axis=1)  # drop the date, not required anymore
    
    # Q4
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
    import pydot
    from io import StringIO
    from sklearn.tree import export_graphviz
    
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file
