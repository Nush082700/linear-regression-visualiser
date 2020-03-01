import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# from sklearn.cross_validation import train_test_split


#please write comments
def categorical_to_continuous(df):
    lst = []
    for i in df.columns:
        if df[i].dtype == "object":
            if len(df[i].unique())>5:
                lst.append(i)
            else:
                df[i] = df[i].astype("category")
                df[i] = df[i].cat.codes
        else:
            if 1.*df[i].nunique()/df[i].count() > 0.50: #i might need to change the threshold
                lst.append(i)

    return lst


#make this function more efficient
def fill_nan_vals (df):
    for i in df.columns:
        df[i] = df[i].replace('',np.nan, inplace = True)

    for i in df.columns:
        df[i] = df[i].replace(np.inf,np.nan, inplace = True)
    
    # for i in df.columns:
    #     df[i] = df[i].fillna(0, inplace = True)

    return df.fillna(0)

dataset = pd.read_csv('train.csv')
pred_var = "Survived" #this needs to have its own inputs

Y = dataset[pred_var] #target variable

dataset = dataset.drop([pred_var], axis = 1)

drop_lst = categorical_to_continuous(dataset)

dataset = dataset.drop(drop_lst, axis = 1)
# dataset = dataset.repalce('', np.nan, inplace = True)
dataset = fill_nan_vals(dataset)

X = dataset

regressor = LinearRegression().fit(X,Y)

print (regressor.score(X,Y))

# pickle.dump(regressor, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl','rb'))
