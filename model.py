import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 

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
            if 1.*df[i].nunique()/df[i].count() > 0.25: #i might need to change the threshold
                lst.append(i)

    return lst


def fill_nan_vals (df):
    for i in df.columns:
        df[i] = df[i].replace('',np.nan, inplace = True)

    for i in df.columns:
        df[i] = df[i].replace(np.inf,np.nan, inplace = True)
    
    return df.fillna(0)

def plot_roc_curve(fpr,tpr):
    print("the false positive rate array is")
    print(fpr)
    print("the true positive rate array is")
    print(tpr)
    plt.plot(fpr, tpr, color='orange', label = 'AUC = %0.2f' % 0.5)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def confusion_matrix(Y_pred,Y_test,threshold):
    #i am assuming its 0 and 1 what is the binary values are 1 or 2. Probably using np.unique
    conf_arr = [[0,0],[0,0]]
    for i in range(len(Y_test)):
        if int(Y_test[i]) == 1:
            if float(Y_pred[i])<threshold:
                conf_arr[1][0] = conf_arr[1][0]+1
            else:
                conf_arr[0][0] = conf_arr[0][0] + 1
        elif int(Y_test[i]) == 0:
            if float(Y_pred[i]) < threshold:
                conf_arr[1][1] = conf_arr[1][1] +1
            else:
                conf_arr[0][1] = conf_arr[0][1] +1
    return conf_arr



def main():
    #use the filename when uploading the csv
    dataset = pd.read_csv('train.csv')
    pred_var = "Survived" #this needs to have its own inputs
    Y = dataset[pred_var] #target variable
    dataset = dataset.drop([pred_var], axis = 1)
    drop_lst = categorical_to_continuous(dataset)
    dataset = dataset.drop(drop_lst, axis = 1)
    dataset = fill_nan_vals(dataset)
    X = dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LinearRegression().fit(X_train,Y_train)
    Y_pred = regressor.predict(X_test)

    tpr = []
    fpr = []
    lst = np.linspace(0,1,10)
    Y_test = Y_test.to_numpy()
    for i in lst:
        matrix = confusion_matrix(Y_pred,Y_test,i)
        tpr.append(matrix[0][0]/(matrix[0][0]+matrix[1][0]))
        fpr.append(1 - (matrix[1][1]/(matrix[0][1]+matrix[1][1])))

    plot_roc_curve(fpr,tpr)
    pickle.dump(regressor,open(".../model.pkl","wb"))