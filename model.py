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
    
    # for i in df.columns:
    #     df[i] = df[i].fillna(0, inplace = True)

    return df.fillna(0)

dataset = pd.read_csv('train.csv')
pred_var = "Survived" #this needs to have its own inputs

Y = dataset[pred_var] #target variable

dataset = dataset.drop([pred_var], axis = 1)

# drop_lst = categorical_to_continuous(dataset)

# dataset = dataset.drop(drop_lst, axis = 1)
# dataset = dataset.repalce('', np.nan, inplace = True)
dataset = fill_nan_vals(dataset)

X = dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression().fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)

def plot_roc_curve(fpr,tpr):
    print("the false positive rate array is")
    print(fpr)
    print("the true positive rate array is")
    print(tpr)
    plt.plot(fpr, tpr, color='orange', label = 'AUC = %0.2f' % 0.5)
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def categorize(arr,threshold):
#     Y_test = Y_test.to_numpy()
#     Y_test = np.where(Y_test>=threshold, 1, 0)
    return np.where(arr>=threshold, 1, 0)

def pandas_confusion_matrix(Y_pred, Y_test):
    df = Y_test.to_frame(name = 'Y_test')
    df['Y_pred'] = Y_pred
    print(df)
    confusion_matrix = pd.crosstab(df['Y_test'], df['Y_Pred'], rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
#     tpr = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
#     fpr = 1 - (confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1]))
    return confusion_matrix

tpr = []
fpr = []
lst = np.linspace(0,1,10)
for i in lst:
    Y_pred = categorize(Y_pred,i)
    mtr = pandas_confusion_matrix(Y_pred,Y_test)
    tpr.append(confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1]))
    fpr.append(1 - (confusion_matrix[1][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])))

plot_roc_curve(fpr,tpr)

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred)
print("the false positive rate array is")
print(fpr)
print("the true positive rate array is")
print(tpr)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

