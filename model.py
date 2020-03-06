import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def confusion_matrix(Y_pred,Y_test,threshold):
    """
    Function to create a confusion matrix based on the values in the predicted outcome/score and the test set divided on
    the threshold.

    Args: a numpy.ndarray with predicted values
          a numpy.ndarray with actual values
          an integer
    
    Returns: the confusion matrix which is a numpy.ndarray itself
    """
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



def main_imp(fname, lst_of_args):
    """
    Function to preprocess the data and get the target feature. It further creates a Linear Regression model and trains it. 
    It is used to get the false positve rates and the the true positive rates for a particular confusion matrix and return it
    as a dataframe so that the points can be plotted.

    Args: a string with the name of the file
          a string with the name of the target feature
    
    Returns: a pandas.DataFrame
    """
    target_feature = lst_of_args[0]
    t_up_file = lst_of_args[1]

    if t_up_file == 'csv':
        dataset = pd.read_csv(fname, sep = ',')
    elif t_up_file == 'excel':
        dataset = pd.read_excel(fname, headers = None)
    else:
        return []

    pred_var = target_feature 
    Y = dataset[pred_var]
    dataset = dataset.drop([pred_var], axis = 1)
    dataset = dataset.fillna(dataset.mean())
    X = dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    regressor = LinearRegression().fit(X_train,Y_train)
    Y_pred = regressor.predict(X_test)

    tpr = []
    fpr = []
    temp_conf_mtr = []
    lst = np.linspace(0,1,1000)
    Y_test = Y_test.to_numpy()
    for i in lst:
        matrix = confusion_matrix(Y_pred,Y_test,i)
        tpr.append(matrix[0][0]/(matrix[0][0]+matrix[1][0]))
        fpr.append(1 - (matrix[1][1]/(matrix[0][1]+matrix[1][1])))
        temp_conf_mtr.append(matrix)
    
    dictionary = {'fpr':fpr, 'tpr':tpr, 'confusionMTR':temp_conf_mtr, 'threshold':lst}
    df_1 = pd.DataFrame(dictionary)
    return df_1