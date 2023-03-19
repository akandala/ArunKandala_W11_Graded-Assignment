#!/usr/bin/python3
# inference.py
# Xavier Vasques 13/04/2021


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
import pandas as pd
from joblib import load

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier



def inference():

    MODEL_PATH_LDA = 'lda.joblib'
    MODEL_PATH_NN = 'nn.joblib'
    MODEL_PATH_GBC = 'gbc.joblib'
        
    # Load, read and normalize training data from test.csv 
    testing = "test.csv"

    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values

    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
    print("Shape of the test data")

    print(X_test.shape)
    print(y_test.shape)
    
    # Data normalization (0,1).
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training.
    
    # Run model LDA score and classification.
    print(MODEL_PATH_LDA)
    clf_lda = load(MODEL_PATH_LDA)
    print("LDA score and classification:")
    prediction_lda = clf_lda.predict(X_test)
    report_lda = classification_report(y_test, prediction_lda)

    print(clf_lda.score(X_test, y_test))
    print('LDA Prediction:', prediction_lda)
    print('LDA Classification Report:', report_lda)

        
    # Run model NN score and classification.
    clf_nn = load(MODEL_PATH_NN)
    print("NN score and classification:")
    prediction_nn = clf_nn.predict(X_test)
    report_nn = classification_report(y_test, prediction_nn)


    print(clf_nn.score(X_test, y_test))
    print('NN Prediction:', prediction_nn)
    print('NN Classification Report:', report_nn)
    
    # Run model GBC score and classification:.
    clf_gbc = load(MODEL_PATH_GBC)
    print("GBC score and classification:")
    prediction_gbc = clf_gbc.predict(X_test)
    report_gbc = classification_report(y_test, prediction_gbc)

    print(clf_gbc.score(X_test, y_test))
    print('GBC Prediction:', prediction_gbc)
    print('GBC Classification Report:', report_gbc)
    
if __name__ == '__main__':
    inference()
