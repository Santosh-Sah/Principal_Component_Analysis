# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:12:57 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importPrincipalComponentAnalysisDataset(principalComponentAnalysisDatasetFileName):
    
    principalComponentAnalysisDataset = pd.read_csv(principalComponentAnalysisDatasetFileName)
    X = principalComponentAnalysisDataset.iloc[:, 0:13].values
    y = principalComponentAnalysisDataset.iloc[:, 13].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def savePrincipalComponentAnalysisStandardScaler(principalComponentAnalysisStandardScalar):
    
    #Write PrincipalComponentAnalysisStandardScaler in a picke file
    with open("PrincipalComponentAnalysisStandardScaler.pkl",'wb') as PrincipalComponentAnalysisStandardScaler_Pickle:
        pickle.dump(principalComponentAnalysisStandardScalar, PrincipalComponentAnalysisStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save PrincipalComponentAnalysisModel as a pickle file.
"""
def savePrincipalComponentAnalysisModel(principalComponentAnalysisModel):
    
    #Write PrincipalComponentAnalysisModel as a picke file
    with open("PrincipalComponentAnalysisModel.pkl",'wb') as PrincipalComponentAnalysisModel_Pickle:
        pickle.dump(principalComponentAnalysisModel, PrincipalComponentAnalysisModel_Pickle, protocol = 2)

"""
read PrincipalComponentAnalysisStandardScalar from pickel file
"""
def readPrincipalComponentAnalysisStandardScaler():
    
    #load PrincipalComponentAnalysisStandardScaler object
    with open("PrincipalComponentAnalysisStandardScaler.pkl","rb") as PrincipalComponentAnalysisStandardScaler:
        principalComponentAnalysisStandardScalar = pickle.load(PrincipalComponentAnalysisStandardScaler)
    
    return principalComponentAnalysisStandardScalar

"""
read PrincipalComponentAnalysisModel from pickle file
"""
def readPrincipalComponentAnalysisModel():
    
    #load PrincipalComponentAnalysisModel model
    with open("PrincipalComponentAnalysisModel.pkl","rb") as PrincipalComponentAnalysisModel:
        principalComponentAnalysisModel = pickle.load(PrincipalComponentAnalysisModel)
    
    return principalComponentAnalysisModel

"""
read X_train from pickle file
"""
def readPrincipalComponentAnalysisXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readPrincipalComponentAnalysisXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readPrincipalComponentAnalysisYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readPrincipalComponentAnalysisYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def savePrincipalComponentAnalysisYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readPrincipalComponentAnalysisYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred

def saveTrainingAndTestingDatasetPrincipleComponentAnalysis(X_train_PrincipleComponentAnalysis, X_test_PrincipleComponentAnalysis):
    
    #Write X_train_PrincipleComponentAnalysis in a picke file
    with open("X_train_PrincipleComponentAnalysis.pkl",'wb') as X_train_PrincipleComponentAnalysis_Pickle:
        pickle.dump(X_train_PrincipleComponentAnalysis, X_train_PrincipleComponentAnalysis_Pickle, protocol = 2)
    
    #Write X_test_PrincipleComponentAnalysis in a picke file
    with open("X_test_PrincipleComponentAnalysis.pkl",'wb') as X_test_PrincipleComponentAnalysis_Pickle:
        pickle.dump(X_test_PrincipleComponentAnalysis, X_test_PrincipleComponentAnalysis_Pickle, protocol = 2)

"""
read X_train_PCA from pickle file
"""
def readPrincipalComponentAnalysisXTrainPCA():
    
    #load X_train_PCA
    with open("X_train_PrincipleComponentAnalysis.pkl","rb") as X_train_PCA_pickle:
        X_train_PCA = pickle.load(X_train_PCA_pickle)
    
    return X_train_PCA

"""
read X_test_PCA from pickle file
"""
def readPrincipalComponentAnalysisXTestPCA():
    
    #load X_test_PCA
    with open("X_test_PrincipleComponentAnalysis.pkl","rb") as X_test_PCA_pickle:
        X_test_PCA = pickle.load(X_test_PCA_pickle)
    
    return X_test_PCA

def savePCA(pca):
    
    #Write PCA in a picke file
    with open("PCA.pkl",'wb') as PCA_Pickle:
        pickle.dump(pca, PCA_Pickle, protocol = 2)
        
"""
read PCA from pickle file
"""
def readPCA():
    
    #load PCA
    with open("PCA.pkl","rb") as PCA_pickle:
        pca = pickle.load(PCA_pickle)
    
    return pca
