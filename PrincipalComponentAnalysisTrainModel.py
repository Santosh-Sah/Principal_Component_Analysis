# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:13:47 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from PrincipalComponentAnalysisUtils import (savePrincipalComponentAnalysisModel, readPrincipalComponentAnalysisXTrain, readPrincipalComponentAnalysisYTrain,
                                             saveTrainingAndTestingDatasetPrincipleComponentAnalysis, readPrincipalComponentAnalysisXTrainPCA,
                                             readPrincipalComponentAnalysisXTest, savePCA)

"""
Train PrincipalComponentAnalysis model 
"""
def trainPrincipalComponentAnalysisModel():
    
    X_train = readPrincipalComponentAnalysisXTrainPCA()
    y_train = readPrincipalComponentAnalysisYTrain()
        
    principalComponentAnalysis = LogisticRegression(random_state = 1234)
    principalComponentAnalysis.fit(X_train, y_train)
    
    savePrincipalComponentAnalysisModel(principalComponentAnalysis)

def findNoOfFeatureComponentForModel():
    
    X_train = readPrincipalComponentAnalysisXTrain()
    X_test = readPrincipalComponentAnalysisXTest()
    
    pca = PCA(n_components = None)
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    principal_component_analysis_explained_variance = pca.explained_variance_ratio_
    print(principal_component_analysis_explained_variance)
    
    """
    Here we can see that two components able to explain more than 56% of the variance. Here we are going to take only 2 components for the model building.
    [0.36884109 0.19318394 0.10752862 0.07421996 0.06245904 0.04909
     0.04117287 0.02495984 0.02308855 0.01864124 0.01731766 0.01252785
     0.00696933]
    """

def selectedFeatureComponentsForModel():
    
    X_train = readPrincipalComponentAnalysisXTrain()
    X_test = readPrincipalComponentAnalysisXTest()
    
    pca = PCA(n_components = 2)
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    savePCA(pca)
    saveTrainingAndTestingDatasetPrincipleComponentAnalysis(X_train, X_test)

if __name__ == "__main__":
    #findNoOfFeatureComponentForModel()
    selectedFeatureComponentsForModel()
    #trainPrincipalComponentAnalysisModel()    
