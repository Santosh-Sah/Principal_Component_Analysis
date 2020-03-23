# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:15:08 2020

@author: Santosh Sah
"""
from sklearn.preprocessing import StandardScaler
from PrincipalComponentAnalysisUtils import (importPrincipalComponentAnalysisDataset, saveTrainingAndTestingDataset, savePrincipalComponentAnalysisStandardScaler)

def preprocess():
    
    X_train, X_test, y_train, y_test = importPrincipalComponentAnalysisDataset("Principal_Component_Analysis_Wines.csv")
    
    principalComponentAnalysisStandardScalar = StandardScaler()
    
    principalComponentAnalysisStandardScalar.fit(X_train)
    savePrincipalComponentAnalysisStandardScaler(principalComponentAnalysisStandardScalar)
    
    X_train = principalComponentAnalysisStandardScalar.transform(X_train)
    X_test = principalComponentAnalysisStandardScalar.transform(X_test)
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()