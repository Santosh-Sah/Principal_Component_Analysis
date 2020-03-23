# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:19:02 2020

@author: Santosh Sah
"""

import pandas as pd
from PrincipalComponentAnalysisUtils import readPrincipalComponentAnalysisModel, readPrincipalComponentAnalysisStandardScaler,readPCA

def predict():
    
    principalComponentAnalysis = readPrincipalComponentAnalysisModel()
    principalComponentAnalysisStandardScaler = readPrincipalComponentAnalysisStandardScaler()
    pca = readPCA()

    inputValue = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]]
    inputValueDataframe = pd.DataFrame(pca.transform(principalComponentAnalysisStandardScaler.transform(inputValue)))
    
    predictedValue = principalComponentAnalysis.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()