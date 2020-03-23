# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:10:10 2020

@author: Santosh Sah
"""

from PrincipalComponentAnalysisUtils import (readPrincipalComponentAnalysisXTestPCA, readPrincipalComponentAnalysisModel,
                                     savePrincipalComponentAnalysisYPred)

"""
test the model on testing dataset
"""
def testLogisticRegressionModel():
    
    X_test = readPrincipalComponentAnalysisXTestPCA()
    
    principalComponentAnalysisModel = readPrincipalComponentAnalysisModel()
    
    y_pred = principalComponentAnalysisModel.predict(X_test)
    savePrincipalComponentAnalysisYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testLogisticRegressionModel()