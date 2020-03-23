# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:19:37 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from PrincipalComponentAnalysisUtils import (readPrincipalComponentAnalysisYTest, readPrincipalComponentAnalysisYPred)

"""

calculating PrincipalComponentAnalysis confussion matrix

"""
def testPrincipalComponentAnalysisConfussionMatrix():
    
    y_test = readPrincipalComponentAnalysisYTest()
    y_pred = readPrincipalComponentAnalysisYPred()
    
    principalComponentAnalysisConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(principalComponentAnalysisConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[14  0  0]
    [ 1 15  0]
    [ 0  0  6]]
    
    """
"""
calculating accuracy score

"""

def testPrincipalComponentAnalysisAccuracy():
    
    y_test = readPrincipalComponentAnalysisYTest()
    y_pred = readPrincipalComponentAnalysisYPred()
    
    principalComponentAnalysisConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(principalComponentAnalysisConfussionAccuracy) #.972%

"""
calculating classification report

"""

def testPrincipalComponentAnalysisClassificationReport():
    
    y_test = readPrincipalComponentAnalysisYTest()
    y_pred = readPrincipalComponentAnalysisYPred()
    
    principalComponentAnalysisConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(principalComponentAnalysisConfussionClassificationReport)
    
    """
              precision    recall  f1-score   support

          1       0.93      1.00      0.97        14
          2       1.00      0.94      0.97        16
          3       1.00      1.00      1.00         6

avg / total       0.97      0.97      0.97        36

    """
    
if __name__ == "__main__":
    #testPrincipalComponentAnalysisConfussionMatrix()
    #testPrincipalComponentAnalysisAccuracy()
    testPrincipalComponentAnalysisClassificationReport()