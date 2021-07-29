#Generic Code for basic model training

#Importing the necessary modules
import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split

#Importing the Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

#Data Handling (Spliting Data in training and testing)


#Outputs X_train, X_test, y_train, y_test
class KNN:

  def __init__(self, data, partition=0.8, k=3):
    #Test Train Split
    msk = np.random.rand(len(data)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.KNN = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', p = 1) 
    self.KNN_Train(train_x, train_y)
    self.score = self.KNN_Test(test_x, test_y)
    
  
  def KNN_Train(self, X_train, y_train, k=3):
    """
    Trains a KNN model using inputted data
    @param: X_train The images that will be used to train the model
    @param: y_train The classification labels of the training images
    @param: k The number of neighbors the model will use 
    """
    self.KNN.fit(X_train, np.ravel(y_train,order='C'))
  

  def KNN_Test(self, X_test, y_test):
    """
    Trains a KNN model using inputted data

    @param: X_test The images that will be used to test the model
    @param: y_test The classification labels of the testing images
    @return: score The accuracy of the model

    """
    KNN_predict = self.KNN.predict(X_test)
    KNN_score = np.mean(KNN_predict == np.ravel(y_test,order='C'))
    # return ("KNN Score: {:.9f}".format(KNN_score))
    return round(KNN_score, 5)
    # Update score in case of retraining

class LogisticReg:

  def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(data)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.LogReg = LogisticRegression()
    self.LogReg_Train(train_x, train_y)

    self.score = self.LogReg_Test(test_x, test_y)

  def LogReg_Train(self, X_train, y_train):

      self.LogReg.fit(X_train, np.ravel(y_train,order='C'))
  
  def LogReg_Test(self, X_test, y_test):

    LogReg_predict = self.LogReg.predict(X_test)
    LogReg_score = np.mean(LogReg_predict == np.ravel(y_test,order='C'))
    return round(LogReg_score, 5)

class SupportVectorMachine:
  def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(data)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.SVM = svm.SVC()
    self.SVM_Train(train_x, train_y)

    self.score = self.SVM_Test(test_x, test_y)
  
  def SVM_Train(self, X_train, y_train):

    self.SVM.fit(X_train, np.ravel(y_train,order='C'))
  
  def SVM_Test(self, X_test, y_test):

    SVM_predict = self.SVM.predict(X_test)
    self.SVM_score = np.mean(SVM_predict == np.ravel(y_test,order='C'))
    return round(self.SVM_score, 5)

class DecisionTree:
  def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(data)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.DT = tree.DecisionTreeClassifier()
    self.DT_Train(train_x, train_y)

    self.score = self.DT_Test(test_x, test_y)

  def DT_Train(self, X_train, y_train):

      self.DT.fit(X_train, np.ravel(y_train,order='C'))

  def DT_Test(self, X_test, y_test):

    DT_predict = self.DT.predict(X_test)
    self.DT_score = np.mean(DT_predict == np.ravel(y_test,order='C'))
    return round(self.DT_score, 5)

class LinearPerception:
    def __init__(self, data, partition=0.8):

      #Test Train Split
      msk = np.random.rand(len(data)) < 0.8
      train, test = data[msk], data[~msk]
      train_x, train_y = train['Features'], train['Labels']
      test_x, test_y = test['Features'], test['Labels']
      self.LP = Perceptron()
      self.LP_Train(train_x, train_y)

      self.score = self.LP_Test(test_x, test_y)
    
    def LP_Train(self, X_train, y_train):

      self.LP.fit(X_train, np.ravel(y_train,order='C'))

    def LP_Test(self, X_test, y_test):
      LP_predict = self.LP.predict(X_test)
      self.LP_score = np.mean(LP_predict == np.ravel(y_test,order='C'))
      return round(self.LP_score, 5)                                                              
"""
ALL THE CLASSES HAVE THE SAME CODE (MAYBE MAKE A GENERAL CLASS FOR THEM)
def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(data)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.ModelName = LogisticRegression()
    self.ModelName_Train(train_x, train_y)

    self.score = self.ModelName_Test(test_x, test_y)

  def ModelName_Train(self, X_train, y_train):

      self.ModelName.fit(X_train, np.ravel(y_train,order='C'))
  
  def ModelName_Test(self, X_test, y_test):

    ModelName_predict = self.ModelName.predict(X_test)
    ModelName_score = np.mean(ModelName_predict == np.ravel(y_test,order='C'))
    return round(ModelName_score, 5)

"""