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

  def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(df)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.KNN = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', p = 1) 
    self.KNN_Train(train_x, train_y)
    self.score = self.KNN_Test(test_x, test_y)
    return self.KNN
    

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



class LogisticRegression:

  def __init__(self, data, partition=0.8):

    #Test Train Split
    msk = np.random.rand(len(df)) < 0.8
    train, test = data[msk], data[~msk]
    train_x, train_y = train['Features'], train['Labels']
    test_x, test_y = test['Features'], test['Labels']
    self.LogReg = LogisticRegression()
    self.LogReg_Train(train_x, train_y)

    self.score = self.LogReg_Test(test_x, test_y)
    return self.KNN

  def LogReg_Train(self, X_train, y_train):

      self.LogReg.fit(X_train, np.ravel(y_train,order='C'))
  
  def LogReg_Test(self, X_test, y_test):

    LogReg_predict = self.LogReg.predict(X_test)
    LogReg_score = np.mean(LogReg_predict == np.ravel(y_test,order='C'))
    return round(LogReg_score, 5)

class SupportVectorMachine:

    def SVM():
        """ Classifier with Support Vector Machine"""
        SVM = svm.SVC()
        SVM.fit(X_train, np.ravel(y_train,order='C'))
        SVM_predict = SVM.predict(X_test)
        SVM_score = np.mean(SVM_predict == np.ravel(y_test,order='C'))
        print("SVM Predictions: " + str(SVM_predict))
        print("SVM Score: {:.2f}".format(SVM_score))
        con_mat = confusion_matrix(np.ravel(y_test,order='C'),SVM_predict, [0,1])
        print(con_mat)
        print("")


class DecisionTree:

  def DT():
      """Classifier with Decision Tree"""
      DT = tree.DecisionTreeClassifier()
      DT.fit(X_train, np.ravel(y_train,order='C'))
      DT_predict = DT.predict(X_test)
      DT_score = np.mean(DT_predict == np.ravel(y_test,order='C'))
      print("DT Predictions: " + str(DT_predict))
      print("DT Score: {:.2f}".format(DT_score))
      con_mat = confusion_matrix(np.ravel(y_test,order='C'),DT_predict, [0,1])
      print(con_mat)
      print("")

class LinearPerception:

    def LP():
        """Classifier with LP"""
        # Fit only to the training data
        LP = Perceptron()
        LP.fit(X_train, np.ravel(y_train,order='C'))
        print("Trained!")
        LP_predict = LP.predict(X_test)
        LP_score = np.mean(LP_predict == np.ravel(y_test,order='C'))
        print("LP Predictions: " + str(LP_predict))
        print("LP Score: {:.8f}".format(LP_score))
        con_mat = confusion_matrix(np.ravel(y_test,order='C'),LP_predict, [0,1])
        print(con_mat)                                                              


