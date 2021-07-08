import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split

Features = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Features Edited.csv')))
Outputs = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Outputs Edited.csv')))

print("Features and Outputs gathered")

#CLASSIFIERS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix

#Splits the data into testing and training

X_train, X_test, y_train, y_test = train_test_split(Features, Outputs, random_state = 1)

#DIMENSION REDUCTION WITH PCA

from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

print("Principal Components Generated")

#Classification

def KNN(x):
    """ Classifier with K-Nearest Neighbors"""
    KNN = KNeighborsClassifier(n_neighbors = x, weights = 'uniform', p = 1) #Pick the number of neighbours
    KNN.fit(X_train, np.ravel(y_train,order='C'))
    KNN_predict = KNN.predict(X_test)
    KNN_score = np.mean(KNN_predict == y_test)
    print(confusion_matrix(KNN_predict, y_test).ravel())
    print("KNN Predictions: " + str(KNN_predict))
    print("KNN Score: {:.2f}".format(KNN_score))
    print("")
    
def MLP(Training, Testing):
    """Classifier with MLP"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(Training)
    Training = scaler.transform(Training)
    Testing = scaler.transform(Testing)
    MLP = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=(30,30,30))
    MLP.fit(Training, np.ravel(y_train,order='C'))
    MLP_predict = MLP.predict(Testing)
    MLP_score = np.mean(MLP_predict == y_test)
    print("MLP Predictions: " + str(MLP_predict))
    print("MLP Score: {:.2f}".format(MLP_score))
    print("")

def DT():
    """Classifier with Decision Tree"""
    DT = tree.DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    DT_predict = DT.predict(X_test)
    DT_score = np.mean(DT_predict == y_test)
    print("DT Predictions: " + str(DT_predict))
    print("DT Score: {:.2f}".format(DT_score))
    print("")

def SGDC():
    """ Classifier with Stochastic Gradient Descent """
    SGD = SGDClassifier(loss="hinge", penalty="l2")
    SGD.fit(X_train, np.ravel(y_train,order='C'))
    SGD_predict = SGD.predict(X_test)
    SGD_score = np.mean(SGD_predict == y_test)
    print("SGD Predictions: " + str(SGD_predict))
    print("SGD Score: {:.2f}".format(SGD_score))
    print("")
    
def GPC():
    """ Classifier with Gaussian Process Classification """
    GPC = GaussianProcessClassifier()
    GPC.fit(X_train, np.ravel(y_train,order='C'))
    GPC_predict = GPC.predict(X_test)
    GPC_score = np.mean(GPC_predict == y_test)
    print("GPC Predictions: " + str(GPC_predict))
    print("GPC Score: {:.2f}".format(GPC_score))
    print("")
    
def SVM():
    """ Classifier with Support Vector Machine"""
    SVM = svm.SVC()
    SVM.fit(X_train, np.ravel(y_train,order='C'))
    SVM_predict = SVM.predict(X_test)
    SVM_score = np.mean(SVM_predict == y_test)
    print("SVM Predictions: " + str(SVM_predict))
    print("SVM Score: {:.2f}".format(SVM_score))
    print("")

#Calling the classifiers
#DT()
MLP(X_train, X_test)
#SGDC()
KNN(10)
#SVM()
#@lru_cache(maxsize=None)
