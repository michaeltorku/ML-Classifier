import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split

Features = pd.read_csv(r'\Users\amoah_k\Desktop\Extended Essay\Extended Essay Dataset\Dataset/Combined Hehe.csv')
Features = Features.transpose()
Labels = pd.read_csv(r'\Users\amoah_k\Desktop\Extended Essay\Extended Essay Dataset\Dataset/Combined Labels.csv')
#Labels = Labels.transpose()

from sklearn.decomposition import PCA


X_train, X_test, y_train, y_test = train_test_split(Features, Labels, random_state = 3) #3 #373 #217 #754 #11 #5235

pca = PCA(.95) #Used 95% instead

principalcomponents = pca.fit_transform(X_train)
X_train = principalcomponents
X_test = pca.transform(X_test)


print("Transformed")

principalDF = pd.DataFrame(data = principalcomponents)
print(pca.n_components_)


#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

def KNN(x):
    """ Classifier with K-Nearest Neighbors"""
    KNN = KNeighborsClassifier(n_neighbors = x, weights = 'uniform', p = 1) #Pick the number of neighbours
    KNN.fit(X_train, np.ravel(y_train,order='C'))
    KNN_predict = KNN.predict(X_test)
    KNN_score = np.mean(KNN_predict == np.ravel(y_test,order='C'))
    #print("KNN Predictions: " + str(KNN_predict))
    print("KNN Score: {:.9f}".format(KNN_score))
    #print("")
    con_mat = confusion_matrix(np.ravel(y_test,order='C'),KNN_predict, [0,1])
    print(con_mat)

def LogReg():
    logreg = LogisticRegression()
    logreg.fit(X_train, np.ravel(y_train,order='C'))
    logreg_predict = logreg.predict(X_test)
    logreg_score = np.mean(logreg_predict == np.ravel(y_test,order='C'))
    print("LogReg Score: {:.9f}".format(logreg_score))
    con_mat = confusion_matrix(np.ravel(y_test,order='C'),logreg_predict, [0,1])
    print(con_mat)

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

LP()
#MLP(X_train, X_test, 'tanh')
# MLP craxy results (solver= Func, random_state=0, hidden_layer_sizes=(400,400, 10))
KNN(146)
LogReg()
SVM()
DT()

#X = 0
#Accuracy = []
#Neighbors = []
#while X != 700:
 #   X = X + 20
  #  Accuracy.append(KNN(X))
   # Neighbors.append(X)
    
#Resultsdf = pd.DataFrame(data = Accuracy)
#Neighborsdf = pd.DataFrame(data = Neighbors)
#Resultsdf = Resultsdf.append(Neighborsdf, ignore_index = True)
#Resultsdf.to_csv(r'\Users\amoah_k\Desktop\Extended Essay\Extended Essay Dataset/OptimizingK.csv')

