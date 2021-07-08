import csv
import numpy as np
import pandas as pd
import sklearn
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib

Features = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Features Edited.csv')))
Outputs = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Outputs Edited.csv')))

# SPLITTING DATASET INTO TRAINING:TESTING 75:25
X_train, X_test, y_train, y_test = train_test_split(Features, Outputs, random_state = 0)

#DIMENSION REDUCTION WITH PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

#Classification
 
def KNN(x):
    """ Classifier with K-Nearest Neighbors"""
    KNN = KNeighborsClassifier(n_neighbors = x) #Pick the number of neighbours
    KNN.fit(X_train, np.ravel(y_train,order='C'))
    filename = 'ClimateChuck.sav'
    joblib.dump(KNN, filename)

def Predict():
    ChuckName = 'ClimateChuck.sav'
    ClimateChuck = joblib.load(ChuckName)
    Prediction = ClimateChuck.predict(Data)
    Probability = ClimateChuck.predict_proba(Data)
    print("I believe that the weather is " + str(Prediction))
    print("There is a " + str(Probability) + " chance that this is true")

print("Loading Demo Data")
print(" ")
print("Data Loaded")
print("Getting Predictions")
print(" ")
Data = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Random Data.csv')))
Predict()

