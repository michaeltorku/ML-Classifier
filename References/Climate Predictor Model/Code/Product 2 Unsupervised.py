import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt
import sklearn
from sklearn.model_selection import train_test_split

Features = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Features.csv')))
Outputs = list(csv.reader(open(r'\Users\amoah_k\Desktop\T Hacks\Dataset/Outputs.csv')))

print("Features and Outputs gathered")

#Splits the data into testing and training

X_train, X_test, y_train, y_test = train_test_split(Features, Outputs, random_state = 0)# For the datatset 0 = 'rain' and 1 = 'snow' and 2 = 'null' IMPORTANT


#DIMENSION REDUCTION WITH PCA

from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

#UNSUPERVISED LEARNING METHODS

from sklearn.cluster import KMeans

def KMM():
    KMM = KMeans(n_clusters = 6)
    KMM.fit(X_train, y_train)
    KMM_predictions = KMM.predict(X_test)
    print(KMM_predictions)
    print(y_test[0])
    print(y_test[1])
    print(y_test[2])
    print(y_test[-1])
    print(y_test[-2])
    print(y_test[-3])
KMM()
