# SUPERVISED LEARNING - Rainfall Prediction in India (1991-2015)

import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import sklearn
from sklearn.model_selection import train_test_split

##### BRING THE CLASSIFIERS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import datasets, linear_model

#Grab the excel file and make it your dataset
Excel = r'\Users\amoah_k\Desktop\T Hacks\Dataset/Dataset.xls'
dataset = pd.read_excel(Excel)

print ("1) Temperature: {}".format(dataset['Temperature']))

#Splits the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(database['Temperature'], database['Rainfall'], random_state = 3)

print(X_train


