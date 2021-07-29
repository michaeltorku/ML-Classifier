from TrainingModels import ModelClasses
from ModelClasses import *
from Data import *

class ClassifierModel:
  KNNmodel = KNN(data)
  LogRegModel = LogisticReg(data)
  SVMmodel = SupportVectorMachine(data)
  DTmodel = DecisionTree(data)
  LinPercModel = LinearPerception(data)