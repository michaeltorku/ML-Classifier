# import 
#Input:- Load the Live Feed from the camera
from ModelClasses import *
from stereovision.stereo_cameras import StereoPair



def __init__(self):
  self._cameras = StereoPair([1,2])






def run(self, full_operation_mode=False):
  if full_operation_mode:
    # special set up
    pass


  while True:
    left_image, right_image = self._cameras.get_frames()
    # integrate image from two cameras
    # process image of 
    # pass integrated feed into classifier
    #highlight determined objects in feed
    #output feed
    #break if ctrl+c or break key detected
    

#Process:- Object Detector 

#Output:- Whatever
Kmodel = KNN(data) #---->score of knn
SVMmodel = SVM(data)
DTreeModel = DTree(data)
NeuralModel = Neural(data)
RegrModel = Regression(data)

# using the integrator as a model and using it alongside the rest
# because the one might be better

def indexer():
  

  return 
  # PCA() to make data easier to process

  # index it

  Kmodel says its a dog - 0.76 ->
  SVmodel says its a cat - 0.58 ->
  Dtree says its a cat - 0.66 ->


# look at accuracy wrt to their output -> so we get a sense of how accrate
#it is regarding its 
# train opinions of the models
# pick most accurate model -> Kmodel
# 