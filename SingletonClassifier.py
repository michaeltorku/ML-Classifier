
import opencv2 as cv2
# import image from PIL
import ImageProcessing
from TrainingModels import ModelClasses

#import 

class SingletonClassifier:

  def __init__(self):
    pass

  cap = cv2.VideoCapture(0)

  if not (cap.isOpened()):
      print("Could not open video device")

  def run():
    while(True): 
    # Capture frame-by-frame
        ret, img = cap.read()

        if ret:
    #resize image
          ImageProcessing.resize(img)

    #transform to greyscale
          ImageProcessing.to_greyscale(img)

    # Display the resulting frame
          cv2.imshow(‘preview’,img)

    #Pass image to classifier
    ........................

        else:
          print('Unable to access camera device')


    # Grabs, decodes and returns the next video frame.

    #Waits for a user input to quit the application

        if cv2.waitKey(1) & 0xFF == ord(‘q’):
            break
