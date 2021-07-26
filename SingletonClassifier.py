
import opencv2 as cv2
# import image from PIL
import ImageProcessing
from TrainingModels import ModelClasses


class SingletonClassifier:

  def __init__(self):
    pass

  cap = cv2.VideoCapture(0)

  if not (cap.isOpened()):
      print("Could not open video device")


  while(True): 
  # Capture frame-by-frame
      ret, img = cap.read()

      if ret:
  #resize image
        image-processing.resize(frame)

  #transform to greyscale
        image-processing.to_greyscale(frame)

  # Display the resulting frame
        cv2.imshow(‘preview’,frame)

  #Pass image to classifier

      else:
        print('Unable to access camera device')


  # Grabs, decodes and returns the next video frame.

  #Waits for a user input to quit the application

      if cv2.waitKey(1) & 0xFF == ord(‘q’):
          break
