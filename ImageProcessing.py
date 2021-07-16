import numpy
import cv2
# import image from PIL


def resizing(img, newsize=[128,128]):
    return cv2.resize(img, newsize, interpolation = INTER_AREA) 



def to_greyscale(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
