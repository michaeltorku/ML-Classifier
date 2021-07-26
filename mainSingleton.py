
import opencv2 as cv2
import matplotlib


cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")


while(True): 
# Capture frame-by-frame

    ret, frame = cap.read()

# Display the resulting frame

    cv2.imshow(‘preview’,frame)
# Grabs, decodes and returns the next video frame.

#Waits for a user input to quit the application

    if cv2.waitKey(1) & 0xFF == ord(‘q’):
        break
