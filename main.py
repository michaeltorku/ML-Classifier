
import Singleton-Classifier
import StereoVision-Classifier


if __name__ == '__main__':
  try:
    num = int(input('How many camera devices do you want to use?'))
    if num ==1:
      #call func for one camera
      model = Singleton-Classifier()
    elif num ==2:
      #call func for stereocameras
      model = StereoVision-Classifier()
    else:
      print(f'{num} camera devices is out of pocket.') 
    #call the Classifier into action
    model.run()     
  except ValueError:
    print('Invalid input: Enter an integer.')
  except Exception as e:
    print(e)

