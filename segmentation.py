#importing opencv
import cv2
#function for making segmentaion of the testing image
def segm():
    
    #load image from file
    originalImage = cv2.imread('/home/tech/Desktop/2 no.jpeg')
    
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    #normal image converting to different scale
    cv2.imshow('Black white image', blackAndWhiteImage)
    cv2.imshow('Original image',originalImage)
    cv2.imshow('Gray image', grayImage)
    #kill the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()