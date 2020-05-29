import numpy as np
import cv2
# from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

print('Project Topic : object detection')
print('Digital Image processing')
print('By Abhishek Balasubramaniam and Shwetha Haran')

# define the cascade classifiers.
cascade_bike = 'bike.xml'
cascade_car = 'cars11.xml'
cascade_bus='Bus_front.xml'
cascade_pedestrian='/home/abhishek/Desktop/dip project/opencv-master1/Computer-Vision-Tutorial-master/Haarcascades/haarcascade_fullbody.xml'

bike_cascade = cv2.CascadeClassifier(cascade_bike)
car_cascade = cv2.CascadeClassifier(cascade_car)
bus_cascade = cv2.CascadeClassifier(cascade_bus)
pedestrian_cascade = cv2.CascadeClassifier(cascade_pedestrian)



#initialize variables.
cars=0
obs=0
bike=0
bus=0

# get input images.
img = cv2.imread('')
# cv2.imshow('image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,300,200)
image = imutils.resize(img, width=min(400, img.shape[1]))


# initialize the cascade.
bike = bike_cascade.detectMultiScale(gray,2, 1)
car = car_cascade.detectMultiScale(gray,3, 2)
bus = bus_cascade.detectMultiScale(gray,1.19, 1)
pedestrian = pedestrian_cascade.detectMultiScale(gray,1.19, 1)
font = cv2.FONT_HERSHEY_SIMPLEX

# initialize hog classifiers. 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
# # do the detection here.
for (x,y,w,h) in bike:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
	cv2.putText(img,'bike',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
	bike=bike+1
	distance = 8414.7*pow((w)*(h), -0.468) / 100
	print("disance of bike",distance)

for (x,y,w,h) in car:

	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),5)
	cv2.putText(img,'car',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
	cars=cars+1
	distance = 8414.7*pow((w)*(h), -0.468) / 100
	print("disance of car",w)

for (x,y,w,h) in bus:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
	cv2.putText(img,'bus',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
	bus=bus+1
	distance = 8414.7*pow((w)*(h), -0.468) / 100
	print("disance of bus",distance)
    
for (x, y, w, h) in pick:
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(img,'pedestrian',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
	distance = 8414.7*pow((w)*(h), -0.468) / 100
	print("disance of pedestrian",distance)
	# cv2.imshow('Pedestrians', img)
	obs=obs+1
	# distance = 8414.7*pow((x)*(y), -0.468) / 100
	# print("disance of obstacle",distance)

# output image.
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output', 720,720)
cv2.imshow('output', img)

# check for interrupt.
k = cv2.waitKey(0)
if cv2.waitKey(25) & 0xFF == ord('q'):
	cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()




