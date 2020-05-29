# -*- coding: utf-8 -*-

import cv2
print('Project Topic : object detection')
print('Digital Image processing')
print('By Abhishek Balasubramaniam and Shwetha Haran')

cascade_bike = 'two_wheeler.xml'
cascade_car = 'cars11.xml'
cascade_bus='Bus_front.xml'
cascade_pedestrian='pedestrian.xml'
video_src = 'project_video.mp4'

cap = cv2.VideoCapture(video_src)

bike_cascade = cv2.CascadeClassifier(cascade_bike)
car_cascade = cv2.CascadeClassifier(cascade_car)
bus_cascade = cv2.CascadeClassifier(cascade_bus)
pedestrian_cascade = cv2.CascadeClassifier(cascade_pedestrian)


cars=0
obs=0
bike=0
bus=0

while True:
    ret, img = cap.read()
    # b,g,r = cv2.split(img)
    # rgb_img = cv2.merge([r,g,b])
    # dst = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)
    # b,g,r = cv2.split(dst)
    # rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,300,200)
    bike = bike_cascade.detectMultiScale(gray,1.19, 1)
    car = car_cascade.detectMultiScale(gray,2, 1)
    bus = bus_cascade.detectMultiScale(gray,1.19, 1)
    pedestrian = pedestrian_cascade.detectMultiScale(gray,2, 3)
   
    for (x,y,w,h) in bike:
    	# bike = bike+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'bike',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
        bike=bike+1
        distance = 8414.7*pow((w)*(h), -0.468)
        print("disance of bike",distance)
        # print("distance",x+w)

    for (x,y,w,h) in car:
    	# car=car+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cars=cars+1
        distance = int(8414.7*pow((w)*(h), -0.468))
        print("disance of car",distance)
        cv2.putText(img,'car: ',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(img,str(distance),(x+60,y), font, .5,(255,0,0),2,cv2.LINE_AA)

    for (x,y,w,h) in bus:
    	# bus =bus+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'bus',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
        bus=bus+1
        distance = 8414.7*pow((w)*(h), -0.468)
        print("disance of bus",distance)
        # print("width",x+w)

    for (x,y,w,h) in pedestrian:
    	# obs=obs+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'obstacle',(x,y), font, .5,(255,0,0),2,cv2.LINE_AA)
        obs=obs+1
        distance = 8414.7*pow((w)*(h), -0.468)
        print("disance of obstacle",distance)
        # print("width",x+w)
    
    cv2.namedWindow('output',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output', 720,720)
    cv2.imshow('output', img)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
            
        break

