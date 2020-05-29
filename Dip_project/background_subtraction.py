import numpy as np
import cv2




# cap = cv2.VideoCapture('/home/abhishek/Desktop/dip project/SDC-Lane-and-Vehicle-Detection-Tracking-master/Part II - Adv Lane Detection and Road Features/videos/project_video.mp4')
# fgbg = cv2.createBackgroundSubtractorMOG2()

# while(1):
#     ret, frame = cap.read()
#     gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # median = cv2.medianBlur(frame, 5)
#     # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     # im =  cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
#     # gauss = cv2.GaussianBlur(im, (5,5), 0)


#     fgmask = fgbg.apply(frame)
#     #
    
#     # images = np.concatenate((median, gauss), axis=1)
#     # cv2.imshow('img', images)
#     cv2.imshow('fgmask',frame)
#     cv2.imshow('frame',fgmask)

    
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
    
import cv2

background = cv2.imread('/home/abhishek/Desktop/dip project/Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades-master/Main Project/Main Project/Bike Detection/background1.png')
overlay = cv2.imread('/home/abhishek/Pictures/input.png')

added_image = cv2.addWeighted(overlay,0.1,background,0.7,5)
cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.imshow('output', added_image)

    # cv2.resizeWindow('output', 720,720)
# cap.release()k = cv2.waitKey(0)
k = cv2.waitKey(0)
if cv2.waitKey(25) & 0xFF == ord('q'):
	cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
