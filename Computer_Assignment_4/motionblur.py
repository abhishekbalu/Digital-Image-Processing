import cv2
import numpy as np 
  
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
  
# Specify the kernel size. 
# The greater the size, the more the motion. 
kernel_size = 15
  
# Create the vertical kernel. 
kernel_v = np.zeros((kernel_size, kernel_size)) 
  
# Create a copy of the same for creating the horizontal kernel. 
kernel_h = np.copy(kernel_v) 
  
# Fill the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
  
# Normalize. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 
  
# Apply the vertical kernel. 
vertical_mb = cv2.filter2D(img, -1, kernel_v) 
  
# Apply the horizontal kernel. 
horizonal_mb = cv2.filter2D(img, -1, kernel_h) 
  
# Save the outputs. 
cv2.imwrite('vertical.jpg', vertical_mb) 
cv2.imwrite('horizontal2.png', horizonal_mb) 