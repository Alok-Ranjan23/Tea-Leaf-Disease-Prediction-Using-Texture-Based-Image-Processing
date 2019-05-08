'''
Segement portions of an image of your choice
'''
import argparse
import cv2
import numpy as np
import os

drawing = False #--- true if mouse is pressed
ix, iy = -1,-1

#--- mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, l, masked_image
    if event == cv2.EVENT_LBUTTONDOWN:
        l = []
        drawing = True
        ix, iy = x, y
        l.append([x, y])

    elif event == cv2.EVENT_MOUSEMOVE: 
        if drawing == True:
                l.append([x, y])
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
#                print([x, y])
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.fillPoly(black, [np.asarray(l)], (255, 255, 255))
        b_th = cv2.threshold(black[:,:,1], 100, 255, cv2.THRESH_BINARY)[1]
        masked_image = cv2.bitwise_and(img2, img2, mask = b_th)
        


#path = r'C:\Users\selwyn77\Desktop\Stack\mask\Interactive_segmenting'        
#img = cv2.imread(os.path.join(path, 'zebra.jpg'))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
 
img = cv2.imread(args["image"], 1)
img = cv2.resize(img, (1040, 752)) 
masked_image = img.copy()
black = np.zeros(img.shape, img.dtype)

img2 = img.copy()
img3 = img.copy()
black3 = black.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    cv2.imshow('black', black)
    cv2.imshow('masked_image', masked_image)

    k = cv2.waitKey(1) & 0xFF
    if k == 32:          #--- Press spacebar to clear screen ---
        img = img3.copy()
        black = black3.copy()
        masked_image = img3.copy()
    elif k == 115:      #--- Press lower case s to save and exit ---
        cv2.imwrite(os.path.join('Masked_image.jpg'), masked_image)
        cv2.imwrite(os.path.join('Mask.jpg'), black)
        break
    elif k == 27:       #--- Press 'Esc' to exit without saving ---    
        break

cv2.destroyAllWindows()
