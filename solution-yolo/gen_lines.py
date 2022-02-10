import sys
import numpy as np
import cv2
from numpy.lib.npyio import save

count_points = 0
height, width = 0, 0
x1, y1 = 0, 0
x2, y2 = 0, 0
intermediate_coor_h = []
intermediate_coor_v = [] 
start_end_coords = []
start_end = 0

def callmouse(event, x, y, flags, params):
    global count_points
    global shape
    global x1, y1, x2, y2
    global intermediate_coor_h, intermediate_coor_v
    global start_end, start_end_coords  

    if event == cv2.EVENT_LBUTTONDOWN:
        count_points = count_points + 1
        if count_points == 1:
            x1, y1 = x, y
        if count_points == 2:
            x2, y2 = x, y
            choice = str(params)
            if choice == "hline":
                m = (y1-y2)/(x1-x2)
                b = y1 - m*x1
                y_0 = m*0 + b  
                y_w = m*width + b
                start_end_coords.append([[x1, y1], [x2, y2]]) #if start_end < 2 else intermediate_coor_h.append([[x1, y1], [x2, y2]])    # save coords  
                cv2.line(img, (0, int(y_0)), (int(width), int(y_w)), (0, 255, 0), 1)
                cv2.imshow("Display window", img) 
            #if choice == "vvline":  
            #    intermediate_coor_v.append([[x1, y1], [x2, y2]])    # save coords  
            #    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)     
            #    cv2.imshow("Display window", img)   # draw line 

            count_points = 0
            start_end = start_end + 1     


## VID2FRAMES : uncomment this section to convert video to frames   ################################################################
# vidcap = cv2.VideoCapture(sys.argv[1])   
# success,image = vidcap.read()
# count = 0
# while success:
#   cv2.imwrite("{}/frame_{}.jpg".format(sys.argv[2], count), image)     # save frame as JPEG file      
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1


# GEN LINES ....
img = cv2.imread('frames/frame545.jpg', cv2.IMREAD_UNCHANGED) #6
height, width = img.shape[:2]
print(img.shape)
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )
cv2.imshow("Display window", img)

print("\n1- vertical lines\n2- Horizontal lines\n")

choice = 49

while True:
    choice = cv2.waitKey(0);

    if choice == 81 or choice == 113:
        np.savez("vp_coords", start_end_coords=start_end_coords, intermediate_coords_h=intermediate_coor_h, intermediate_coords_v=intermediate_coor_v)           # save the coordinates     # save the coordinates 
        exit()

    if choice == 49:
        cv2.setMouseCallback("Display window", callmouse, 'hline')

    if choice == 50:
        cv2.setMouseCallback("Display window", callmouse, 'vvline') 


cv2.destroyAllWindows();


