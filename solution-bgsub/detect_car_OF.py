#part of the this file work was inspired by Chuan-en Lin's code:
#https://nanonets.com/blog/optical-flow/

import cv2
import numpy as np


cap = cv2.VideoCapture('traffic.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

background=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# Read until video is completed

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)

#paramether for image processing-----------------
kernel=None

#listas de registo-------------------------------
moving_cars={}
prev_updates=[]
updates=[] #esta lista vai sempre crescer

k=0
current_k=0
prev_centroids=[]


# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = (0, 255, 0) # Variable for color to draw optical flow track
ret, frame = cap.read() 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

car_count=0

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    #faz o optical flow dos pixeis anteriores, ou seja dos carros que já estão na lista--------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params) # Calculates sparse optical flow by Lucas-Kanade method
    cinza=gray.copy()
    cars_lst=list(moving_cars.keys())
    for car in cars_lst:
      prev=[[[moving_cars[car][len(moving_cars[car])-1][0],moving_cars[car][len(moving_cars[car])-1][1] ]]]
      prev=np.array(prev)
      prev=np.float32(prev.reshape(-1, 1, 2))
    
      next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)


      good_old = prev[status == 1].astype(int) # Selects good feature points for previous position
      good_new = next[status == 1].astype(int) # Selects good feature points for next position
      # Draws the optical flow tracks
      for i, (new, old) in enumerate(zip(good_new, good_old)):
          a, b = new.ravel()  #  (x, y) coordinates for new point
          c, d = old.ravel() # R (x, y) coordinates for old point
          #mask = cv.line(mask, (a, b), (c, d), color, 2)  # Draws line between new and old position 
          frame = cv2.circle(frame, (a, b), 6, color, -1)  # Draws filled circle  at new position. thickness = -1 fills the circle completely, 3 is the radius 
          if b>550:
            frame = cv2.circle(frame, (a, b), 10, (255,0,0), -1)  # Draws filled circle  at new position. thickness = -1 fills the circle completely, 3 is the radius 
            total_frames=len(moving_cars[car])
            sec=total_frames/30
            velocidade=25/sec #frames/distancia neste caso consideramos 12 metros
            velocidadekm=velocidade*3600/1000
            print(f"{car_count}:{velocidadekm} km/h")
            car_count+=1
            cv2.putText(frame, str(velocidadekm), (a, b-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
            moving_cars.pop(car)


      tuple=(np.uint32(next[0][0][0]).item(),np.uint32(next[0][0][1]).item())
      if car in moving_cars:
        moving_cars[car].append(tuple)

    prev_gray = cinza.copy()



    #deteta novos carros--------------------------------------
    mask=background.apply(frame) 
    #remove gray shadows:
    _,mask=cv2.threshold(mask,250,255,cv2.THRESH_BINARY)

    mask=cv2.erode(mask,kernel,iterations=4)
    mask=cv2.dilate(mask,kernel,iterations=10)

    
    h,w,n=frame.shape
    cv2.line(frame, (0,400), (w,400), (0,255,0), 2)
    cv2.line(frame, (0,550), (w,550), (0,255,0), 2)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #lista de centroides da frame actual
    centroids=[]
    
    for cnt in contours:
        if cv2.contourArea(cnt)>400:
            x,y,w,h=cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx,cy))

            #Verificar se o centroide faz parte de um carro que ainda n passou a linha
            #esta parte funciona bem
            for c in prev_centroids:
              if cy>400 and c[1]<400:
                dist=np.sqrt((cx-c[0])**2+(cy-c[1])**2)
                if dist<15:
                  #deteta um carro a passar a linha de cima
                  updates.append(1)
                  #current_k+=1

                  cv2.putText(frame, str(current_k), (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

                  cv2.drawMarker(frame, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=20,
                                        line_type=cv2.LINE_AA)

                  moving_cars[current_k]=[]
                  moving_cars[current_k].append((cx,cy))
                  current_k+=1


                  break


            cv2.imshow('Original Frame',frame)
                



    prev_centroids=centroids

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()