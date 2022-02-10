#Some of the code was inspired in 'Code Files for Vehicle Detection with OpenCV (Contours + Background Subtraction)' written by Taha Anwar 

import cv2
import numpy as np


cap = cv2.VideoCapture('traffic.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

background=cv2.createBackgroundSubtractorMOG2(detectShadows=True)


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)

#parametro para processamento de imagem-----------------
kernel=None

#listas de registo--------------------------------------
moving_cars={}
prev_updates=[]
updates=[] #esta lista vai sempre crescer

k=0
current_k=0
prev_centroids=[]
moving_cars_frame_count={}




while(cap.isOpened()):
  
  ret, frame = cap.read()
  if ret == True:
    #aplicacao de background subtraction
    mask=background.apply(frame)

    _,mask=cv2.threshold(mask,250,255,cv2.THRESH_BINARY)

    mask=cv2.erode(mask,kernel,iterations=4)
    mask=cv2.dilate(mask,kernel,iterations=10)
    

    h,w,n=frame.shape
    cv2.line(frame, (0,400), (w,400), (0,255,0), 2)
    cv2.line(frame, (0,550), (w,550), (0,255,0), 2)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #lista de centroides da frame atual
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

                  moving_cars[current_k]=[]
                  moving_cars[current_k].append((cx,cy))
                  moving_cars_frame_count[current_k]=1
                  cv2.putText(frame, str(current_k), (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                  current_k+=1
                  cv2.drawMarker(frame, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=20,
                                        line_type=cv2.LINE_AA)
                  cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                      line_type=cv2.LINE_AA)

                  break
                

              
            #Verificar se o centroide é um update a um dos outros carros
            cycle=list(moving_cars.keys())
            cycle.sort()
            for i in cycle: #loop sobre os indices que ainda estao em movimento
              if cy>moving_cars[i][len(moving_cars[i])-1][1]:#so calcula para carros a descer a estrada
                dist_to_cars=np.sqrt((cx-moving_cars[i][len(moving_cars[i])-1][0])**2+(cy-moving_cars[i][len(moving_cars[i])-1][1])**2)
                if dist_to_cars<15: 
                  ##detetou o carro i
                  moving_cars[i].append((cx,cy))

                  #regista que o carro foi atualizado nesta frame:
                  updates[i]+=1

                  cv2.putText(frame, str(i), (cx, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
                  if len(moving_cars[i])>2:
                    if moving_cars[i][len(moving_cars[i])-2][1]>550: #significa que já passou a linha de baixo

                      frame=cv2.drawMarker(frame, (moving_cars[i][len(moving_cars[i])-2][0], moving_cars[i][len(moving_cars[i])-2][1]), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=20,
                                          line_type=cv2.LINE_AA)

               
                      moving_cars.pop(i)
                      sec=moving_cars_frame_count[i]/30
                      velocidade=25/sec #frames/distancia neste caso consideramos 12 metros
                      print('velocidade')
                      velocidadekm=velocidade*3600/1000
                      print(f"{i}:{velocidadekm} km/h")




            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                      line_type=cv2.LINE_AA)

    ##Verifica se os carros foram atualizados:

    
    cycle=list(moving_cars.keys()) #ciclo apenas sobre os carros que ainda estao em movimento
    for i in cycle:
      if prev_updates[i]>1:
       if prev_updates[i]==updates[i]:
           
         #significa que a posicao do carro nao foi atualizada
         #calcula as distancias entre as ultimas duas posicoes que o carro teve
          dy=moving_cars[i][len(moving_cars[i])-1][1]-moving_cars[i][len(moving_cars[i])-2][1]
          dx=moving_cars[i][len(moving_cars[i])-1][0]-moving_cars[i][len(moving_cars[i])-2][0]



          new_coords=(moving_cars[i][len(moving_cars[i])-1][0]+dx,moving_cars[i][len(moving_cars[i])-1][1]+dy)
          cv2.drawMarker(frame, (new_coords[0], new_coords[1]), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                      line_type=cv2.LINE_AA)
          cv2.putText(frame, str(i), (new_coords[0], new_coords[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

          if new_coords[1]<550:
            #atualiza a posicao do carro:
            moving_cars[i].append(new_coords)

            cv2.putText(frame, str(i), (new_coords[0], new_coords[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
            updates[i]+=1
            
          else:
              cv2.drawMarker(frame, (new_coords[0], new_coords[1]), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=20,
                                        line_type=cv2.LINE_AA)


              moving_cars.pop(i)

              #calculo da velocidade
              sec=moving_cars_frame_count[i]/30
              velocidade=25/sec #frames/distancia neste caso consideramos 12 metros
              velocidadekm=velocidade*3600/1000
              print(f"{i}:{velocidadekm} km/h")
              cv2.putText(frame, str(velocidadekm), (new_coords[0], new_coords[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

 
    cv2.imshow('Original Frame',frame)
    for car in cycle:
      moving_cars_frame_count[i]+=1

    prev_updates=updates



    prev_centroids=centroids


    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break


cap.release()

# Closes all the frames
cv2.destroyAllWindows()