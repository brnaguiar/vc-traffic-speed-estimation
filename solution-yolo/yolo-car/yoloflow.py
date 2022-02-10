# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pprint

#  parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",      default="videos/carsv1.mp4",          help="path to input video") #1 1
ap.add_argument("-o", "--output",     default="output/carsyolo1.mp4",        help="path to output video")
ap.add_argument("-y", "--yolo",       default = "yolo-coco",                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,              help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold",  type=float, default=0.3,              help="threshold when applyong non-maxima suppression")
ap.add_argument("-vp", "--vpcoords",  default="vp_coords.npz",              help="default vanishing point coordinates")   
args = vars(ap.parse_args())   

np_data = np.load('vp_coords.npz') 

# get the pair of points to contruct the lines
points = []
for pair in np_data['start_end_coords']:
    points = points + [pair] 

# construct the lines
lines = []
for (p1, p2) in points: 
    lines = lines + [
        [
            (p1[1]-p2[1])/(p1[0]-p2[0]), # slope m
            p1[1] - (p1[1]-p2[1])/(p1[0]-p2[0])*p1[0] # y-intercept b
        ]
    ]
   
# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])  
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42) 
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

#paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO 
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
for i in net.getUnconnectedOutLayers(): 
    print(i)
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]   


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)  
mask = None 
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

tick = 0

prev_image = None
actual_image = None
prev_points = np.array([])   
actual_points = np.array([])  #  ()        
lk_params = dict(winSize = (70,70), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05)) #4040 # 40


prev_iter = 0
current_iter = 0   
vel_array = []   

vel_boxes = []

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read() # read the next frame from the file 
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        mask = np.zeros_like(frame)  

    # draw the horizontal green llines 
    # for (m,b) in lines:
    #     y_0 = int(m*0 + b)
    #     y_w = int(m*W + b)
    #     cv2.line(frame, (0, y_0), (W, y_w), (0, 255, 0), 1)  
         
    actual_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # construct a blob from the input frame 
    # and then perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, # 
                                 1 / 255.0, # scalefactor	multiplier for images values.  
                                 (416, 416),   ##	spatial size for output image 
                                 swapRB=True, #swapRB	flag which indicates that swap first and last channels in 3-channel image is necessary. 
                                 crop=False) # crop	flag which indicates whether image will be cropped after resize or not 
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    points = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)  of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height    
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top  and and left corner of the bounding box 
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))    
                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])   


	# ensure at least one detection exists
    if len(idxs) > 0:
		# loop over the indexes we are keeping
        for i in idxs.flatten(): 
			# extract the bounding box coordinates
            center_x, center_y = np.float32(int(x+(width/2))), np.float32(int(y+(height/2)))   # center of bounding box 
            points = points + [[center_x, center_y]]  # add center of bounding box to list of points 
            (x, y) = (boxes[i][0], boxes[i][1]) # top and left corner...    
            (w, h) = (boxes[i][2], boxes[i][3])
			#draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

    actual_points = np.array(points) 
    actual_points = actual_points.reshape(len(actual_points), 1, 2)  # reshape points to be 2D array      
    output = None              

    if tick % 2 == 0 and tick > 0:    # every 2 frames
        if actual_points.size != 0 and prev_points.size != 0 :  
            prev_iter = current_iter
            current_iter = tick   
            next, status, error = cv2.calcOpticalFlowPyrLK(prev_image, actual_image, prev_points, None, **lk_params)  # calculate optical flow   
            old = prev_points[status == 1].astype(int)
            new = next[status == 1].astype(int) 
            for i, (n,o) in enumerate(zip(new, old)):  
                a, b = n.ravel()  #  (x,y)  
                c, d = o.ravel()
                #mask = cv2.line(mask, (a, b), (c, d), color, 2)        
                #frame = cv2.circle(frame, (a, b), 5, color, -1)  # draw points    
                vb_aux = []
                v = 0
                for j, ((m1,b1), (m2,b2)) in enumerate(zip(lines, lines[1:])):
                    if ((m1*a + b1-b) * (m2*a + b2-b)) < 0:           
                        y_start = int(m1*a + b1)
                        y_end = int(m2*a + b2) 
                        diff_y_street = y_end - y_start # 2m ...
                        diff_y_car = b-d
                        if args["input"] == "videos/carsv1.mp4":
                            dist = diff_y_car*(2 if j % 2 == 0 else 3)/diff_y_street # linha = 2m, intervalo = 3m 
                        else:
                            dist = diff_y_car*(0.5 if j % 2 == 0 else 10)/diff_y_street  
                        v = (dist*30*(current_iter-prev_iter))*(3600/1000)                                  
                vb_aux = vb_aux + [[a, b, v] if args["input"] == "videos/carsv1.mp4" else [a, b, abs(v)]]   #videovideo video  
                print("Velocidade; ", v)    
                vel_boxes = vb_aux #vel_boxes      

    if vel_boxes != []:
        #print(vel_boxes)
        for i, obj in enumerate(vel_boxes):
            #print("obj", obj)
            a, b, _ = obj
            _, _, v = vel_boxes[-i]
            if v > 0:
                text = "Speed: {} km/h".format(round(v, 1))  
                cv2.putText(frame, text, (a, b + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)    
    
    if tick % 2 == 0 and tick > 0:    # every 2 frames
        if actual_points.size != 0 and prev_points.size != 0 :  
            prev_image = actual_image.copy() 
            prev_points = new.reshape(-1, 1, 2)  #  
            output = cv2.add(frame, mask)  # add mask to frame 
            cv2.imshow("Yolo", output) ### Frame 
    else:
        prev_image = actual_image.copy() 
        prev_points = actual_points  
        cv2.imshow("Yolo", frame)    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
  
	# check if the video writer is None
    if writer is None: 
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MP4V") #  
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
		# some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total)) 
	# write the output frame to disk
    writer.write(frame)

    tick = tick + 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()  