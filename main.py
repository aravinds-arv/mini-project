import cv2
import time
import math
import mediapipe as mp
from cvzone.SerialModule import SerialObject

# initialize video capture and serial comm
cap = cv2.VideoCapture(0)
mySerial = SerialObject("COM5", 9600, 2)

# solutions module
solutions = mp.solutions
mpHands = solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75) # type: ignore
mpDraw = solutions.drawing_utils # type: ignore

# fps variables
cTime = 0
pTime = 0

# required landmarks
fTips = [4, 8, 12, 16, 20]
fJoints = [1, 5, 9, 13, 17]

# helper function to process and display hands
def findHands(img: cv2.Mat):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpHands.process(imgRGB)
    allHands = []
    handPresence = False
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        handPresence = True
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            ## lmList
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)

            ## bounding box dimensions
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                        bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            # flip type
            if handType.classification[0].label == "Right":
                myHand["type"] = "Left"
            else:
                myHand["type"] = "Right"
            
            allHands.append(myHand)

            ## draw
            mpDraw.draw_landmarks(img, handLms,
                                        solutions.hands.HAND_CONNECTIONS) # type: ignore
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), 
                          (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), 
                          (255, 0, 255), 2)
            cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 
                        2, (255, 0, 255), 2)
            
    return img, allHands, handPresence

# helper function to estimate lenghts of opened fingers
def estimateLengths(allHands: list, handPresence: bool, fTips: list, fJoints: list):
    scaledFLengths = []
    fLengths = []
    for myHand in allHands:
        myLmList = myHand["lmList"]
        # find distance bw finger tips and finger joints
        if handPresence:
            diagLength = math.hypot(myLmList[5][0] - myLmList[0][0], 
                                    myLmList[5][1] - myLmList[0][1])
            for id in range(0, 5):
                length = (math.hypot(myLmList[fTips[id]][0] - myLmList[fJoints[id]][0], 
                                     myLmList[fTips[id]][0] - myLmList[fJoints[id]][0]))
                print(id)
                print(diagLength)
                print(length)
                print()
                scaledFLengths.append(length)
            for length in scaledFLengths:
                proportion = int(8 * (length / diagLength))
                fLengths.append(proportion)

    return fLengths
    
# helper function to estimate degree of rotation of each finger
def estimateRotation(fLengths: list):
    avgFLengths = [12, 8, 9, 8, 6]
    if fLengths:
        if len(fLengths) == 5:
            fAngles = [int((fLengths[i] / avgFLengths[i]) * 180)
                       for i in range(5)]
            
            return fAngles

# execution loop
while True:
    # read frames from the capture
    success, img = cap.read()
    
    # hand detection
    img, allHands, handPresence = findHands(img)

    # finger length estimation
    fLengths = estimateLengths(allHands, handPresence, fTips, fJoints)
    # print(fLengths)

    # finger rotation/gesture estimation
    fAngles = estimateRotation(fLengths)
    # print(fAngles)
    # print()
            
    # fps estimation and display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    
    # sending the serial data to the mc
    if fLengths:
        mySerial.sendData(fLengths)
    
    # displaying the result
    cv2.imshow("Image", img)
    cv2.waitKey(1)
