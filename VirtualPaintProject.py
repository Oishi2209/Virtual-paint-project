import cv2
import numpy as np
from collections import deque

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   

# HSV ranges: [hMin, sMin, vMin, hMax, sMax, vMax]
myColors = [
    [35, 80, 60, 85, 255, 255],      # Green
    [95, 80, 50, 130, 255, 255],     # Blue
    [135, 80, 60, 175, 255, 255],    # Magenta / pink-purple
]
myColorValues = [[0, 255, 0],        # Green (BGR)
                 [255, 0, 0],        # Blue (BGR)
                 [255, 0, 255]]      # Magenta (BGR)

MIN_AREA = 1500            # noise
SMOOTH_ALPHA = 0.3          
MAX_JUMP = 80               
HOLD_FRAMES = 5            
DOT_RADIUS = 8
MAX_POINTS = 400           
USE_SKIN_REJECTION = True

kernel = np.ones((5, 5), np.uint8)

smoothPts = [None] * len(myColors)  
lostFrames = [0] * len(myColors)     
myPoints = deque(maxlen=MAX_POINTS)  


def skinMask(imgBGR):
    ycrcb = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def getColorMask(imgHSV, color, skin):
    mask = cv2.inRange(imgHSV, np.array(color[0:3]), np.array(color[3:6]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if skin is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    return mask


def getCenter(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


showMasks = False

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)   

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    skin = skinMask(blurred) if USE_SKIN_REJECTION else None

    cursors = []
    maskViews = []

    for i, color in enumerate(myColors):
        mask = getColorMask(imgHSV, color, skin)
        maskViews.append(mask)
        center = getCenter(mask)

        if center is not None and smoothPts[i] is not None:
            dx = center[0] - smoothPts[i][0]
            dy = center[1] - smoothPts[i][1]
            if (dx * dx + dy * dy) ** 0.5 > MAX_JUMP:
                center = None

        if center is None:
            lostFrames[i] += 1
            if lostFrames[i] > HOLD_FRAMES:
                smoothPts[i] = None   
            continue

        lostFrames[i] = 0
        cx, cy = center
        if smoothPts[i] is None:
            smoothPts[i] = (float(cx), float(cy))
        else:
            sx, sy = smoothPts[i]
            smoothPts[i] = (sx + (cx - sx) * SMOOTH_ALPHA,
                            sy + (cy - sy) * SMOOTH_ALPHA)

        pt = (int(smoothPts[i][0]), int(smoothPts[i][1]))
        myPoints.append([pt[0], pt[1], i])
        cursors.append((pt, myColorValues[i]))

    imgResult = img.copy()
    for x, y, c in myPoints:
        cv2.circle(imgResult, (x, y), DOT_RADIUS, myColorValues[c], cv2.FILLED)

    cv2.imshow("Result", imgResult)

    if showMasks:
        for i, m in enumerate(maskViews):
            cv2.imshow(f"Mask {i}", m)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        myPoints.clear()
    if key == ord('m'):
        showMasks = not showMasks
        if not showMasks:
            for i in range(len(myColors)):
                try:
                    cv2.destroyWindow(f"Mask {i}")
                except cv2.error:
                    pass

cap.release()
cv2.destroyAllWindows()