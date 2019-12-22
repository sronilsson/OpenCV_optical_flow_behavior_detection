import cv2 as cv
import numpy as np
from itertools import chain
import cv2
import os

cap = cv.VideoCapture("video file path")
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(first_frame)
mask[..., 1] = 255
savePath = 'image save path
loop = 0

while(cap.isOpened()):
    imName = str(loop) + 'save image file type'
    imPath = os.path.join(savePath, imName)
    print(imPath)
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    listMagnitude = list(chain.from_iterable(magnitude))
    listMagnitude.sort(reverse=True)
    listMagnitude = list(np.around(np.array(listMagnitude), 5))
    meanListMag = (sum(listMagnitude) / len(listMagnitude))
    print(meanListMag)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    if meanListMag > 0.30:
        cv2.putText(frame, str('attack prediction'), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rgb, str('attack prediction'), (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    concat = np.concatenate((frame, rgb), axis=1)
    cv2.imwrite(imPath, concat)
    prev_gray = gray
    loop+=1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()