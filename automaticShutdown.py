import numpy as np
import cv2
import time
import os

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

time_shutdown = 0

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set Width
cap.set(4, 480) # set Height

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for x, y, w, h in faces:
        faces = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = frame[y : y + h, x : x + w]
        roi_gray = gray[y : y + h, x : x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_color,
            scaleFactor= 1.1,
            minNeighbors=5,
            minSize=(3, 3),
        )

        if len(eyes) != 0:
            time_shutdown = 0
            cv2.putText(faces, 'Tidak Tidur', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            time_shutdown += 1
            cv2.putText(faces, 'Tidur', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if time_shutdown == 1000:
        print('waiting for shutdown....')
        time.sleep(3)
        # os.system("shutdown /s /t 1")     # Shutdown
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")    # Sleep
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press q to quit
        break
    
cap.release()
cv2.destroyAllWindows()
