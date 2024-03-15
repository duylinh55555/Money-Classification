import numpy as np
import cv2
import time
import os

label = "60"

cap = cv2.VideoCapture(0)

frame_number = 0
while(True):
    frame_number += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    # Stream on screen
    cv2.imshow('frame',frame)

    # Saving
    if frame_number>=60:
        print("Image's number captured = ", frame_number - 60)
        
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(frame_number) + ".png",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()