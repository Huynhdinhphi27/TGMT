import cv2 as cv
import numpy as np
import time

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    base_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is not None:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (21,21), 0)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break    
        if base_frame is None:
            base_frame = gray
            continue

        delta = cv.absdiff(base_frame, gray)
        nguong = cv.threshold(delta, 40, 255, cv.THRESH_BINARY)[1]

        # Erosion to remove small noise, then dilation
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        nguong = cv.erode(nguong, kernel, iterations=1)
        nguong = cv.dilate(nguong, kernel, iterations=2)

        bien,_ = cv.findContours(nguong, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for b in bien:
            if cv.contourArea(b) < 500:
                continue
            (x,y,w,h) = cv.boundingRect(b)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # Update base_frame periodically for better adaptation
        base_frame = cv.addWeighted(base_frame, 0.95, gray, 0.05, 0)
        
        cv.imshow("Webcam", frame)

        cv.imshow("Nguong", nguong)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    #cv.waitkey(0)
    cv.destroyAllWindows()