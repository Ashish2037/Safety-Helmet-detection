from threading import Thread
import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

# Function to process frames and perform object detection
def process_frames():
    global img, results
    
    while True:
        success, img = cap.read()
        if not success:
            break
        img_resized = cv2.resize(img, (640, 480))  # Resize image to reduce processing time
        results = model(img_resized, stream=True)

# Main function
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # For Webcam
    model = YOLO("ppe.pt")

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)

    # Initialize results variable
    results = None

    # Start thread for processing frames
    frame_thread = Thread(target=process_frames)
    frame_thread.start()

    while True:
        if results:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]

                    if conf > 0.8:
                        if currentClass == 'NO-Hardhat':
                            myColor = (0, 0, 255)
                            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                           colorT=(255, 255, 255), colorR=myColor, offset=5)
                        elif currentClass == 'Hardhat':
                            myColor = (0, 255, 0)
                            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                            cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                           colorT=(255, 255, 255), colorR=myColor, offset=5)

                        

            # Show processed frame
            cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
