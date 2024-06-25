import streamlit as st
import cv2
import math
from ultralytics import YOLO
import cvzone
import time

# Initialize model and class names
model = YOLO("best.pt")
classNames = ['Hardhat', 'NO-Hardhat']

# Function to process frames and perform object detection
def process_frame(img):
    img_resized = cv2.resize(img, (480, 360))  # Resize image to reduce processing time
    results = model(img_resized, stream=True)
    counter = 0
    currentClasses = []  # List to store the detected classes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = box.conf[0]
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            currentClasses.append(currentClass)  # Append the current class to the list
            counter += 1

            if conf > 0.5:
                color = (0, 255, 0) if currentClass == 'Hardhat' else (0, 0, 255)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(img_resized, f'{classNames[cls]} {conf:.2f}',
                                   (max(0, x1), max(25, y1 - 15)), scale=0.8, thickness=1, colorB=color,
                                   colorT=(255, 255, 255), colorR=color, offset=3)
    return img_resized, counter, currentClasses

# Main function
def main():
    st.title("Safety Helmet Detection")
                                
    # Layout the sidebar and main content
    sidebar = st.sidebar
    sidebar.header("Controls")
    start_button = sidebar.button("Start Detection")
    stop_button = sidebar.button("Stop Detection")

    # Create two columns for layout
    col1, col2 = st.columns([5, 1])

    # Placeholders for video and detection info
    with col1:
        stframe = st.empty()

    with col2:
        st.markdown("**Person Count:**")
        person_count_placeholder = st.empty()
        st.markdown("**Detection:**")
        class_placeholder = st.empty()

    cap = None
    if start_button:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap and cap.isOpened():
        success, img = cap.read()
        if not success or stop_button:
            break

        img_processed, counter, currentClasses = process_frame(img)
        stframe.image(img_processed, channels="BGR", width=540)  # Adjusted width here

        # Update person count
        person_count_placeholder.text(f"{counter}")

        # Update detected classes
        class_placeholder.markdown("\n".join(f"- {cls}" for cls in currentClasses))

        time.sleep(0.08)  # Add a small delay to reduce CPU usage

    if cap:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
