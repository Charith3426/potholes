import streamlit as st
from ultralytics import YOLO
import cv2
import math
import cvzone
import tempfile

# Load YOLO model
model = YOLO('best.pt')
classnames = ['Potholes']

st.title('Pothole Detection System')

f = st.file_uploader("Upload a video", type=["mp4", "avi"])
if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())

    cap = cv2.VideoCapture(tfile.name)

    frameST = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)
                    if classnames[Class] == "Potholes":
                        st.write("Pothole detected")
                    else:
                        st.write("Not a pothole detected")

        frameST.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
