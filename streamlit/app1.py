import streamlit as st
from ultralytics import YOLO
import cv2
import math
import cvzone
import tempfile
import numpy as np
import base64

def set_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (specify the path to your image file)
set_background_image('basic.jpg')



# Load YOLO model
model = YOLO('best.pt')
classnames = ['Potholes']

st.title('Pothole Detection System')

# Add options for user input: Image, Video, or Live Stream
input_option = st.radio("Choose input type", ('Image', 'Video', 'Live Stream'))

def process_frame(frame):
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
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                if classnames[Class] == "Potholes":
                    st.write("Pothole detected")
                else:
                    st.write("Not a pothole detected")
    return frame

if input_option == 'Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        processed_image = process_frame(image)
        st.image(processed_image, channels="BGR", use_column_width=True)

elif input_option == 'Video':
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frameST = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            frameST.image(processed_frame, channels="BGR", use_column_width=True)

        cap.release()

elif input_option == 'Live Stream':
    st.write("Live Stream from Webcam")

    # Streamlit components for displaying webcam feed and processed frames
    frame_window = st.image([])
    frame_processed = st.image([])

    # Start capturing from the webcam
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

    frame_count = 0  # Counter for unique button key

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the original and processed frames
        # frame_window.image(frame, channels='BGR')
        frame_processed.image(processed_frame, channels='BGR')

        # Break the loop if necessary, using a unique key for the button
        if st.button('Stop Streaming', key=f'stop_button_{frame_count}'):
            break

        frame_count += 1

    cap.release()
    # Live streaming is complex in Streamlit and would generally require an external service or setup.

