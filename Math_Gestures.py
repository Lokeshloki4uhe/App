import cvzone
import cv2
import numpy as np
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import streamlit as st

# Streamlit configuration
st.set_page_config(layout="wide")
st.image('MathGestures.png')

# Project Summary
st.markdown("""
# Math Gestures

**Math Gestures** is an innovative application that uses hand gesture recognition to allow users to draw and solve math problems on a virtual canvas. Here's how it works:
This application combines computer vision with AI to provide a seamless and interactive experience for solving math problems.
""")

# Instructions for using the app
st.markdown("""
## Instructions:
- **Index Finger Up**: Write on the canvas.
- **Thumb Up**: Clear the canvas.
- **Two Fingers (Index and Middle)**: Move the writing on the canvas.
- **All Fingers Except Little Finger Up**: Submit the drawing to AI for analysis and receive the solution.
""")

# Layout adjustments
col1, col2 = st.columns([3, 1])
with col1:
    run = st.checkbox('Run', value=True)
    stop_button = st.button('Stop')
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.empty()

# Google Generative AI configuration
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = None
if run:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 720)

# Initialize hand detector with minimal parameters
detector = HandDetector(detectionCon=0.3)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers1 = detector.fingersUp(hand1)
        return (fingers1, lmList)
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmlist[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, color=(255, 0, 255), thickness=5)
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
        canvas = np.zeros_like(img)
    else:
        prev_pos = None

    return prev_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # All fingers except thumb up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem ", pil_image])
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = ""

while run and not stop_button:
    success, img = cap.read()
    if not success:
        st.error("Failed to capture image from the webcam.")
        break
    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Remove cv2.waitKey(1) as it's not needed in Streamlit

if cap is not None:
    cap.release()

# Footer with copyright information
st.markdown("""
---
© 2025 Lokesh Chintalapudi - Math Gestures
""")
