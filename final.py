import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize mediapipe for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize mediapipe for face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen size
screen_w, screen_h = pyautogui.size()

# Constants for hand tracking paint application
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_names = ['BLUE', 'GREEN', 'RED', 'YELLOW']
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
colorIndex = 0
pencil_size = 2

# Sensitivity parameters for mouse movement
mouse_speed_x = 3
mouse_speed_y = 3

# Callback function for trackbar
def set_pencil_size(x):
    global pencil_size
    pencil_size = x

# Create a window for the paint application
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Pencil Size', 'Paint', pencil_size, 20, set_pencil_size)

while True:
    # Read frame from the webcam
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Convert frame to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh to get eye and face landmarks
    face_output = face_mesh.process(rgb_frame)
    landmark_points = face_output.multi_face_landmarks

    # Process hand tracking to get hand landmarks
    hand_output = hands.process(rgb_frame)

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        if len(landmarks) >= 478:  # Ensure enough landmarks are detected
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        
        left = []
        if len(landmarks) >= 160:  # Ensure enough landmarks are detected
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
                
            if (left[0].y - left[1].y) < 0.003:
                pyautogui.click()
                pyautogui.sleep(0.2)

    # Process hand tracking output
    if hand_output.multi_hand_landmarks:
        hand_landmarks = hand_output.multi_hand_landmarks[0].landmark
        fore_finger = (int(hand_landmarks[8].x * frame_w), int(hand_landmarks[8].y * frame_h))
        center = fore_finger
        thumb = (int(hand_landmarks[4].x * frame_w), int(hand_landmarks[4].y * frame_h))
        cv2.circle(frame, center, pencil_size, (0, 255, 0), -1)
        
        if (thumb[1] - center[1] < 30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    # Draw colored boxes with color names
    for i, color in enumerate(colors):
        cv2.rectangle(paintWindow, (40 + i * 115, 1), (140 + i * 115, 65), color, -1)
        cv2.putText(paintWindow, color_names[i], (60 + i * 115, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], pencil_size)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], pencil_size)

    # Display the frame and paint window
    cv2.imshow('Eye Controlled Mouse & Hand Tracking Paint', frame)
    cv2.imshow('Paint', paintWindow)

    # Check for exit key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()

