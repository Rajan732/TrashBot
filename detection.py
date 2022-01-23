import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow import saved_model
from tensorflow.keras.preprocessing import image
import numpy as np
import mediapipe as mp
import time
pTime = 0
cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model("trash1.h5")
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
l = []

while True:
    success, frame = cap.read()
    test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    new_image = cv2.resize(test_image, (60,60))
    image_array = image.img_to_array(new_image)
    final_image = np.expand_dims(image_array, axis = 0)
    # model.summary()
    result = model.predict(final_image)
    #print(result)
   
    
    results = pose.process(test_image)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                     
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
