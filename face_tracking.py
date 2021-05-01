import cv2
import os
from PIL import Image
from inference.fer import FER

ckpt_path = './checkpoints/epoch=013-val_loss=0.385.ckpt'
fer = FER(ckpt_path)

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    faces = faceCascade.detectMultiScale(
        frames,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

        crop_img = frames[y:y+h, x:x+w]

        # Predict the emotion
        pil_image = Image.fromarray(crop_img)
        pred = fer.predict(pil_image)

        # Print the predicted emotion above the face rectangle
        cv2.putText(img, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()