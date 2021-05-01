# Code by Karthik Murugesan (Github: @karthikmuru)

import cv2
import os
import argparse
from PIL import Image
from inference.fer import FER

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
      "--checkpoint_path", type=str, required=True, help="Path to the model checkpoint."
    )

    return parser

def start_tracking(predictor):
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
            pred = predictor.predict(pil_image)

            # Print the predicted emotion above the face rectangle
            cv2.putText(img, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
        # Display the resulting frame
        cv2.imshow('Video', frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args.checkpoint_path)
    predictor = FER(args.checkpoint_path)
    
    start_tracking(predictor)

if __name__ == "__main__":
    main()