#!/usr/bin/env python3

from copy import deepcopy
import cv2
import argparse
import numpy as np
import csv


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('../SAVI TP1/haarcascade_frontalface_default.xml')

    video = cv2.VideoCapture(0)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    while True:

        # Capture the video frame by frame
        ret, image = video.read()

        # Make a copy of the original image
        image_gui = deepcopy(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(250, 250))

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(image_gui, (x, y), (x + w, y + h), (255, 0, 0), 2)
            image = image_full[self.y1:self.y1 + self.h, self.x1:self.x1 + self.w]

        # Display the resulting frame
        cv2.imshow('image', image_gui)

        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
