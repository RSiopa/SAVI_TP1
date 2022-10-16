#!/usr/bin/env python3
import copy
from copy import deepcopy
import cv2
import argparse
import numpy as np
import csv
import argparse
import face_recognition
from simple_facerec import SimpleFacerec


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    parser = argparse.ArgumentParser(description='Face Detector and Recogniser')  # arguments
    parser.add_argument('-ud', '--use_database', type=bool, required=True,
                        help='use the already made database of faces to recognise.\n ')
    args = vars(parser.parse_args())

    # Load the cascade
    # face_cascade = cv2.CascadeClassifier('../SAVI TP1/haarcascade_frontalface_default.xml')

    # face_database = []
    sfr = SimpleFacerec()

    if args['use_database']:
        sfr.load_encoding_images("Faces/")
        # face_rafael = cv2.imread('../SAVI TP1/Rafael.jpg')
        # rgb_face_rafael = cv2.cvtColor(face_rafael, cv2.COLOR_BGR2RGB)
        # face_rafael_encode = face_recognition.face_encodings(rgb_face_rafael)[0]
        # face_database.append(face_rafael_encode)
    else:
        sfr.load_encoding_images("Made_Up_File_Name")

    video = cv2.VideoCapture(0)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    print('\nPress "q" to quit the program')
    print('Press "d" to show the Database')
    print('Press "h" to show these instructions')

    while True:

        # result = False

        # Capture the video frame by frame
        ret, frame = video.read()
        image_gui = copy.deepcopy(frame)

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(image_gui)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(image_gui, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Make a copy of the original image
        # image_gui = deepcopy(image)

        # Convert to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(250, 250))

        # Draw the rectangle around each face
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image_gui, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #
        # print(faces)

        # print(face_database)

        # if faces is not None:
        #     image_gui_rgb = cv2.cvtColor(image_gui, cv2.COLOR_BGR2RGB)
        #     image_gui_encoded = face_recognition.face_encodings(image_gui_rgb)[0]
        #     # result = face_recognition.compare_faces([face_rafael_encode], image_gui_encoded, tolerance=0.6)

        # if result:
        #     cv2.putText(image_gui, 'Rafael Siopa', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        #                 cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('image', image_gui)

        key = cv2.waitKey(1)

        # The 'q' button is set as the quitting button
        if key == ord('q'):
            break

        # The 'd' button is used to show the database
        if key == ord('d'):
            print('\nFace Recognition Database')
            print('--------------------------------------------')
            for name in zip(face_names):
                print(name)

            print('--------------------------------------------')

        # The 'h' button is used to show the instructions
        if key == ord('h'):
            print('\nPress "q" to quit the program')
            print('Press "d" to show the Database')
            print('Press "h" to show these instructions')

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
