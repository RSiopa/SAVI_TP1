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

    # Setting up the arguments
    parser = argparse.ArgumentParser(description='Face Detector and Recogniser')  # arguments
    parser.add_argument('-ud', '--use_database', type=bool, required=True,
                        help='use the already made database of faces to recognise.\n ')
    args = vars(parser.parse_args())

    sfr = SimpleFacerec()

    # If user wants to use the Database
    if args['use_database']:
        sfr.load_encoding_images("Faces/")
    else:
        # Else use a wrong file name to not find any images
        sfr.load_encoding_images("Made_Up_File_Name")

    # Start video
    video = cv2.VideoCapture(0)

    # Show commands
    print('\nProgram Commands')
    print('\nPress "q" to quit the program')
    print('Press "d" to show the Database')
    print('Press "h" to show these instructions')

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    # Video Cycle
    while True:

        # Capture the video frame by frame
        ret, frame = video.read()

        # Copy original image
        image_gui = copy.deepcopy(frame)

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(image_gui)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Draw rectangle and name
            cv2.putText(image_gui, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('image', image_gui)

        # Wait for the video
        key = cv2.waitKey(1)

        # The 'q' button is set as the quitting button
        if key == ord('q'):
            break

        # The 'd' button is used to show the database
        if key == ord('d'):
            print('\nFace Recognition Database')
            print('--------------------------------------------')
            for name in face_names:
                print(str(name))
            print('--------------------------------------------')

        # The 'h' button is used to show the instructions
        if key == ord('h'):
            print('\nProgram Commands')
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
