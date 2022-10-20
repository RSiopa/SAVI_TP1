#!/usr/bin/env python3
import copy
import cv2
import argparse
from simple_facerec import SimpleFacerec
import shutil
import os
import pyttsx3
from colorama import Fore, Back, Style



def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Setting up the arguments
    parser = argparse.ArgumentParser(description='Face Detector and Recogniser')  # arguments
    parser.add_argument('-ud', '--use_database', action='store_true',
                        help='use the already made database of faces to recognise.\n ')
    args = vars(parser.parse_args())

    # Paths of Databases
    database_path = 'Faces/'
    no_database_path = 'No_Database/'

    sfr = SimpleFacerec()

    # If user wants to use the Database
    if args['use_database']:
        sfr.load_encoding_images(database_path)
    else:
        # Else use an empty folder to not find any images
        sfr.load_encoding_images(no_database_path)

    # Start video
    video = cv2.VideoCapture(0)

    # Initiate the text to speech
    # engine = pyttsx3.init()

    # Show commands
    print('\nProgram Commands')
    print(Fore.BLACK + Back.RED + '\nPress "q" to quit the program')
    print(Fore.BLACK + Back.BLUE + 'Press "d" to show the Database')
    print(Fore.BLACK + Back.YELLOW + 'Press "h" to show these instructions')
    print(Fore.BLACK + Back.GREEN + 'Press "p" to take a picture and add it to Database')

    # Initialize some variables to be used in the program
    error_time_counter = 0
    last_face_names = None
    picture_countdown = 3

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    # Video Cycle
    while True:

        # Capture the video frame by frame
        ret, frame = video.read()

        # Copy original image
        image_gui = copy.deepcopy(frame)

        try:
            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(image_gui)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # Draw rectangle and name
                cv2.putText(image_gui, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prevent the spam a bit
                if not face_names == last_face_names:
                    if name == 'Unknown':
                        # Ask to add face to Database
                        print(Fore.GREEN + 'Hello, I do not know you.\nCould you take a picture with "p" and introduce yourself?')
                    else:
                        print(Fore.GREEN + 'Hello, ' + Fore.BLUE + str(name))
                        # engine.say('Hello, ' + str(name))
                        # engine.runAndWait()

            last_face_names = face_names

        except:
            # If Database directory has no pictures with faces
            if error_time_counter == 0:
                print(Fore.RED + '\nThere are no faces in the Database')
            # Variable error_time_counter prevents spam
            #dar fix ( spam infinito)
            error_time_counter += 1
            if error_time_counter == 10:
                error_time_counter = 0

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
                print(Fore.BLUE + str(name))
            print('--------------------------------------------')
            print(not args['use_database'])

        # The 'h' button is used to show the instructions
        if key == ord('h'):
            print('\nProgram Commands')
            print(Fore.BLACK + Back.RED + '\nPress "q" to quit the program')
            print(Fore.BLACK + Back.BLUE + 'Press "d" to show the Database')
            print(Fore.BLACK + Back.YELLOW + 'Press "h" to show these instructions')
            print(Fore.BLACK + Back.GREEN + 'Press "p" to take a picture and add it to Database')

        # The 'p' button is used to take a picture and add it to the Database
        if key == ord('p') or picture_countdown != 3:
            # Shows countdown
            print(picture_countdown)
            picture_countdown -= 1
            if picture_countdown == 0:
                cv2.imshow('image', frame)
                print('Snap!')
                new_name = input("What is your name? ")
                # After adding a new picture, loads the pictures in the folder again
                if args['use_database']:
                    # Using the Database, only need to write image in Database
                    cv2.imwrite(os.path.join(database_path, new_name + '.png'), frame)
                    sfr.load_encoding_images(database_path)
                else:
                    # Not using Database, image will be added to the empty Database
                    cv2.imwrite(os.path.join(no_database_path, new_name + '.png'), frame)
                    sfr.load_encoding_images(no_database_path)

                # Return picture_countdown to original value
                picture_countdown = 3

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    if not args['use_database']:
        # Delete directory
        shutil.rmtree('No_Database')
        # Recreate directory
        os.makedirs('No_Database')

    # After the loop release the cap object
    video.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
