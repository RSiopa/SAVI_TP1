#!/usr/bin/env python3
import copy
import cv2
import argparse
from simple_facerec import SimpleFacerec, Detection, Tracker
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
    parser.add_argument('-ut', '--use_text_to_speech', action='store_true',
                        help='use text to speech to greet known and detected users.\n ')
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
    if video.isOpened() is False:
        print("Error opening video stream or file")

    # Initiate the text to speech
    if args['use_text_to_speech']:
        engine = pyttsx3.init()

    # Show commands
    print('\nProgram Commands')
    print(Fore.BLACK + Back.RED + '\nPress "q" to quit the program' + Style.RESET_ALL)
    print(Fore.BLACK + Back.BLUE + 'Press "d" to show the Database' + Style.RESET_ALL)
    print(Fore.BLACK + Back.YELLOW + 'Press "h" to show these instructions' + Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN + 'Press "p" to take a picture and add it to Database' + Style.RESET_ALL)

    # Initialize some variables to be used in the program
    # error_time_counter = 0
    last_face_names = []
    picture_countdown = 3
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8
    no_face_counter = 0

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    # Video Cycle
    frame_counter = 0
    while video.isOpened():

        # Capture the video frame by frame
        ret, frame = video.read()
        stamp = float(video.get(cv2.CAP_PROP_POS_MSEC))/1000
        if not ret:
            break

        # Copy original image
        image_gui = copy.deepcopy(frame)
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Detect Faces
            detections = []
            face_locations, face_names = sfr.detect_known_faces(image_gui)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # ------------------------------------------
                # Create Detections per face detected
                # ------------------------------------------
                detection = Detection(x1, y1, x2, y2, image_gray, id=detection_counter, name=name, stamp=stamp)
                detection_counter += 1
                detection.draw(image_gui)
                detections.append(detection)
                # Draw rectangle and name
                #cv2.putText(image_gui, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                #cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prevent the spam a bit
                if not face_names == last_face_names:

                    if name == 'Unknown':
                        # Ask to add face to Database
                        print(Fore.GREEN + 'Hello, I do not know you.\nCould you take a picture with "p" and introduce yourself?')
                        print(Style.RESET_ALL)
                  
                    else:
                        if not args['use_text_to_speech']:
                            print(Fore.GREEN + 'Hello, ' + Fore.BLUE + str(name))
                            print(Style.RESET_ALL)

                        if args['use_text_to_speech']:
                            engine.say('Hello, ' + str(name))
                            engine.runAndWait()

            # Also help with spam
            if face_names:
                last_face_names = face_names

        except:
            # If Database directory has no pictures with faces
            if no_face_counter == 0:
                print(Fore.RED + '\nThere are no faces in the Database' + Style.RESET_ALL)
                no_face_counter = 1

        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections:  # cycle all detections
            for tracker in trackers:  # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                # print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold:  # associate detection with tracker
                    tracker.addDetection(detection, image_gray)

        # ------------------------------------------
        # Track using template matching
        # ------------------------------------------
        for tracker in trackers:  # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            #print(last_detection_id)
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                print('Tracker ' + str(tracker.id) + ' Doing some tracking')
                tracker.track(image_gray)

        # ------------------------------------------
        # Deactivate Tracker if no detection for more than T
        # ------------------------------------------
        for tracker in trackers:  # cycle all trackers
            tracker.updateTime(stamp)

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        for detection in detections:
            if not detection.assigned_to_tracker:
                tracker = Tracker(detection, id=tracker_counter, image=image_gray)
                tracker_counter += 1
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------
        for tracker in trackers:
            tracker.draw(image_gui)

        # Display the resulting image
        cv2.imshow('Face Recognition Software', image_gui)
        frame_counter += 1

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
                print(Style.RESET_ALL)
            print('--------------------------------------------')
            print(not args['use_database'])

        # The 'h' button is used to show the instructions
        if key == ord('h'):
            print('\nProgram Commands')
            print(Fore.BLACK + Back.RED + '\nPress "q" to quit the program' + Style.RESET_ALL)
            print(Fore.BLACK + Back.BLUE + 'Press "d" to show the Database' + Style.RESET_ALL)
            print(Fore.BLACK + Back.YELLOW + 'Press "h" to show these instructions' + Style.RESET_ALL)
            print(Fore.BLACK + Back.GREEN + 'Press "p" to take a picture and add it to Database' + Style.RESET_ALL)

        # The 'p' button is used to take a picture and add it to the Database
        if key == ord('p') or picture_countdown != 3:
            # Shows countdown
            print(picture_countdown)
            picture_countdown -= 1
            if picture_countdown == 0:
                cv2.imshow('Face Recognition Software', frame)
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
