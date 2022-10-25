#!/usr/bin/env python3

import copy
import cv2
import argparse
from simple_facerec import SimpleFacerec, Detection, Tracker, pictureDetection
import shutil
import os
import pyttsx3
from colorama import Fore, Back, Style
import time


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
    database_path = 'faces_database.data'
    no_database_path = 'temporary_database.data'

    try:
        f = open(database_path, 'rb')
        f.close()
    except:
        f = open(database_path, 'xb')
        f.close


    sfr = SimpleFacerec()

    # If user wants to use the Database
    if args['use_database']:
        sfr.load_encodings(database_path)
        print(sfr.known_face_names)
    else:
        try:
            f = open(no_database_path, 'x')
            f.close        
            sfr.load_encodings(no_database_path)
        except:
            sfr.load_encodings(no_database_path)

    # Start video
    video = cv2.VideoCapture(0)
    if video.isOpened() is False:
        print("Error opening video stream or file")

    # Initiate the text to speech
    if args['use_text_to_speech']:
        engine = pyttsx3.init()

    # Show commands
    print(Style.BRIGHT + '\nProgram Commands' + Style.RESET_ALL)
    print('\nPress "' + Fore.RED + 'q' + Style.RESET_ALL + '" to ' + Fore.RED + 'quit the program' + Style.RESET_ALL)
    print('Press "' + Fore.BLUE + 'd' + Style.RESET_ALL + '" to ' + Fore.BLUE + 'show the Database' + Style.RESET_ALL)
    print('Press "' + Fore.YELLOW + 'h' + Style.RESET_ALL + '" to ' + Fore.YELLOW + 'show these instructions' + Style.RESET_ALL)
    print('Press "' + Fore.GREEN + 'p' + Style.RESET_ALL + '" to ' + Fore.GREEN + 'take a picture and add it to Database\n' + Style.RESET_ALL)

    # Initialize some variables to be used in the program
    # error_time_counter = 0
    last_face_names = []
    picture_countdown = 3
    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8
    no_face_counter = 0
    face_name_counter = 0

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
            face_locations, face_names, face_encodings = sfr.detect_known_faces(image_gui)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # ------------------------------------------
                # Create Detections per face detected
                # ------------------------------------------
                detection = Detection(x1, y1, x2, y2, image_gray, id=detection_counter, name=name, stamp=stamp)
                detection_counter += 1
                # detection.draw(image_gui)
                detections.append(detection)
                # Draw rectangle and name
                #cv2.putText(image_gui, name, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                #cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prevent the spam a bit
                if not face_names == last_face_names:

                    if name == 'Unknown':
                        # Ask to add face to Database
                        print(Fore.GREEN + 'Hello, I do not know you.\nCould you take a picture with "p" and introduce yourself?' + Style.RESET_ALL)
                    
                    else:
                        if not args['use_text_to_speech']:
                            print(Fore.GREEN + '\nHello, ' + Fore.BLUE + str(name))
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
                    tracker.addDetection(detection, image_gray, detection.name)
                    tracker.active = True
        # ------------------------------------------
        # Tracking Using TrackerCSRT
        # ------------------------------------------
        for tracker in trackers:  # cycle all trackers
            last_detection_id = tracker.detections[-1].id
            #print(last_detection_id)
            detection_ids = [d.id for d in detections]
            if not last_detection_id in detection_ids:
                # print('Tracker ' + str(tracker.id) + ' Doing some tracking')
                tracker.track(image_gray)

        # ------------------------------------------
        # Update trackers
        # ------------------------------------------
        for tracker in trackers:  # cycle all trackers
            tracker.updateTime(stamp)

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        for detection in detections:
            # With more people names bug out, need to know whats happening
            if not detection.assigned_to_tracker:
                # print(face_names)
                tracker = Tracker(detection, id=tracker_counter, name=face_names[face_name_counter], image=image_gray)
                tracker_counter += 1
                face_name_counter += 1
                trackers.append(tracker)

        face_name_counter = 0

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
            print(Style.BRIGHT + '\nProgram Commands' + Style.RESET_ALL)
            print('\nPress "' + Fore.RED + 'q' + Style.RESET_ALL + '" to ' + Fore.RED + 'quit the program' + Style.RESET_ALL)
            print('Press "' + Fore.BLUE + 'd' + Style.RESET_ALL + '" to ' + Fore.BLUE + 'show the Database' + Style.RESET_ALL)
            print('Press "' + Fore.YELLOW + 'h' + Style.RESET_ALL + '" to ' + Fore.YELLOW + 'show these instructions' + Style.RESET_ALL)
            print('Press "' + Fore.GREEN + 'p' + Style.RESET_ALL + '" to ' + Fore.GREEN + 'take a picture and add it to Database\n' + Style.RESET_ALL)

        # The 'p' button is used to take a picture and add it to the Database
        if key == ord('p') or picture_countdown != 3:
            # Shows countdown
            print(picture_countdown)
            picture_countdown -= 1
            if picture_countdown == 0:
                if len(sfr.simpleFace_detector(image_gui)) == 0:
                    print('No faces detected, try again!')
                    picture_countdown = 3
                # elif len(sfr.simpleFace_detector(image_gui)) == 1:
                #     encoding = sfr.simpleFace_detector(frame)
                #     # print(encoding)
                #     image_gui = copy.deepcopy(frame)
                #     # Draw rectangle and name
                #     cv2.putText(image_gui, 'Who is this?', (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                #     cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                #     cv2.imshow('Picture for Database', image_gui)
                #     cv2.waitKey(30)
                #     new_name = input("What is your name? ")
                #     # Adds encoding and name to data base
                #     if args['use_database']:
                #         sfr.save_encondings(database_path, new_name, encoding)
                #         sfr.load_encodings(database_path)
                #     else:
                #         # Not using Database, image will be added to the empty Database
                #         sfr.save_encondings(no_database_path, new_name, encoding)
                #         sfr.load_encodings(no_database_path)

                #     # Return picture_countdown to original value
                #     picture_countdown = 3
                #     cv2.destroyWindow('Picture for Database')
                else:

                    # Detect Faces
                    face_locations, face_names, face_encodings = sfr.detect_known_faces(image_gui)
                    
                    # Detect Faces0
                    face_locations, face_names, face_encodings = sfr.detect_known_faces(image_gui)
                    i = 0
                    for face_loc, name in zip(face_locations, face_names):
                        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                        if name == 'Unknown':
                            image_gui = copy.deepcopy(frame)
                            # Draw rectangle and name
                            cv2.putText(image_gui, 'Who is this?', (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
                            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (0, 255, 0), 2)
                           
                            cv2.imshow('Picture for Database', image_gui)
                            cv2.waitKey(30)
                        
                            # Ask to add face to Database
                            new_name = input(Fore.GREEN + 'What is the name of the person inside the box? ' + Style.RESET_ALL)
                            if args['use_database']:
                                # print(face_encodings[i])
                                sfr.save_encondings(database_path, new_name, face_encodings[i])
                            else:
                                # Not using Database, image will be added to the empty Database
                                sfr.save_encondings(no_database_path, new_name, face_encodings[i])
                                sfr.load_encodings(no_database_path)
                        i += 1
                    cv2.destroyWindow('Picture for Database')
                    sfr.load_encodings(database_path)
                    picture_countdown = 3         
 

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    if not args['use_database']:
        os.remove(no_database_path)

    # After the loop release the cap object
    video.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
