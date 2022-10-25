#!/usr/bin/env python3

import face_recognition
import cv2
import numpy as np
import csv
import pickle
from colorama import Fore, Back, Style

class BoundingBox:
    
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.w = self.x2 - self.x1
        self.h = self.y2 - self.y1

        self.area = self.w * self.h

    def computeIOU(self, bbox2):

        x1_intr = min(self.x1, bbox2.x1)
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr

        A_union = self.area + bbox2.area - A_intr
        
        return A_intr / A_union

    def extractSmallImage(self, image_full):
        return image_full[self.y1:self.y2, self.x1:self.x2]


class Detection(BoundingBox):

    def __init__(self, x1, y1, x2, y2, image_full, id, name, stamp):
        super().__init__(x1, y1, x2, y2) # call the super class constructor        
        self.id = id
        self.name = name
        self.stamp = stamp
        self.image = self.extractSmallImage(image_full)
        self.assigned_to_tracker = False

    def draw(self, image_gui, color=(255, 0, 0)):
        1==1
        cv2.rectangle(image_gui, (self.x1, self.y1), (self.x2, self.y2), color, 3)

        image = cv2.putText(image_gui, 'D' + str(self.id) + ' ' + self.name, (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)


class Tracker:

    def __init__(self, detection, id, name, image):
        self.id = id
        self.name = name
        self.template = None
        self.active = True
        self.bboxes = []
        self.detections = []
        self.tracker = cv2.TrackerCSRT_create()
        self.time_since_last_detection = None

        self.addDetection(detection, image, name)

    def getLastDetectionStamp(self):
        return self.detections[-1].stamp

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(), 1)

        if self.time_since_last_detection > 2:  # deactivate tracker
            self.active = False

    def drawLastDetection(self, image_gui, color=(255, 0, 255)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui, (last_detection.x1, last_detection.y1),
                      (last_detection.x2, last_detection.y2), color, 3)

        image = cv2.putText(image_gui, 'T' + str(self.id),
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)

    def draw(self, image_gui, color=(255, 0, 255)):

        if self.active:

            bbox = self.bboxes[-1]  # get last bbox

            cv2.rectangle(image_gui, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 3)

            # cv2.putText(image_gui, 'T' + str(self.id),
            #                     (bbox.x2-40, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX,
            #                 1, color, 2, cv2.LINE_AA)

            # cv2.putText(image_gui, str(self.time_since_last_detection) + ' s',
            #                     (bbox.x2-40, bbox.y1-25), cv2.FONT_HERSHEY_SIMPLEX,
            #                 1, color, 2, cv2.LINE_AA)

            cv2.putText(image_gui, self.name, (bbox.x1, bbox.y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2, cv2.LINE_AA)

    def addDetection(self, detection, image, name):

        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))
        self.name = name
        self.detections.append(detection)
        detection.assigned_to_tracker = True
        self.template = detection.image
        bbox = BoundingBox(detection.x1, detection.y1, detection.x2, detection.y2)
        self.bboxes.append(bbox)

    def track(self, image):

        ret, bbox = self.tracker.update(image)
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        # print(bbox)
        bbox = BoundingBox(x1, y1, x2, y2)

        self.bboxes.append(bbox)

        # Update template using new bbox coordinates
        self.template = bbox.extractSmallImage(image)
        
    def __str__(self):
        text = 'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        

        # Resize frame for a faster speed
        self.frame_resizing = 0.1

    def save_encondings(self, encodings_path, name, encodings):
        
        try:
            with open(encodings_path, 'rb') as f:
                face_encodings = pickle.load(f) 
        except:
            face_encodings = {}

        face_encodings[name] = ' '.join(map(str, encodings))

        # print(face_encodings)
        with open(encodings_path, 'wb') as u:
            pickle.dump(face_encodings, u)

    def load_encodings(self, encodings_path):
        
        try:
            self.known_face_encodings = []
            # Load face encodings
            with open(encodings_path, 'rb') as f:
                all_face_encodings = pickle.load(f)
            # print(all_face_encodings)
            # Grab the list of names and the list of encodings
            face_names = list(all_face_encodings.keys())
            
            face_encodings = list(all_face_encodings.values())
            for encoding in face_encodings:
                face_encoding = face_encodings[face_encodings.index(encoding)].split(' ')
                face_encoding = list(map(float,face_encoding))
                self.known_face_encodings.append(face_encoding)
            # print(self.known_face_encodings)
            self.known_face_names = face_names
            
            print(Fore.GREEN + "Encodings loaded!" + Style.RESET_ALL)
            # print(self.known_face_encodings)
        except:
            print(Fore.RED + 'Nothing yet on the data base!' + Style.RESET_ALL)

    def simpleFace_detector(selfe, frame):
        encodings = face_recognition.face_encodings(frame,model = "small")
        # print(encodings)
        return encodings

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model = "small")
        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            
            if not self.known_face_encodings == []:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, 0.6)

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
            else:    
                face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        
        return face_locations.astype(int), face_names, face_encodings
