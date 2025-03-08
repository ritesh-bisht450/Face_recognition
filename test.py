import cv2
import face_recognition
import time
import json
import numpy as np
from datetime import datetime

curr_date = datetime.now().strftime("%d%m%Y")

def detect_face():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return

    face_detected = False
    start_time = None
    face_encoding = None

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_locations:
            if not face_detected:
                start_time = time.time()
                face_detected = True

            top, right, bottom, left = face_locations[0]  
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]  

            cv2.imshow("Frame", frame)

            if time.time() - start_time >= 3:
                break

        else:
            face_detected = False
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    if face_detected and face_encoding is not None:
        name = input("Enter name: ")
        roll_no = input("Enter your roll number: ")
        
        face_data = {
            "name": name,
            "face_encoding": face_encoding.tolist()  
        }

        try:
            with open('saved_faces.json', 'r') as file:
                content = json.load(file)
        except Exception as e:
            print("No previous data found. Creating new file.")
            content = {}

        content[roll_no] = face_data

        try:
            with open('saved_faces.json', 'w') as file:
                json.dump(content, file, indent=4)  
            print(f"Data saved for roll number {roll_no}.")
        except Exception as e:
            print(f"Error saving data to JSON file: {e}")

    else:
        print("No face was detected.")


def recognize_face():
    try:
        with open('saved_faces.json', 'r') as file:
            content = json.load(file)
    except Exception as e:
        print("Error loading face data:", e)
        return

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Unable to open camera")
        return

    face_detected = False
    start_time = None
    name = None
    roll_no = None

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Couldn't capture the frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_locations:
            if not face_detected:
                face_detected = True
                start_time = time.time()

            top, right, bottom, left = face_locations[0]
            face_encoding = face_encodings[0]

            for roll_no, data in content.items():
                stored_encoding = np.array(data["face_encoding"])

                matches = face_recognition.compare_faces([stored_encoding], face_encoding)

                if matches[0]:
                    name = data['name']
                    roll_no = roll_no
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{data['name']}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if start_time and time.time() - start_time >= 3:
                        capture.release()
                        cv2.destroyAllWindows()

                        try:
                            with open(f"{curr_date}.txt", 'a') as file:
                                file.write(f"{name}   {roll_no}\n")
                        except Exception as e:
                            print(f"Error writing to file: {e}")

                        return

        else:
            face_detected = False
            start_time = None

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


recognize_face()
detect_face()



    
    
        

