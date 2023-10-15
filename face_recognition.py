# Database folder which folders with the name of Students and inside the photo of Students 
# from deepface import DeepFace
# import tensorflow as tf
# import cv2
# import numpy as np
# import datetime
# import csv
# import os

# model = 'liveness.model'
# model = tf.keras.models.load_model(model)
# cap = cv2.VideoCapture(0)

# recognized_faces = []  # List to store recognized faces
# absent_students = set()  # Set to store absent students

# # Get the current date
# current_date = datetime.date.today()

# # Create and open the CSV file with a header
# with open('attendance.csv', mode='a', newline='') as file:
#     writer = csv.writer(file)
#     # Check if the CSV file is empty
#     is_csv_empty = os.path.getsize('attendance.csv') == 0
#     if not is_csv_empty:
#         for _ in range(5): 
#             file.write('\n')
#     writer.writerow(['Attendance for ' + current_date.strftime('%d %B, %Y')])
#     writer.writerow(['List of students who are present:'])

# while True:
#     state, frame = cap.read()

#     if not state:
#         break

#     res = DeepFace.find(frame, db_path='./Database/', enforce_detection=False, model_name='Facenet512')

#     if res:
#         if len(res[0]['identity']) > 0:
#             image_path = res[0]['identity'][0]
#             folder_name = os.path.dirname(image_path)
#             name = os.path.basename(folder_name)

#             xmin = int(res[0]['source_x'][0])
#             ymin = int(res[0]['source_y'][0])
#             w = res[0]['source_w'][0]
#             h = res[0]['source_h'][0]
#             xmax = int(xmin + w)
#             ymax = int(ymin + h)

#             face_img = frame[ymin:ymax, xmin:xmax]
#             face_img = cv2.resize(face_img, (32, 32))
#             face_img = face_img.astype('float') / 255.0
#             face_img = tf.keras.preprocessing.image.img_to_array(face_img)
#             face_img = np.expand_dims(face_img, axis=0)

#             liveness = model.predict(face_img)
#             liveness = liveness[0].argmax()

#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
#             cv2.rectangle(frame, (xmin, ymin - 25), (xmax, ymin), (255, 255, 255), -1)
#             cv2.putText(frame, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#             if liveness == 1:
#                 cv2.putText(frame, "Real", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 # Check if the face has already been recognized
#                 if name not in recognized_faces:
#                     recognized_faces.append(name)

#                     # Append attendance to the CSV file with timestamp
#                     with open('attendance.csv', mode='a', newline='') as file:
#                         writer = csv.writer(file)
#                         now = datetime.datetime.now()
#                         timestamp = now.strftime('%H:%M:%S')
#                         writer.writerow([name, timestamp])

#                     # Display present students on the screen
#                     print(f'{len(recognized_faces)}. {name} - {timestamp}')

#             else:
#                 cv2.putText(frame, "Adopted, sorry it is not your bad", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#         else:
#             name = "Unknown"
#             cv2.imshow('attendance', frame)

#             # Add unknown students to the absent students set
#             absent_students.add(name)

#     cv2.imshow('attendance', frame)
#     c = cv2.waitKey(1)

#     if c == ord('q'):
#         # Add code here to handle the 'q' key press
#         break  # To exit the loop


# cap.release()
# cv2.destroyAllWindows()

# # Initialize a set to store recognized students
# recognized_students = set(recognized_faces)

# # Iterate through the folders in the 'Database' directory
# database_path = './Database/'
# for folder_name in os.listdir(database_path):
#     if folder_name == 'representations_facenet512.pkl' or folder_name == 'Unknown':
#         continue

#     student_name = folder_name
#     if student_name not in recognized_students:
#         absent_students.add(student_name)

# # Write absent students to the CSV file
# with open('attendance.csv', mode='a', newline='') as file:
#     writer = csv.writer(file)

#     for i, student in enumerate(absent_students, start=1):
#         writer.writerow([f"{i}. {student}"])































# Database: Class: Name
from deepface import DeepFace
import tensorflow as tf
import cv2
import numpy as np
import datetime
import csv
import os

model = 'liveness.model'
model = tf.keras.models.load_model(model)
cap = cv2.VideoCapture(1)

recognized_faces = []  # List to store recognized faces
absent_students = set()  # Set to store absent students

# Get the current date
current_date = datetime.date.today()
date_str = current_date.strftime('%d.%m.%Y')

# Create and open the CSV file with a header
with open('attendance.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Check if the CSV file is empty
    is_csv_empty = os.path.getsize('attendance.csv') == 0
    if not is_csv_empty:
        for _ in range(5):
            file.write('\n')
    writer.writerow(['Attendance of the student on ' + date_str])
    writer.writerow(['Present Students list:'])

while True:
    state, frame = cap.read()

    if not state:
        break

    res = DeepFace.find(frame, db_path='./Database/', enforce_detection=False, model_name='Facenet512')

    if res:
        if len(res[0]['identity']) > 0:
            image_path = res[0]['identity'][0]
            student_folder = os.path.dirname(image_path)
            class_folder = os.path.dirname(student_folder)
            student_name = os.path.basename(student_folder)
            class_name = os.path.basename(class_folder)

            timestamp = datetime.datetime.now().strftime('%H:%M:%S')

            # Add student to recognized list
            if student_name not in recognized_faces:
                recognized_faces.append(student_name)

                # Append attendance to the CSV file with timestamp
                with open('attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{len(recognized_faces)}. {student_name}: Class of {class_name} - {timestamp}"])

                # Display present students on the screen
                print(f'{len(recognized_faces)}. {student_name}: Class of {class_name} - {timestamp}')

            # Draw a blue rectangle with the name and "Real" or "Fake" text
            xmin = int(res[0]['source_x'][0])
            ymin = int(res[0]['source_y'][0])
            w = res[0]['source_w'][0]
            h = res[0]['source_h'][0]
            xmax = int(xmin + w)
            ymax = int(ymin + h)

            face_img = frame[ymin:ymax, xmin:xmax]
            face_img = cv2.resize(face_img, (32, 32))
            face_img = face_img.astype('float') / 255.0
            face_img = tf.keras.preprocessing.image.img_to_array(face_img)
            face_img = np.expand_dims(face_img, axis=0)

            liveness = model.predict(face_img)
            liveness = liveness[0].argmax()

            # Draw a green rectangle around the face
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            # Draw a blue rectangle for the name
            cv2.rectangle(frame, (xmin, ymin - 25), (xmax, ymin), (255, 0, 0), -1)

            # Add your name inside the blue rectangle
            cv2.putText(frame, student_name, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Add "Real" or "Fake" text in blue
            if liveness == 1:
                cv2.putText(frame, "Real", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Adopted, sorry it is not your bad", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('attendance', frame)
    c = cv2.waitKey(1)

    if c == ord('q'):
        # Add code here to handle the 'q' key press
        break  # To exit the loop


cap.release()
cv2.destroyAllWindows()



# Initialize a set to store recognized students
recognized_students = set(recognized_faces)

# Iterate through the folders in the 'Database' directory
database_path = './Database/'

# Create a variable to keep track of the numbering for absent students
absent_number = 1

with open('attendance.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Absent Student list:'])

for class_name in os.listdir(database_path):
    if class_name == 'representations_facenet512.pkl' or class_name == 'Unknown':
        continue

    class_folder = os.path.join(database_path, class_name)

    for student_name in os.listdir(class_folder):
        student_folder = os.path.join(class_folder, student_name)

        if student_name not in recognized_students:
            with open('attendance.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"{absent_number}. {student_name}: Class of {class_name}"])
            
            # Increment the absent student numbering
            absent_number += 1

# Display absent students
for i, student in enumerate(absent_students, start=len(recognized_faces) + 1):
    print(f"{i}. {student}: Class of {class_name}")