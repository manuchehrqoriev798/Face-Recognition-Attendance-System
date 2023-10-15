# # Database: Name
# from deepface import DeepFace
# import tensorflow as tf
# import cv2
# import numpy as np
# import datetime
# import csv  # Import csv module for working with CSV files
# import os

# model = 'liveness.model'
# model = tf.keras.models.load_model(model)
# cap = cv2.VideoCapture(1)

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
#                 cv2.putText(frame, "real", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#                 # Append attendance to a CSV file
#                 with open('attendance.csv', mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     writer.writerow([name, datetime.date.today()])

#             else:
#                 cv2.putText(frame, "Adopted, sorry it is not your bad", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

#         else:
#             name = "Unknown"
#             cv2.imshow('attendance', frame)

#     cv2.imshow('attendance', frame)
#     c = cv2.waitKey(1)

#     if c == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()















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

while True:
    state, frame = cap.read()

    if not state:
        break

    res = DeepFace.find(frame, db_path='./Database/', enforce_detection=False, model_name='Facenet512')

    if res:
        if len(res[0]['identity']) > 0:
            image_path = res[0]['identity'][0]
            folder_name = os.path.dirname(image_path)
            name = os.path.basename(folder_name)
            # Extract class information from the folder structure (if applicable)
            class_name = os.path.basename(folder_name)
            
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

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.rectangle(frame, (xmin, ymin - 25), (xmax, ymin), (255, 255, 255), -1)
            cv2.putText(frame, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if liveness == 1:
                cv2.putText(frame, "real", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Append attendance to a CSV file
                with open('attendance.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{name}: Class of {class_name}" if class_name else name, datetime.date.today()])

            else:
                cv2.putText(frame, "Adopted, sorry it is not your bad", (xmin, ymax + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            name = "Unknown"
            cv2.imshow('attendance', frame)

    cv2.imshow('attendance', frame)
    c = cv2.waitKey(1)

    if c == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
