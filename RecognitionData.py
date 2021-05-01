import cv2
import numpy as np
import os
import sqlite3
from PIL import Image
import webbrowser

# trainning hinh anh nhan dien voi thu vien nhan dien khuon mat
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('D:\Python\getDataFromWebcam\\recognizer\\trainningData.yml')


# get profile by id from database
def getProfile(id):
    conn = sqlite3.connect('D:\Python\getDataFromWebcam\data.db')
    query = "SELECT * FROM people WHERE ID=" + str(id)
    cursor = conn.execute(query)

    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fontface = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        id, confidence = recognizer.predict(roi_gray)

        if confidence < 40:
            profile = getProfile(id)
            if profile != None:
                cv2.putText(frame, "" + str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                if not os.path.exists('dataLog'):
                    os.makedirs('dataLog')
                f = open('dataLog/log', 'w', encoding="utf-8")
                f.write(str(profile[1]) + " đã điểm danh")
                f.close()
                url = "http://localhost:8080/portal"
                webbrowser.open_new_tab(url)
                quit()

        else:
            cv2.putText(frame, "Unknow", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)

        cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
